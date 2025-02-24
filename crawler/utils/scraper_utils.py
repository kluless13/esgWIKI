from crawl4ai import BrowserConfig, ExtractionStrategy
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import os
import requests
import re
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .vector_store import ESGVectorStore

# Initialize Groq client settings
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

class ESGExtractionStrategy(ExtractionStrategy):
    def __init__(self, model: str, temperature: float, prompt: str):
        self.name = "ESG Metrics Extraction"
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.prompt = prompt
        self.last_request_time = 0
        self.min_request_interval = 2.0
        
        # Initialize vector store
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.vector_store = ESGVectorStore(base_dir)
        
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type((requests.exceptions.HTTPError, requests.exceptions.RequestException))
    )
    def _call_groq_api(self, payload: dict, headers: dict) -> dict:
        """Make API call to Groq with rate limiting and retries"""
        try:
            # Implement rate limiting
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last_request
                print(f"Rate limiting: Waiting {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
            self.last_request_time = time.time()
            
            # Check for rate limit response
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 5))
                print(f"Rate limited by Groq API. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                raise requests.exceptions.HTTPError("Rate limited", response=response)
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit
                print(f"Rate limit error: {str(e)}")
            elif e.response.status_code >= 500:  # Server error
                print(f"Groq API server error: {str(e)}")
            else:
                print(f"HTTP error: {str(e)}")
            raise
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error: {str(e)}")
            raise
        except requests.exceptions.Timeout as e:
            print(f"Timeout error: {str(e)}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"Request error: {str(e)}")
            raise
        
    def _extract_metric_contexts(self, text: str) -> str:
        """Extract relevant context around key metrics"""
        contexts = []
        
        # Enhanced patterns for emissions data in various formats
        metric_patterns = [
            # General emissions table pattern with flexible structure
            (r'(?:Table|TABLE).*?(?:GHG|Greenhouse Gas|Carbon|Emissions).*?\n(?:[^\n]*\n){0,10}?(?:.*?(?:Scope|Total|Energy|Emissions).*?\n){1,15}?(?:.*?(?:End|Total|Sub-?total).*?)?', "Emissions Table"),
            
            # Scope emissions with flexible table row formats
            (r'(?:^|\n|\||\s{2,})(?:[^\n|]*?)?Scope\s*1(?:[^\n|]*?)(?:\||$|\s{2,})(?:[^\n|]*?)(\d[\d,.]*)(?:\s*(?:tCO2-?e|tonnes?\s*CO2-?e|kt|Mt|million\s*tonnes?))', "Scope 1"),
            (r'(?:^|\n|\||\s{2,})(?:[^\n|]*?)?Scope\s*2(?:[^\n|]*?)(?:\||$|\s{2,})(?:[^\n|]*?)(\d[\d,.]*)(?:\s*(?:tCO2-?e|tonnes?\s*CO2-?e|kt|Mt|million\s*tonnes?))', "Scope 2"),
            (r'(?:^|\n|\||\s{2,})(?:[^\n|]*?)?Scope\s*3(?:[^\n|]*?)(?:\||$|\s{2,})(?:[^\n|]*?)(\d[\d,.]*)(?:\s*(?:tCO2-?e|tonnes?\s*CO2-?e|kt|Mt|million\s*tonnes?))', "Scope 3"),
            
            # Carbon inventory with flexible structure
            (r'(?:Carbon|GHG|Emissions)\s*(?:inventory|profile|footprint)[\s\S]{0,200}?(?:(?:Scope|Total|Direct|Indirect)[\s\S]{0,100}?(?:\d[\d,.]*\s*(?:tCO2-?e|tonnes?\s*CO2-?e|kt|Mt|million\s*tonnes?))){1,5}', "Carbon Inventory"),
            
            # Total emissions with various formats
            (r'(?:Total|Group|Overall|Gross)\s*(?:GHG|Carbon|Scope)?\s*emissions[^\n]*?(\d[\d,.]*)(?:\s*(?:tCO2-?e|tonnes?\s*CO2-?e|kt|Mt|million\s*tonnes?))', "Total Emissions"),
            
            # Energy consumption with flexible units
            (r'(?:Energy|Electricity)\s*consumption[^\n]*?(\d[\d,.]*)(?:\s*(?:kWh|MWh|GWh|PJ|TJ))', "Energy Consumption"),
            
            # Renewable energy metrics
            (r'(?:Renewable|Clean)\s*energy\s*(?:target|percentage|share)[^\n]*?(\d+(?:\.\d+)?)%', "Renewable Energy"),
            (r'RE100.*?target.*?(?:source|achieve)\s*100%.*?renewable\s*sources.*?by\s*(\d{4})', "RE100 Target"),
            
            # Net zero and reduction targets
            (r'(?:Net[- ]Zero|Carbon[- ]Neutral)\s*(?:target|commitment|goal)[^\n]*?(?:by\s*)?(\d{4})', "Net Zero Target"),
            (r'(?:emissions?|carbon)\s*reduction\s*target[^\n]*?(\d+(?:\.\d+)?)%[^\n]*?(?:by\s*)?(\d{4})', "Reduction Target"),
            
            # Environmental finance
            (r'(?:Environmental|Sustainable|Green)\s*finance[^\n]*?\$?\s*(\d+(?:\.\d+)?)\s*(?:billion|million|B|M)', "Environmental Finance"),
            
            # Base year and current progress
            (r'(?:base|reference)\s*year[^\n]*?(\d{4})', "Base Year"),
            (r'(?:achieved|reduced|decreased)[^\n]*?(\d+(?:\.\d+)?)%[^\n]*?(?:reduction|decrease)', "Current Progress")
        ]
        
        # Find all matches for each pattern
        for pattern, section_name in metric_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            for match in matches:
                context = match.group(0).strip()
                # Only include contexts with numeric values
                if re.search(r'\d', context):
                    # Clean up the context
                    context = re.sub(r'\s+', ' ', context)
                    # Add surrounding lines for better context
                    lines = text.split('\n')
                    for i, line in enumerate(lines):
                        if context in line:
                            start = max(0, i-4)  # 4 lines before
                            end = min(len(lines), i+5)  # 4 lines after
                            # Include table headers if present
                            header_pattern = r'(?:Table|TABLE).*?(?:GHG|Greenhouse Gas|Carbon|Emissions)'
                            for j in range(max(0, i-10), i):
                                if re.search(header_pattern, lines[j]):
                                    start = j
                                    break
                            context = '\n'.join(lines[start:end])
                            break
                    contexts.append(f"\n=== {section_name} ===\n{context}")
        
        # If we found any contexts, combine them
        if contexts:
            return "\n".join(contexts)
        return text

    def extract(self, text: str, url: str = None, **kwargs) -> str:
        """Extract ESG metrics using both regex patterns and vector store."""
        try:
            # Clean and prepare the text
            cleaned_text = self._clean_text(text)
            
            # Extract relevant metric contexts using regex
            regex_contexts = self._extract_metric_contexts(cleaned_text)
            
            # Get relevant chunks from vector store
            vector_contexts = self._get_vector_store_contexts(cleaned_text)
            
            # Combine contexts
            all_contexts = self._combine_contexts(regex_contexts, vector_contexts)
            
            # Extract company name from URL if possible
            company_name = self._extract_company_name(url) if url else None
            
            # Create prompt with combined context
            analysis_prompt = f"{self.prompt}\n\n"
            if company_name:
                analysis_prompt += f"Company: {company_name}\n\n"
            analysis_prompt += f"Document text:\n{all_contexts}"
            
            # Call Groq API with enhanced context
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": "You are an expert at analyzing sustainability and ESG reports. Extract numeric metrics precisely and return only valid JSON."},
                    {"role": "user", "content": analysis_prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": 1000
            }
            
            print("Making API call to Groq...")
            print("\nExtracted contexts:")
            print(all_contexts[:1000] + "..." if len(all_contexts) > 1000 else all_contexts)
            
            llm_response = self._call_groq_api(payload, headers)
            response_text = llm_response["choices"][0]["message"]["content"]
            print("Received response from Groq")
            
            # Debug: Print the raw response
            print("\nRaw LLM response:")
            print(response_text[:500])  # Print first 500 chars
            
            # Extract JSON from the response
            json_match = re.search(r'({[\s\S]*})', response_text)
            if not json_match:
                print("No JSON found in response")
                return [self._get_empty_metrics_json(company_name)]
                
            json_str = json_match.group(1)
            
            # Validate and clean the JSON
            try:
                metrics = json.loads(json_str)
                
                # Ensure company name is set
                if company_name and not metrics.get('company_name'):
                    metrics['company_name'] = company_name
                
                # Validate numeric fields
                numeric_fields = [
                    'scope1_emissions', 'scope2_emissions', 'scope3_emissions',
                    'renewable_energy_percentage', 'renewable_energy_target',
                    'emission_reduction_target', 'current_reduction_percentage'
                ]
                
                for field in numeric_fields:
                    if field in metrics:
                        try:
                            val = metrics[field]
                            if val is not None:
                                metrics[field] = float(val)
                        except (ValueError, TypeError):
                            print(f"Invalid numeric value for {field}: {metrics[field]}")
                            metrics[field] = None
                
                # Validate year fields
                year_fields = ['year', 'target_year', 'net_zero_commitment_year']
                for field in year_fields:
                    if field in metrics:
                        val = metrics[field]
                        if val is not None:
                            # Extract year if it's a string like "by 2030" or "2030"
                            year_match = re.search(r'20\d{2}', str(val))
                            if year_match:
                                metrics[field] = year_match.group(0)
                            else:
                                metrics[field] = None
                
                print("\nExtracted metrics:")
                print(json.dumps(metrics, indent=2))
                
                return [json.dumps(metrics)]
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {str(e)}")
                print("Invalid JSON string:", json_str[:200])
                return [self._get_empty_metrics_json(company_name)]
            
        except Exception as e:
            print(f"Error extracting metrics: {str(e)}")
            return [self._get_empty_metrics_json(company_name)]

    def _clean_text(self, text: str) -> str:
        """Clean and prepare text for analysis"""
        if not text:
            return ""
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Fix common PDF formatting issues
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)  # Fix split numbers
        text = re.sub(r'(\d),\s+(\d)', r'\1,\2', text)  # Fix split thousands
        text = re.sub(r'(\d+)\s*[.,]\s*(\d+)', r'\1.\2', text)  # Standardize decimals
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII chars
        
        # Fix unit formatting
        text = re.sub(r'Mt\s*CO2[- ]?e', 'MtCO2-e', text, flags=re.IGNORECASE)
        text = re.sub(r'kt\s*CO2[- ]?e', 'ktCO2-e', text, flags=re.IGNORECASE)
        text = re.sub(r't\s*CO2[- ]?e', 'tCO2-e', text, flags=re.IGNORECASE)
        
        # Fix percentage formatting
        text = re.sub(r'(\d+)\s*%', r'\1%', text)
        text = re.sub(r'(\d+)\s*percent', r'\1%', text, flags=re.IGNORECASE)
        
        # Fix monetary values
        text = re.sub(r'\$\s*(\d+)', r'$\1', text)
        text = re.sub(r'(\d+)\s*billion', r'\1B', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*million', r'\1M', text, flags=re.IGNORECASE)
        
        # Standardize year formats
        text = re.sub(r'FY\s*(\d{2})', r'20\1', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d{2})(\d{2})/(\d{2})', r'\1\2', text)  # Convert 2023/24 to 2023
        
        # Remove footnote markers
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\(\d+\)', '', text)
        
        # Truncate to fit context window (leaving room for prompt)
        max_length = 6000  # Reduced to ensure we stay within context window
        if len(text) > max_length:
            # Try to break at a sentence
            truncated = text[:max_length]
            last_period = truncated.rfind('.')
            if last_period > max_length * 0.8:  # Only use if we don't lose too much text
                truncated = truncated[:last_period + 1]
            return truncated
        return text

    def _extract_company_name(self, url: str) -> Optional[str]:
        """Extract company name from URL"""
        if not url:
            return None
            
        # Try to extract from common patterns
        patterns = [
            r'www\.([^/]+)\.com',
            r'documents/([^/]+)/',
            r'([^/]+)/sustainability'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url.lower())
            if match:
                name = match.group(1)
                # Clean up the name
                name = name.replace('-', ' ').title()
                return name
        return None
        
    def _get_empty_metrics_json(self, company_name: str = None) -> str:
        """Return an empty metrics JSON structure"""
        return json.dumps({
            'company_name': company_name,
            'year': None,
            'scope1_emissions': None,
            'scope2_emissions': None,
            'scope3_emissions': None,
            'emissions_unit': None,
            'emissions_base_year': None,
            'renewable_energy_percentage': None,
            'renewable_energy_target': None,
            'target_year': None,
            'emission_reduction_target': None,
            'emission_reduction_base_year': None,
            'current_reduction_percentage': None,
            'net_zero_commitment_year': None,
            'carbon_neutral_certified': None,
            'internal_carbon_price': None,
            'sustainable_finance_target': None,
            'climate_related_investment': None
        })

    def _get_vector_store_contexts(self, text: str) -> str:
        """Get relevant contexts from vector store."""
        contexts = []
        
        # Search for emissions data
        emissions_results = self.vector_store.search_emissions_data(k=3)
        if emissions_results:
            contexts.append("\n=== Emissions Data from Vector Store ===\n")
            contexts.extend(doc.page_content for doc in emissions_results)
        
        # Search for targets and commitments
        target_results = self.vector_store.search_targets_and_commitments(k=2)
        if target_results:
            contexts.append("\n=== Targets and Commitments from Vector Store ===\n")
            contexts.extend(doc.page_content for doc in target_results)
        
        # Search for financial metrics
        financial_results = self.vector_store.search_financial_metrics(k=2)
        if financial_results:
            contexts.append("\n=== Financial Metrics from Vector Store ===\n")
            contexts.extend(doc.page_content for doc in financial_results)
        
        return "\n".join(contexts)
        
    def _combine_contexts(self, regex_contexts: str, vector_contexts: str) -> str:
        """Combine and deduplicate contexts from different sources."""
        # Split contexts into sections
        regex_sections = regex_contexts.split("\n===")
        vector_sections = vector_contexts.split("\n===")
        
        # Combine unique sections
        unique_sections = []
        seen_content = set()
        
        for section in regex_sections + vector_sections:
            # Clean and normalize section content
            cleaned_section = re.sub(r'\s+', ' ', section).strip()
            if cleaned_section and cleaned_section not in seen_content:
                seen_content.add(cleaned_section)
                unique_sections.append(section)
        
        # Reassemble sections
        combined_text = "\n===".join(unique_sections)
        return combined_text

def get_browser_config() -> BrowserConfig:
    return BrowserConfig(
        headless=True,
        ignore_https_errors=True,
        accept_downloads=True
    )

@dataclass
class ESGMetrics:
    company_name: str
    year: int
    scope1_emissions: Optional[float] = None
    scope2_emissions: Optional[float] = None
    scope3_emissions: Optional[float] = None
    renewable_energy_percentage: Optional[float] = None
    renewable_energy_target: Optional[float] = None
    target_year: Optional[int] = None
    emission_reduction_target: Optional[float] = None
    current_reduction_percentage: Optional[float] = None
    net_zero_commitment_year: Optional[int] = None
    carbon_price_used: Optional[float] = None
    energy_efficiency_initiatives: Optional[List[str]] = None
    renewable_projects: Optional[List[str]] = None
    
def get_llm_strategy() -> ESGExtractionStrategy:
    """Get the LLM strategy for extracting ESG metrics"""
    prompt = """
    You are an expert at analyzing climate and sustainability reports. Your task is to extract specific ESG metrics from the provided text.
    Focus on finding these key metrics:
    1. Emissions data (Scope 1, 2, and 3) in tCO2-e or similar units
    2. Renewable energy targets and current percentage
    3. Net zero commitment year and interim targets
    4. Emission reduction targets (percentage and target year)
    5. Sustainable finance commitments (dollar amounts)
    6. Climate-related investments
    7. Carbon neutral certification status

    Return the metrics in this JSON format:
    {
        "company_name": "string",
        "year": "YYYY",
        "scope1_emissions": number,
        "scope2_emissions": number,
        "scope3_emissions": number,
        "emissions_unit": "string",
        "emissions_base_year": "YYYY",
        "renewable_energy_percentage": number,
        "renewable_energy_target": number,
        "target_year": "YYYY",
        "emission_reduction_target": number,
        "emission_reduction_base_year": "YYYY",
        "current_reduction_percentage": number,
        "net_zero_commitment_year": "YYYY",
        "carbon_neutral_certified": boolean,
        "internal_carbon_price": number,
        "sustainable_finance_target": number,
        "climate_related_investment": number
    }

    Only include metrics that you find with high confidence. Use null for values you cannot find or are uncertain about.
    Look for both numerical values and contextual statements that confirm these metrics.
    """
    
    return ESGExtractionStrategy(
        model="llama-3.3-70b-versatile",
        temperature=0.1,  # Lower temperature for more focused extraction
        prompt=prompt
    ) 