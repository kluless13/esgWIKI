from crawl4ai import BrowserConfig, ExtractionStrategy
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import os
import requests
import re
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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
        self.min_request_interval = 2.0  # Increased to 2 seconds between requests
        
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
        
        # Key phrases to look for and their context windows
        metric_patterns = [
            (r'scope 1[\s\S]{0,300}', "Scope 1 Emissions"),
            (r'scope 2[\s\S]{0,300}', "Scope 2 Emissions"),
            (r'scope 3[\s\S]{0,300}', "Scope 3 Emissions"),
            (r'(?:total|operational)?\s*emissions[\s\S]{0,300}', "Total Emissions"),
            (r'renewable\s*energy[\s\S]{0,300}', "Renewable Energy"),
            (r'net\s*zero[\s\S]{0,300}', "Net Zero"),
            (r'carbon\s*neutral[\s\S]{0,300}', "Carbon Neutral"),
            (r'emission.*reduction.*target[\s\S]{0,300}', "Emission Reduction"),
            (r'climate.*investment[\s\S]{0,300}', "Climate Investment"),
            (r'sustainable\s*finance[\s\S]{0,300}', "Sustainable Finance")
        ]
        
        # Find all matches for each pattern
        for pattern, section_name in metric_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Get the matched text and some context after it
                context = match.group(0)
                
                # Look for numeric values in the context
                if re.search(r'\d', context):
                    contexts.append(f"\n=== {section_name} ===\n{context.strip()}")
        
        # If we found any contexts, combine them
        if contexts:
            return "\n".join(contexts)
        return text

    def extract(self, text: str, url: str = None, **kwargs) -> str:
        """
        Extract ESG metrics from the text using Groq's LLM.
        """
        try:
            # Clean and prepare the text
            cleaned_text = self._clean_text(text)
            
            # Extract relevant metric contexts
            metric_contexts = self._extract_metric_contexts(cleaned_text)
            
            # Extract company name from URL if possible
            company_name = self._extract_company_name(url) if url else None
            
            # Create prompt with context
            analysis_prompt = f"{self.prompt}\n\n"
            if company_name:
                analysis_prompt += f"Company: {company_name}\n\n"
            analysis_prompt += f"Document text:\n{metric_contexts}"
            
            # Call Groq API
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
            print(metric_contexts[:1000] + "..." if len(metric_contexts) > 1000 else metric_contexts)
            
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
    """Enhanced LLM strategy for extracting ESG metrics"""
    prompt = """You are an expert at analyzing sustainability, climate, and ESG reports. Your task is to thoroughly read through the document and extract key ESG metrics.

Key Instructions:
1. Read the entire document carefully
2. Look for both explicit metrics and calculated values
3. Pay attention to different measurement units (e.g., tCO2-e, MtCO2-e, ktCO2-e)
4. Consider both absolute values and intensity metrics
5. Look for both current performance and future targets
6. Extract the most recent year's data unless specified otherwise

Focus Areas to Extract:

1. Emissions Data:
   - Scope 1 emissions (direct)
   - Scope 2 emissions (indirect from energy)
   - Scope 3 emissions (value chain)
   - Look for both location-based and market-based Scope 2
   - Note any emissions intensity metrics
   - Check for base year emissions

2. Energy & Renewables:
   - Current renewable energy usage (%)
   - Renewable energy targets
   - Energy consumption data
   - Energy efficiency metrics
   - Power purchase agreements (PPAs)

3. Climate Targets:
   - Emission reduction targets (short, medium, long-term)
   - Current progress against targets
   - Net zero commitment year
   - Science-based targets
   - Carbon neutral certifications

4. Additional Metrics:
   - Carbon offsets purchased/retired
   - Internal carbon price
   - Green/sustainable financing
   - Climate-related investments
   - Physical risk metrics

Return the data in this exact JSON format:
{
    "company_name": string,  // Company name from document
    "year": string,         // Most recent reporting year
    "scope1_emissions": number or null,  // In tCO2-e
    "scope2_emissions": number or null,  // In tCO2-e (market-based if available)
    "scope3_emissions": number or null,  // In tCO2-e
    "emissions_unit": string or null,    // e.g., "tCO2-e", "ktCO2-e", "MtCO2-e"
    "emissions_base_year": string or null,  // Base year for targets
    "renewable_energy_percentage": number or null,  // Current renewable %
    "renewable_energy_target": number or null,      // Target renewable %
    "target_year": string or null,                 // Year for renewable target
    "emission_reduction_target": number or null,    // Reduction target %
    "emission_reduction_base_year": string or null, // Base year for reduction
    "current_reduction_percentage": number or null, // Current reduction achieved %
    "net_zero_commitment_year": string or null,    // Net zero target year
    "carbon_neutral_certified": boolean or null,   // Current carbon neutral status
    "internal_carbon_price": number or null,      // Internal carbon price used
    "sustainable_finance_target": number or null,  // In billions
    "climate_related_investment": number or null   // In millions
}

Important Guidelines:
1. Extract NUMERIC values whenever possible
2. Convert all percentages to numbers (e.g., "40%" → 40)
3. Use null for metrics not found
4. Standardize units (convert kt to t if needed)
5. Include the unit of measurement in emissions_unit
6. Look for both absolute values and percentage changes
7. Pay attention to footnotes and technical appendices
8. Consider both operational and financed emissions for financial institutions
9. Only include metrics you are confident about
10. Return only the JSON object, no other text"""
    
    return ESGExtractionStrategy(
        model="llama-3.3-70b-versatile",
        temperature=0.1,  # Low temperature for consistent extraction
        prompt=prompt
    ) 