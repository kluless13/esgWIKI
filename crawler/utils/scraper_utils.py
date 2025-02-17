from crawl4ai import BrowserConfig, ExtractionStrategy
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import os
import requests
import re

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
        
    def extract(self, text: str, url: str = None, **kwargs) -> str:
        """
        Extract ESG metrics from the text using Groq's LLM.
        """
        try:
            # Clean and prepare the text
            cleaned_text = self._clean_text(text)
            
            # Extract company name from URL if possible
            company_name = self._extract_company_name(url) if url else None
            
            # Create prompt with context
            analysis_prompt = f"{self.prompt}\n\n"
            if company_name:
                analysis_prompt += f"Company: {company_name}\n\n"
            analysis_prompt += f"Document text:\n{cleaned_text}"
            
            # Call Groq API
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "llama-3.3-70b-versatile",  # Using Groq's LLaMA model
                "messages": [
                    {"role": "system", "content": "You are an expert at analyzing sustainability and ESG reports."},
                    {"role": "user", "content": analysis_prompt}
                ],
                "temperature": self.temperature
            }
            
            response = requests.post(GROQ_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            
            llm_response = response.json()["choices"][0]["message"]["content"]
            
            # Validate the response is proper JSON
            try:
                parsed_json = json.loads(llm_response)
                if company_name and not parsed_json.get('company_name'):
                    parsed_json['company_name'] = company_name
                return [json.dumps(parsed_json)]
            except json.JSONDecodeError:
                print(f"Error: LLM response was not valid JSON")
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
        
        # Remove common PDF artifacts
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)  # Fix split numbers
        text = re.sub(r'[^\x00-\x7F]+', '', text)     # Remove non-ASCII chars
        
        # Truncate to fit context window (leaving room for prompt)
        return text[:8000]

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
                name = name.replace('-', ' ').title()
                return name
        
        return None

    def _get_empty_metrics_json(self, company_name: str = None) -> str:
        """Return empty metrics JSON structure"""
        return json.dumps({
            "company_name": company_name,
            "year": None,
            "scope1_emissions": None,
            "scope2_emissions": None,
            "scope3_emissions": None,
            "renewable_energy_percentage": None,
            "renewable_energy_target": None,
            "target_year": None,
            "emission_reduction_target": None,
            "current_reduction_percentage": None,
            "net_zero_commitment_year": None,
            "carbon_price_used": None,
            "energy_efficiency_initiatives": None,
            "renewable_projects": None
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
    prompt = """
    You are an expert at analyzing sustainability and ESG reports. Extract the following metrics from the report:

    1. Emissions:
       - Scope 1 emissions (direct GHG emissions) in tCO2e
       - Scope 2 emissions (indirect GHG from energy) in tCO2e
       - Scope 3 emissions (other indirect GHG) in tCO2e

    2. Renewable Energy:
       - Current renewable energy percentage
       - Renewable energy target percentage
       - Target year for renewable energy goal

    3. Emission Reduction:
       - Overall emission reduction target (percentage)
       - Current progress towards reduction (percentage)
       - Net zero commitment year

    4. Other:
       - Carbon price used in planning ($/tCO2e)
       - Key energy efficiency initiatives (list)
       - Major renewable energy projects (list)

    Format the response as JSON with the following structure:
    {
        "company_name": string,
        "year": int,
        "scope1_emissions": float or null,
        "scope2_emissions": float or null,
        "scope3_emissions": float or null,
        "renewable_energy_percentage": float or null,
        "renewable_energy_target": float or null,
        "target_year": int or null,
        "emission_reduction_target": float or null,
        "current_reduction_percentage": float or null,
        "net_zero_commitment_year": int or null,
        "carbon_price_used": float or null,
        "energy_efficiency_initiatives": [string] or null,
        "renewable_projects": [string] or null
    }

    Important:
    - Extract the most recent year's data if multiple years are present
    - Use null for any metrics not found in the report
    - Convert all percentages to decimal (e.g., 75% -> 0.75)
    - Include units in the initiatives/projects descriptions
    - Be precise with numbers, don't round unless specified in the report
    """
    
    return ESGExtractionStrategy(
        model="llama-3.3-70b-versatile",  # Using Groq's LLaMA model
        temperature=0.1,  # Low temperature for more consistent extraction
        prompt=prompt
    ) 