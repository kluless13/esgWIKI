import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
from utils.db_manager import ESGDatabaseManager
import logging
import re
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import json
import requests
from openai import OpenAI
import time
from random import uniform

# Configure logging first for immediate feedback
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("Starting ESG metrics extraction...")

# Load environment variables
logger.info("Loading environment variables...")
load_dotenv()

# Configure API keys and models
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'o3-mini-2025-1-31')
GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2')

# Common patterns for emissions data
scope1_patterns = [
    'scope 1',
    'scope one',
    'scope1',
    'direct emissions',
    'direct ghg emissions',
    'direct greenhouse gas',
    'direct co2'
]

scope2_patterns = [
    'scope 2',
    'scope two',
    'scope2',
    'indirect emissions',
    'electricity emissions',
    'indirect ghg emissions',
    'energy indirect',
    'purchased electricity'
]

scope3_patterns = [
    'total scope 3',
    'scope 3 total',
    'scope three total',
    'total scope three',
    'scope 3 emissions total',
    'total scope 3 emissions',
    'value chain emissions',
    'other indirect emissions'
]

# Additional patterns for metrics
base_year_patterns = [
    'base year',
    'baseline year',
    'reference year',
    'starting year',
    'base period',
    'baseline period'
]

target_patterns = [
    'emission reduction target',
    'ghg reduction target',
    'emissions target',
    'reduction target',
    'emissions reduction goal',
    'climate target',
    'carbon reduction goal'
]

net_zero_patterns = [
    'net zero',
    'net-zero',
    'carbon neutral',
    'climate neutral',
    'zero emissions',
    'carbon neutrality',
    'climate neutrality'
]

carbon_price_patterns = [
    'carbon price',
    'price on carbon',
    'internal carbon',
    'carbon pricing',
    'shadow carbon price',
    'co2 price'
]

sustainable_finance_patterns = [
    'sustainable finance',
    'green finance',
    'sustainable investment',
    'climate finance',
    'green investment',
    'esg investment',
    'sustainability-linked'
]

class ESGMetrics(BaseModel):
    scope1_emissions: Optional[float] = Field(None, description="Scope 1 direct GHG emissions in mtCO2e")
    scope2_emissions: Optional[float] = Field(None, description="Scope 2 indirect GHG emissions in mtCO2e")
    scope3_emissions: Optional[float] = Field(None, description="Scope 3 value chain GHG emissions in mtCO2e")
    renewable_energy_percentage: Optional[float] = Field(None, description="Percentage of renewable energy usage (0-100)")
    energy_consumption: Optional[float] = Field(None, description="Total energy consumption in MWh")
    renewable_energy_target: Optional[float] = Field(None, description="Renewable energy target percentage (0-100)")
    target_year: Optional[int] = Field(None, description="Target year for renewable energy goal")
    emissions_unit: Optional[str] = Field(None, description="Unit for emissions data (e.g., mtCO2e)")
    energy_consumption_unit: Optional[str] = Field(None, description="Unit for energy consumption")

class ExcelDataProcessor:
    def __init__(self, db_manager: ESGDatabaseManager = None):
        self.db_manager = db_manager or ESGDatabaseManager()
        
        # Initialize embeddings only when needed
        self._embeddings = None
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # System prompt for metric extraction
        self.system_prompt = """You are an expert in analyzing ESG (Environmental, Social, and Governance) data.
        Given text from an Excel sheet, extract key ESG metrics focusing on emissions data, energy consumption,
        and renewable energy information. Be precise and return only high-confidence metrics.
        
        Guidelines:
        1. Convert all emissions to mtCO2e (million tonnes CO2 equivalent)
        2. Convert all energy values to MWh
        3. Express renewable percentages as numbers between 0 and 100
        4. Distinguish between current values and targets
        5. Look for contextual clues about units and time periods
        6. If a value seems unreasonable, note it but include it
        7. For conflicting values, prefer the most recent or most detailed one"""

    @property
    def embeddings(self):
        """Lazy loading of embeddings model"""
        if self._embeddings is None:
            logger.info("Initializing embeddings model...")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )
        return self._embeddings

    def extract_metrics_from_sheet(self, df: pd.DataFrame, sheet_name: str) -> Dict:
        """
        Extract ESG metrics from a DataFrame using OpenAI's API.
        """
        # Convert DataFrame to text for AI analysis
        text_data = self._df_to_text(df)
        
        try:
            # Call OpenAI API for metric extraction
            completion = self.client.beta.chat.completions.parse(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Sheet Name: {sheet_name}\n\nData:\n{text_data}"}
                ],
                response_format=ESGMetrics,
            )
            
            # Extract metrics from the response
            metrics = completion.choices[0].message.parsed
            logger.info(f"Successfully extracted metrics using OpenAI: {metrics}")
            return metrics.model_dump(exclude_none=True)
            
        except Exception as e:
            logger.error(f"Error in OpenAI metric extraction: {e}")
            return {}

    def clean_numeric_value(self, value: str) -> Optional[float]:
        """
        Enhanced method to clean and convert string to numeric value with better handling of edge cases
        """
        if value is None:
            return None
            
        if isinstance(value, (int, float)):
            return float(value)
            
        if not isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                return None
                
        # Remove common non-numeric characters
        clean_value = value.strip()
        clean_value = re.sub(r'[,$€£¥]', '', clean_value)
        
        # Handle parentheses (negative numbers)
        if clean_value.startswith('(') and clean_value.endswith(')'):
            clean_value = '-' + clean_value[1:-1]
            
        # Handle percentage signs
        has_percent = '%' in clean_value
        clean_value = clean_value.replace('%', '')
        
        # Handle thousand separators and decimal points
        # First, standardize decimal points if comma is used (e.g. European format)
        if ',' in clean_value and '.' not in clean_value:
            clean_value = clean_value.replace(',', '.')
        else:
            clean_value = clean_value.replace(',', '')
            
        # Handle special cases like 'N/A', '-', etc.
        if clean_value.lower() in ['n/a', 'na', '-', '', 'nil', 'null']:
            return None
            
        try:
            value = float(clean_value)
            # Convert percentage to decimal if needed
            if has_percent:
                value = value / 100
            return value
        except (ValueError, TypeError):
            return None

    def standardize_unit(self, value: float, from_unit: str, to_unit: str) -> Optional[float]:
        """
        Standardize units for energy and emissions metrics
        """
        if value is None:
            return None
            
        # Define conversion factors
        energy_conversions = {
            'kwh': {
                'mwh': 0.001,
                'gwh': 0.000001,
                'twh': 0.000000001,
                'gj': 0.0036,
                'tj': 0.0000036
            },
            'mwh': {
                'kwh': 1000,
                'gwh': 0.001,
                'twh': 0.000001,
                'gj': 3.6,
                'tj': 0.0036
            },
            'gwh': {
                'kwh': 1000000,
                'mwh': 1000,
                'twh': 0.001,
                'gj': 3600,
                'tj': 3.6
            },
            'gj': {
                'kwh': 277.778,
                'mwh': 0.277778,
                'gwh': 0.000277778,
                'tj': 0.001
            },
            'tj': {
                'gj': 1000,
                'kwh': 277778,
                'mwh': 277.778,
                'gwh': 0.277778
            }
        }
        
        emissions_conversions = {
            'tco2e': {
                'ktco2e': 0.001,
                'mtco2e': 0.000001,
                'gtco2e': 0.000000001
            },
            'ktco2e': {
                'tco2e': 1000,
                'mtco2e': 0.001,
                'gtco2e': 0.000001
            },
            'mtco2e': {
                'tco2e': 1000000,
                'ktco2e': 1000,
                'gtco2e': 0.001
            },
            'gtco2e': {
                'tco2e': 1000000000,
                'ktco2e': 1000000,
                'mtco2e': 1000
            }
        }
        
        try:
            # Normalize units to lowercase
            from_unit = from_unit.lower().replace('-', '').replace(' ', '')
            to_unit = to_unit.lower().replace('-', '').replace(' ', '')
            
            # If units are the same, return original value
            if from_unit == to_unit:
                return value
                
            # Handle energy conversions
            if from_unit in energy_conversions and to_unit in energy_conversions[from_unit]:
                return value * energy_conversions[from_unit][to_unit]
                
            # Handle emissions conversions
            if from_unit in emissions_conversions and to_unit in emissions_conversions[from_unit]:
                return value * emissions_conversions[from_unit][to_unit]
                
            logger.warning(f"Unsupported unit conversion: {from_unit} to {to_unit}")
            return None
            
        except Exception as e:
            logger.error(f"Error in unit conversion: {str(e)}")
            return None

    def extract_unit_from_text(self, text: str) -> Optional[str]:
        """
        Extract unit from text with common variations
        """
        # Common unit patterns
        unit_patterns = {
            r'\b(?:k?wh|kilowatt\s*hours?)\b': 'kwh',
            r'\b(?:mwh|megawatt\s*hours?)\b': 'mwh',
            r'\b(?:gwh|gigawatt\s*hours?)\b': 'gwh',
            r'\b(?:twh|terawatt\s*hours?)\b': 'twh',
            r'\b(?:gj|gigajoules?)\b': 'gj',
            r'\b(?:tj|terajoules?)\b': 'tj',
            r'\b(?:t\s*co2(?:e|-eq)?|tonnes?\s*(?:of\s*)?co2(?:e|-eq)?)\b': 'tco2e',
            r'\b(?:kt\s*co2(?:e|-eq)?|kilotonnes?\s*(?:of\s*)?co2(?:e|-eq)?)\b': 'ktco2e',
            r'\b(?:mt\s*co2(?:e|-eq)?|megatonnes?\s*(?:of\s*)?co2(?:e|-eq)?)\b': 'mtco2e',
            r'\b(?:gt\s*co2(?:e|-eq)?|gigatonnes?\s*(?:of\s*)?co2(?:e|-eq)?)\b': 'gtco2e'
        }
        
        text = text.lower()
        for pattern, unit in unit_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return unit
        return None

    def _analyze_with_groq(self, text: str, sheet_name: str) -> Dict:
        """
        Analyze text data using Groq's LLM API for ESG metric extraction.
        Handles large payloads by chunking the data and implements rate limiting based on Groq's limits:
        - 30 RPM (Requests Per Minute)
        - 1,000 RPD (Requests Per Day)
        - 6,000 TPM (Tokens Per Minute)
        - 100,000 TPD (Tokens Per Day)
        """
        GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
        MAX_CHUNK_SIZE = 4000  # Reduced chunk size to stay within token limits
        MAX_RETRIES = 3
        MIN_DELAY = 2  # Minimum delay between requests (to stay under 30 RPM)
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        system_prompt = """You are an expert in analyzing ESG (Environmental, Social, and Governance) data.
        Your task is to extract key ESG metrics from Excel data, focusing on emissions, energy consumption,
        and renewable energy information. Be precise and return only high-confidence metrics."""
        
        # Split text into smaller chunks to stay within token limits
        chunks = []
        current_chunk = []
        current_length = 0
        
        for line in text.split('\n'):
            line_length = len(line)
            if current_length + line_length > MAX_CHUNK_SIZE:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_length = line_length
            else:
                current_chunk.append(line)
                current_length += line_length
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        if not chunks:
            chunks = [text]
        
        # Process each chunk and merge results
        all_metrics = {}
        last_request_time = 0
        
        for i, chunk in enumerate(chunks):
            user_prompt = f"""Analyze the following portion ({i+1}/{len(chunks)}) of Excel data and extract ESG metrics.
            Sheet Name: {sheet_name}
            
            Data:
            {chunk}
            
            Extract and return a JSON object with the following metrics (only if found with high confidence):
            - scope1_emissions: Scope 1 direct GHG emissions in mtCO2e
            - scope2_emissions: Scope 2 indirect GHG emissions in mtCO2e
            - scope3_emissions: Scope 3 value chain GHG emissions in mtCO2e
            - renewable_energy_percentage: Current renewable energy usage (0-100)
            - energy_consumption: Total energy consumption in MWh
            - renewable_energy_target: Future renewable energy target percentage (0-100)
            - target_year: Target year for renewable energy goal
            - emissions_unit: Unit for emissions data
            - energy_consumption_unit: Unit for energy consumption
            
            Return ONLY the JSON object, no additional text."""
            
            payload = {
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            # Implement rate limiting and retries
            for retry in range(MAX_RETRIES):
                try:
                    # Ensure minimum delay between requests (rate limiting)
                    current_time = time.time()
                    time_since_last_request = current_time - last_request_time
                    if time_since_last_request < MIN_DELAY:
                        sleep_time = MIN_DELAY - time_since_last_request + uniform(0.1, 0.5)  # Add jitter
                        logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                        time.sleep(sleep_time)
                    
                    if retry > 0:
                        # Exponential backoff for retries
                        delay = MIN_DELAY * (2 ** retry) + uniform(0.1, 0.5)
                        logger.info(f"Retrying chunk {i+1} after {delay:.2f} seconds (attempt {retry + 1}/{MAX_RETRIES})")
                        time.sleep(delay)
                    
                    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
                    last_request_time = time.time()
                    
                    # Handle rate limit response headers
                    remaining_requests = int(response.headers.get('x-ratelimit-remaining-requests', 0))
                    remaining_tokens = int(response.headers.get('x-ratelimit-remaining-tokens', 0))
                    
                    if remaining_requests < 5 or remaining_tokens < 1000:
                        logger.warning(f"Rate limit approaching: {remaining_requests} requests, {remaining_tokens} tokens remaining")
                        time.sleep(MIN_DELAY * 2)  # Add extra delay when approaching limits
                    
                    if response.status_code == 429:  # Rate limit exceeded
                        retry_after = int(response.headers.get('retry-after', MIN_DELAY))
                        if retry < MAX_RETRIES - 1:
                            logger.warning(f"Rate limit exceeded, waiting {retry_after} seconds")
                            time.sleep(retry_after)
                            continue
                        else:
                            logger.error(f"Rate limit exceeded for chunk {i+1} after all retries")
                            break
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    if "choices" in result and len(result["choices"]) > 0:
                        try:
                            chunk_metrics = json.loads(result["choices"][0]["message"]["content"])
                            logger.info(f"Successfully extracted metrics from chunk {i+1}: {chunk_metrics}")
                            
                            # Merge metrics, keeping the highest confidence values
                            for key, value in chunk_metrics.items():
                                if value is not None:
                                    if key not in all_metrics or (
                                        isinstance(value, (int, float)) and 
                                        value > all_metrics.get(key, 0)
                                    ):
                                        all_metrics[key] = value
                            
                            break  # Success, break retry loop
                            
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse Groq response as JSON for chunk {i+1}: {e}")
                            if retry == MAX_RETRIES - 1:
                                continue  # Skip to next chunk if all retries failed
                    else:
                        logger.error(f"No valid response from Groq API for chunk {i+1}")
                        if retry == MAX_RETRIES - 1:
                            continue  # Skip to next chunk if all retries failed
                    
                except Exception as e:
                    logger.error(f"Error calling Groq API for chunk {i+1}: {e}")
                    if retry == MAX_RETRIES - 1:
                        continue  # Skip to next chunk if all retries failed
        
        return all_metrics

    def _df_to_text(self, df: pd.DataFrame) -> str:
        """
        Convert DataFrame to a text format suitable for AI analysis.
        """
        text_parts = []
        
        # Add column information
        text_parts.append("Column Headers:")
        for col in df.columns:
            text_parts.append(f"- {col}")
        
        # Add row data with row numbers
        text_parts.append("\nData Rows:")
        for idx, row in df.iterrows():
            row_text = f"Row {idx + 1}: " + " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            text_parts.append(row_text)
        
        return "\n".join(text_parts)

    def extract_company_name(self, file_path: str) -> str:
        """Extract company name from Excel file"""
        try:
            xl = pd.ExcelFile(file_path)
            
            # Common sheet names that might contain company name
            priority_sheets = ['cover', 'home', 'title', 'overview', 'about', 'general']
            
            # Common patterns that might indicate company name
            company_patterns = [
                r'(?:company|organization|organisation)\s*name\s*[:]\s*([A-Za-z0-9\s\.\-&]+(?:Ltd|Limited|Inc|Corporation|Corp|Plc|LLC|LLP)?)',
                r'(?:about|for)\s+([A-Za-z0-9\s\.\-&]+(?:Ltd|Limited|Inc|Corporation|Corp|Plc|LLC|LLP))',
                r'([A-Za-z0-9\s\.\-&]+(?:Ltd|Limited|Inc|Corporation|Corp|Plc|LLC|LLP))\s+sustainability\s+report',
                r'([A-Za-z0-9\s\.\-&]+(?:Ltd|Limited|Inc|Corporation|Corp|Plc|LLC|LLP))\s+esg\s+report',
                r'welcome\s+to\s+([A-Za-z0-9\s\.\-&]+(?:Ltd|Limited|Inc|Corporation|Corp|Plc|LLC|LLP)?)',
                r'©\s*([A-Za-z0-9\s\.\-&]+(?:Ltd|Limited|Inc|Corporation|Corp|Plc|LLC|LLP)?)\s+20\d{2}'
            ]
            
            def clean_company_name(name: str) -> str:
                """Clean and validate company name"""
                # Remove common unwanted terms
                unwanted_terms = ['sustainability report', 'esg report', 'annual report', 'report', 
                                'copyright', '©', 'all rights reserved']
                name = name.strip()
                for term in unwanted_terms:
                    name = name.lower().replace(term.lower(), '').strip()
                
                # Remove URLs and email addresses
                name = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', name)
                name = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', name)
                
                # Remove extra whitespace and punctuation at ends
                name = name.strip('.,;:-_/\\')
                name = ' '.join(name.split())
                
                # Capitalize words
                name = ' '.join(word.capitalize() for word in name.split())
                
                return name
            
            # First try priority sheets
            for sheet_name in xl.sheet_names:
                if any(p.lower() in sheet_name.lower() for p in priority_sheets):
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    # Convert all values to string and join
                    text = ' '.join(str(val) for val in df.values.flatten() if pd.notna(val))
                    
                    # Try each pattern
                    for pattern in company_patterns:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            company_name = clean_company_name(match.group(1))
                            if len(company_name) > 2:  # Basic validation
                                logger.info(f"Found company name in {sheet_name}: {company_name}")
                                return company_name
            
            # If not found in priority sheets, try all sheets
            for sheet_name in xl.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Look in the first few rows only for efficiency
                top_rows = df.head(5)
                text = ' '.join(str(val) for val in top_rows.values.flatten() if pd.notna(val))
                
                for pattern in company_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        company_name = clean_company_name(match.group(1))
                        if len(company_name) > 2:  # Basic validation
                            logger.info(f"Found company name in {sheet_name}: {company_name}")
                            return company_name
            
            # If still not found, try to extract from filename
            file_name = Path(file_path).stem
            words = re.split(r'[-_\s]', file_name)
            # Look for words that might be part of company name (exclude common words and years)
            company_words = [w for w in words if not re.match(r'(20\d{2}|data|pack|sustainability|esg|report)', w.lower())]
            if company_words:
                company_name = clean_company_name(' '.join(company_words))
                if len(company_name) > 2:  # Basic validation
                    logger.info(f"Extracted company name from filename: {company_name}")
                    return company_name
            
            # If all attempts fail, use a default name
            logger.warning("Could not find company name in Excel file, using default")
            return "Unknown Company"
            
        except Exception as e:
            logger.error(f"Error extracting company name: {e}")
            return "Unknown Company"

    def find_renewable_energy_target(self, file_path: str) -> Optional[Dict]:
        """
        Enhanced method to find renewable energy targets with context awareness
        Returns a dictionary containing target value, year, and confidence score
        """
        try:
            xl = pd.ExcelFile(file_path)
            target_info = {
                'value': None,
                'year': None,
                'confidence': 0,
                'context': ''
            }

            # Keywords that indicate targets with weights
            target_indicators = {
                r'\btarget\b': 3,
                r'\bgoal\b': 3,
                r'\bcommitment\b': 3,
                r'\baim\b': 2,
                r'\bpledge\b': 2,
                r'\bby\s+\d{4}\b': 3,
                r'\bplan\b': 1,
                r'\bstrategy\b': 1
            }

            # Keywords that indicate current status (to avoid)
            current_indicators = {
                r'\bcurrent\b': -3,
                r'\bachieved\b': -3,
                r'\bactual\b': -3,
                r'\bpresent\b': -2,
                r'\btoday\b': -2,
                r'\b\d{4}\s+performance\b': -2
            }

            # Priority sheets to check
            priority_sheets = [
                'Sustainability',
                'ESG',
                'Environment',
                'Climate',
                'Targets',
                'Goals',
                'Overview',
                'Summary'
            ]

            sheets_to_check = [sheet for sheet in xl.sheet_names if 
                             any(p.lower() in sheet.lower() for p in priority_sheets)]
            sheets_to_check.extend([s for s in xl.sheet_names if s not in sheets_to_check])

            for sheet_name in sheets_to_check:
                df = xl.parse(sheet_name)
                
                # Convert all values to string for pattern matching
                df = df.astype(str)

                # Scan for renewable energy related content
                for idx, row in df.iterrows():
                    row_text = ' '.join(row.astype(str)).lower()
                    
                    # Skip if no renewable energy mention
                    if not any(term in row_text for term in ['renewable', 'clean energy', 'green energy']):
                        continue

                    # Calculate confidence score based on keywords
                    score = 0
                    context_parts = []
                    
                    # Check for target indicators
                    for pattern, weight in target_indicators.items():
                        if re.search(pattern, row_text, re.IGNORECASE):
                            score += weight
                            match = re.search(pattern, row_text, re.IGNORECASE)
                            if match:
                                context_parts.append(match.group(0))

                    # Check for current status indicators (negative score)
                    for pattern, weight in current_indicators.items():
                        if re.search(pattern, row_text, re.IGNORECASE):
                            score += weight

                    # Look for percentage values
                    percentage_matches = re.finditer(
                        r'(\d{1,3}(?:\.\d+)?)\s*%|\b(\d{1,3}(?:\.\d+)?)\s+percent\b',
                        row_text
                    )
                    
                    # Look for target years
                    year_matches = re.finditer(r'\b20[2-5]\d\b', row_text)
                    
                    for match in percentage_matches:
                        value = float(match.group(1) or match.group(2))
                        
                        # Skip unlikely values
                        if value <= 0 or value > 100:
                            continue
                            
                        # Look for year in proximity
                        years = [int(y.group()) for y in year_matches]
                        target_year = min(years) if years else None
                        
                        # Adjust confidence based on value reasonableness
                        if 10 <= value <= 100:  # Most renewable targets are in this range
                            score += 2
                        
                        # If this is a better match than what we have
                        if score > target_info['confidence']:
                            target_info.update({
                                'value': value,
                                'year': target_year,
                                'confidence': score,
                                'context': ' | '.join(context_parts)
                            })
                            logger.info(f"Found potential renewable target: {value}% by {target_year} "
                                      f"(confidence: {score}, context: {' | '.join(context_parts)})")

            # Only return if we have a reasonably confident match
            if target_info['confidence'] >= 3 and target_info['value'] is not None:
                return target_info
            return None

        except Exception as e:
            logger.error(f"Error finding renewable energy target: {str(e)}")
            return None

    def calculate_renewable_energy_gap(self, current: Optional[float], target_info: Optional[Dict]) -> Optional[float]:
        """
        Calculate the gap between current renewable energy usage and target
        """
        try:
            if current is None or target_info is None or target_info.get('value') is None:
                return None
            return target_info['value'] - current
        except Exception as e:
            logger.error(f"Error calculating renewable energy gap: {str(e)}")
            return None

    def process_excel(self, file_path: str, company_code: str) -> Dict:
        """Process an Excel data pack and extract ESG metrics using AI"""
        file_path = Path(file_path)
        logger.info(f"Processing Excel data pack: {file_path}")
        
        try:
            # Extract company name from Excel file
            logger.info("Extracting company name...")
            company_name = self.extract_company_name(file_path)
            
            # Add company to database
            logger.info("Adding company to database...")
            company_id = self.db_manager.add_company(company_name, company_code)
            
            # Extract year from filename
            year_match = re.search(r'20\d{2}', file_path.name)
            year = int(year_match.group(0)) if year_match else 2024
            
            # Add or update document in database
            logger.info("Adding/updating document in database...")
            document_id, is_new = self.db_manager.add_or_update_document(
                company_id=company_id,
                file_name=file_path.name,
                file_path=str(file_path),
                document_type='sustainability_data',
                reporting_year=year
            )
            
            logger.info(f"{'Added new' if is_new else 'Updated existing'} document with ID: {document_id}")
            
            # Initialize all_metrics with company name and year
            all_metrics = {
                'company_name': company_name,
                'year': str(year)
            }
            
            # Read Excel file
            logger.info("Reading Excel sheets...")
            xl = pd.ExcelFile(file_path)
            
            # Score and sort sheets based on relevance
            logger.info("Analyzing sheet relevance...")
            scored_sheets = self._score_sheets(xl.sheet_names)
            
            # Process only the top relevant sheets (max 5 sheets)
            processed_sheets = 0
            max_sheets = 5
            
            for sheet_name, score in scored_sheets[:max_sheets]:
                logger.info(f"Processing sheet {processed_sheets + 1}/{max_sheets}: {sheet_name} (relevance score: {score})")
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Skip empty or very small sheets
                if df.empty or len(df) < 2:
                    continue
                
                try:
                    # Use Groq for metric extraction
                    sheet_metrics = self._analyze_with_groq(self._df_to_text(df), sheet_name)
                    
                    # Update metrics if we got valid results
                    if sheet_metrics and isinstance(sheet_metrics, dict):
                        for key, value in sheet_metrics.items():
                            if value is not None:
                                # For numeric metrics, keep the highest confidence ones
                                if key in all_metrics and isinstance(value, (int, float)):
                                    if value > all_metrics[key]:
                                        all_metrics[key] = value
                                        logger.info(f"Updated {key} with higher value: {value}")
                                else:
                                    all_metrics[key] = value
                                    logger.info(f"Added new metric {key}: {value}")
                    
                except Exception as e:
                    logger.error(f"Error processing sheet {sheet_name}: {e}")
                    continue
                
                processed_sheets += 1
            
            # Find renewable energy target after processing sheets
            logger.info("Looking for renewable energy targets...")
            renewable_target = self.find_renewable_energy_target(file_path)
            if renewable_target:
                all_metrics['renewable_energy_target'] = renewable_target['value']
                all_metrics['target_year'] = renewable_target['year']
                all_metrics['renewable_energy_gap'] = self.calculate_renewable_energy_gap(
                    all_metrics.get('renewable_energy_percentage'),
                    renewable_target
                )
            
            # Store metrics in database
            logger.info("Storing metrics in database...")
            if is_new:
                self.db_manager.add_metrics(document_id, all_metrics)
            else:
                self.db_manager.update_metrics(document_id, all_metrics)
            
            logger.info(f"Successfully processed Excel data pack. Found metrics: {all_metrics}")
            return all_metrics
            
        except Exception as e:
            logger.error(f"Error processing Excel data pack: {e}")
            if 'document_id' in locals():
                self.db_manager.log_processing(
                    document_id,
                    "error",
                    f"Processing failed: {str(e)}"
                )
            return None

    def _score_sheets(self, sheet_names: List[str]) -> List[tuple]:
        """Score and sort sheets based on relevance"""
        # Define priority sheets and their weights
        priority_sheets = {
            'energy': 10,
            'emissions': 10,
            'ghg': 10,
            'climate': 9,
            'environment': 8,
            'esg': 8,
            'sustainability': 7,
            'performance': 6,
            'metrics': 6,
            'data': 5
        }
        
        scored_sheets = []
        for sheet_name in sheet_names:
            sheet_lower = sheet_name.lower()
            score = 0
            
            # Calculate score based on priority keywords
            for keyword, weight in priority_sheets.items():
                if keyword in sheet_lower:
                    score += weight
            
            # Additional score for sheets with specific ESG-related terms
            if any(term in sheet_lower for term in ['scope', 'target', 'renewable', 'carbon']):
                score += 5
            
            if score > 0:
                scored_sheets.append((sheet_name, score))
        
        # Sort sheets by score in descending order
        return sorted(scored_sheets, key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python process_excel_data.py <excel_file> <company_code>")
        sys.exit(1)
        
    processor = ExcelDataProcessor()
    metrics = processor.process_excel(sys.argv[1], sys.argv[2])
    if metrics:
        print("Successfully extracted metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}") 