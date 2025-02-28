import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from openai import OpenAI
import json
from concurrent.futures import ThreadPoolExecutor
import time
from random import uniform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ESGMetrics(BaseModel):
    """Schema for ESG metrics extracted from Excel files"""
    scope1_emissions: Optional[float] = Field(None, description="Scope 1 direct GHG emissions")
    scope2_emissions: Optional[float] = Field(None, description="Scope 2 indirect GHG emissions")
    scope3_emissions: Optional[float] = Field(None, description="Scope 3 value chain GHG emissions")
    emissions_unit: Optional[str] = Field(None, description="Unit for emissions data")
    renewable_energy_percentage: Optional[float] = Field(None, description="Current renewable energy percentage")
    energy_consumption: Optional[float] = Field(None, description="Total energy consumption")
    energy_unit: Optional[str] = Field(None, description="Unit for energy consumption")
    renewable_energy_target: Optional[float] = Field(None, description="Target percentage for renewable energy")
    target_year: Optional[int] = Field(None, description="Year for achieving renewable energy target")
    base_year: Optional[int] = Field(None, description="Base year for emissions/energy calculations")
    confidence_score: Optional[float] = Field(None, description="AI model's confidence in extracted values (0-1)")

class ExcelProcessor:
    def __init__(self):
        """Initialize the Excel processor with OpenAI client and system prompt"""
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4-turbo-preview')
        
        # System prompt for direct interpretation of Excel data
        self.system_prompt = """You are an expert in analyzing ESG (Environmental, Social, and Governance) data 
        directly from Excel files. Your task is to interpret raw Excel content and extract key ESG metrics 
        without any preprocessing.

        Guidelines:
        1. Analyze the content as-is, using context to understand the meaning of cells and their relationships
        2. Look for both explicit metrics and implied values from contextual information
        3. Consider the entire sheet's context when interpreting individual values
        4. Provide confidence scores for each extracted metric
        5. Handle different units and formats naturally through contextual understanding
        6. Note any ambiguities or multiple possible interpretations
        7. Convert all values to standard units when confident about the conversion

        For each metric, explain your reasoning and provide a confidence score (0-1).
        """

    def _convert_sheet_to_context(self, df: pd.DataFrame, sheet_name: str) -> str:
        """Convert a DataFrame to a contextual string representation"""
        # Convert DataFrame to string while preserving structure
        context = f"Sheet Name: {sheet_name}\n\n"
        
        # Add column headers
        headers = df.columns.tolist()
        context += "Headers: " + " | ".join(str(h) for h in headers) + "\n\n"
        
        # Convert data to string representation
        for idx, row in df.iterrows():
            row_str = " | ".join(str(val) if pd.notna(val) else "" for val in row)
            if row_str.strip():  # Only add non-empty rows
                context += f"Row {idx}: {row_str}\n"
        
        return context

    def _extract_metrics_with_ai(self, context: str) -> Dict:
        """Use AI to extract metrics from the context"""
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": (
                    "Please analyze this Excel content and extract ESG metrics. "
                    "Provide your reasoning and confidence for each metric.\n\n"
                    f"{context}"
                )}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parse the response and extract metrics
            result = json.loads(response.choices[0].message.content)
            
            # Convert to ESGMetrics model
            metrics = ESGMetrics(**result)
            return metrics.model_dump(exclude_none=True)
            
        except Exception as e:
            logger.error(f"Error in AI metric extraction: {e}")
            return {}

    def process_excel_file(self, file_path: str) -> Dict:
        """Process an Excel file and extract ESG metrics using AI interpretation"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Excel file not found: {file_path}")
            
            logger.info(f"Processing Excel file: {file_path}")
            
            # Read all sheets from the Excel file
            excel_file = pd.ExcelFile(file_path)
            sheets_data = {}
            
            # Process each sheet
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    
                    # Convert sheet to context
                    context = self._convert_sheet_to_context(df, sheet_name)
                    
                    # Extract metrics using AI
                    metrics = self._extract_metrics_with_ai(context)
                    
                    if metrics:
                        sheets_data[sheet_name] = metrics
                        
                except Exception as e:
                    logger.error(f"Error processing sheet {sheet_name}: {e}")
                    continue
            
            # Combine metrics from all sheets
            combined_metrics = self._combine_sheet_metrics(sheets_data)
            return combined_metrics
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return {}

    def _combine_sheet_metrics(self, sheets_data: Dict[str, Dict]) -> Dict:
        """Combine metrics from multiple sheets, using confidence scores to resolve conflicts"""
        if not sheets_data:
            return {}
        
        # Initialize combined metrics
        combined = {}
        confidence_scores = {}
        
        # Combine metrics from all sheets
        for sheet_metrics in sheets_data.values():
            for key, value in sheet_metrics.items():
                if key == 'confidence_score':
                    continue
                    
                current_confidence = sheet_metrics.get('confidence_score', 0.5)
                
                # Update metric if it has higher confidence
                if key not in combined or current_confidence > confidence_scores.get(key, 0):
                    combined[key] = value
                    confidence_scores[key] = current_confidence
        
        # Add overall confidence score
        if confidence_scores:
            combined['confidence_score'] = sum(confidence_scores.values()) / len(confidence_scores)
        
        return combined

def process_directory(directory_path: str) -> Dict[str, Dict]:
    """Process all Excel files in a directory"""
    processor = ExcelProcessor()
    results = {}
    
    directory = Path(directory_path)
    excel_files = list(directory.glob("**/*.xls*"))
    
    logger.info(f"Found {len(excel_files)} Excel files to process")
    
    # Process files in parallel
    with ThreadPoolExecutor() as executor:
        future_to_file = {
            executor.submit(processor.process_excel_file, str(file)): file
            for file in excel_files
        }
        
        for future in future_to_file:
            file = future_to_file[future]
            try:
                results[str(file)] = future.result()
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
                results[str(file)] = {}
    
    return results

if __name__ == "__main__":
    # Process single file
    file_path = "/Users/kluless/esgWIKI/crawler/tests/test_output/downloads/2024-sustainability-data-pack.xlsx"
    processor = ExcelProcessor()
    results = processor.process_excel_file(file_path)
    
    # Save results to JSON file
    output_file = "esg_metrics_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
