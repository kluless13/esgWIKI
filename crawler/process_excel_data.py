import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from utils.db_manager import ESGDatabaseManager
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExcelDataProcessor:
    def __init__(self, db_manager: ESGDatabaseManager = None):
        self.db_manager = db_manager or ESGDatabaseManager()
        
    def clean_numeric_value(self, value: str) -> float:
        """Clean and convert string to numeric value"""
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            # Remove asterisks and other special characters
            clean_value = value.replace('*', '').replace(',', '').replace('$', '').replace('%', '').strip()
            try:
                return float(clean_value)
            except (ValueError, TypeError):
                return None
        return None

    def extract_metrics_from_sheet(self, df: pd.DataFrame, sheet_name: str) -> Dict:
        """Extract metrics from a specific sheet"""
        metrics = {}
        
        # Skip non-relevant sheets
        if sheet_name not in ['GHG Emissions', 'Energy', 'Position']:
            return metrics
            
        # Convert column names to strings and clean them
        df.columns = df.columns.astype(str)
        df.columns = [col.strip() for col in df.columns]
        
        # Reset index if it's not numeric
        if not isinstance(df.index, pd.RangeIndex):
            df = df.reset_index()
            
        # For GHG Emissions sheet
        if sheet_name == 'GHG Emissions':
            # Look for scope emissions in rows
            for idx, row in df.iterrows():
                row_str = str(row.iloc[0]).lower() if not pd.isna(row.iloc[0]) else ''
                
                # Scope 1
                if 'scope 1' in row_str or 'scope1' in row_str.replace(' ', ''):
                    values = row.dropna()
                    if len(values) > 1:
                        val = self.clean_numeric_value(values.iloc[-1])
                        if val is not None:
                            metrics['scope1_emissions'] = val
                        
                # Scope 2
                elif 'scope 2' in row_str or 'scope2' in row_str.replace(' ', ''):
                    values = row.dropna()
                    if len(values) > 1:
                        val = self.clean_numeric_value(values.iloc[-1])
                        if val is not None:
                            metrics['scope2_emissions'] = val
                        
                # Scope 3
                elif 'scope 3' in row_str or 'scope3' in row_str.replace(' ', ''):
                    values = row.dropna()
                    if len(values) > 1:
                        val = self.clean_numeric_value(values.iloc[-1])
                        if val is not None:
                            metrics['scope3_emissions'] = val
                        
                # Look for units
                elif any(term in row_str for term in ['unit', 'measurement', 'metric']):
                    values = row.dropna()
                    if len(values) > 1:
                        unit_val = str(values.iloc[-1]).lower()
                        if any(term in unit_val for term in ['tco2', 'co2', 'carbon']):
                            metrics['emissions_unit'] = unit_val
                        
                # Look for base year
                elif 'base' in row_str and 'year' in row_str:
                    values = row.dropna()
                    if len(values) > 1:
                        year_val = values.iloc[-1]
                        if isinstance(year_val, (int, float)):
                            metrics['emissions_base_year'] = int(year_val)
                        elif isinstance(year_val, str):
                            # Try to extract year from string
                            year_match = re.search(r'20\d{2}', year_val)
                            if year_match:
                                metrics['emissions_base_year'] = int(year_match.group(0))
                        
        # For Energy sheet
        elif sheet_name == 'Energy':
            for idx, row in df.iterrows():
                row_str = str(row.iloc[0]).lower() if not pd.isna(row.iloc[0]) else ''
                
                # Renewable energy percentage - look for more specific patterns
                if any(term in row_str for term in [
                    'renewable energy consumption',
                    'renewable electricity',
                    'renewable power',
                    'green power',
                    '% renewable'
                ]):
                    values = row.dropna()
                    if len(values) > 1:
                        val = values.iloc[-1]
                        if isinstance(val, (int, float)):
                            if val <= 100:  # Sanity check for percentage
                                metrics['renewable_energy_percentage'] = float(val)
                        elif isinstance(val, str):
                            # Try to extract percentage
                            if '%' in val:
                                clean_val = self.clean_numeric_value(val)
                                if clean_val is not None and clean_val <= 100:
                                    metrics['renewable_energy_percentage'] = clean_val
                            else:
                                # Try to find percentage in the row description
                                pct_match = re.search(r'(\d+(?:\.\d+)?)%', row_str)
                                if pct_match:
                                    pct_val = float(pct_match.group(1))
                                    if pct_val <= 100:
                                        metrics['renewable_energy_percentage'] = pct_val
                
                # Look for energy consumption data
                elif any(term in row_str for term in [
                    'total energy consumption',
                    'energy use',
                    'electricity consumption'
                ]):
                    values = row.dropna()
                    if len(values) > 1:
                        val = self.clean_numeric_value(values.iloc[-1])
                        if val is not None:
                            metrics['total_energy_consumption'] = val
                            
                            # Look for units in the row
                            unit_match = re.search(r'([kMG]Wh|joules?|MJ)', row_str, re.IGNORECASE)
                            if unit_match:
                                metrics['energy_consumption_unit'] = unit_match.group(1)
        
        # For Position sheet (strategic targets)
        elif sheet_name == 'Position':
            for idx, row in df.iterrows():
                row_str = str(row.iloc[0]).lower() if not pd.isna(row.iloc[0]) else ''
                
                # Net zero commitment
                if 'net zero' in row_str or 'carbon neutral' in row_str:
                    values = row.dropna()
                    if len(values) > 1:
                        val = values.iloc[-1]
                        if isinstance(val, (int, float)):
                            metrics['net_zero_commitment_year'] = int(val)
                        elif isinstance(val, str):
                            year_match = re.search(r'20\d{2}', val)
                            if year_match:
                                metrics['net_zero_commitment_year'] = int(year_match.group(0))
                        
                # Emission reduction targets
                elif 'reduction target' in row_str or 'emissions target' in row_str:
                    values = row.dropna()
                    if len(values) > 1:
                        target_val = values.iloc[-1]
                        if isinstance(target_val, (int, float)):
                            metrics['emission_reduction_target'] = float(target_val)
                        elif isinstance(target_val, str):
                            clean_val = self.clean_numeric_value(target_val)
                            if clean_val is not None:
                                metrics['emission_reduction_target'] = clean_val
                                
                            # Look for target year in the same string
                            year_match = re.search(r'20\d{2}', target_val)
                            if year_match:
                                metrics['target_year'] = int(year_match.group(0))
                        
                # Sustainable finance
                elif 'sustainable finance' in row_str or 'green finance' in row_str:
                    values = row.dropna()
                    if len(values) > 1:
                        finance_val = values.iloc[-1]
                        if isinstance(finance_val, (int, float)):
                            metrics['sustainable_finance_target'] = float(finance_val)
                        elif isinstance(finance_val, str):
                            clean_val = self.clean_numeric_value(finance_val)
                            if clean_val is not None:
                                metrics['sustainable_finance_target'] = clean_val
        
        return metrics
        
    def process_excel(self, file_path: str, company_name: str, company_code: str) -> Dict:
        """Process an Excel data pack and extract ESG metrics"""
        file_path = Path(file_path)
        logger.info(f"Processing Excel data pack: {file_path}")
        
        try:
            # Add company to database
            company_id = self.db_manager.add_company(company_name, company_code)
            
            # Extract year from filename
            year_match = re.search(r'20\d{2}', file_path.name)
            year = int(year_match.group(0)) if year_match else 2024
            
            # Add or update document in database
            document_id, is_new = self.db_manager.add_or_update_document(
                company_id=company_id,
                file_name=file_path.name,
                file_path=str(file_path),
                document_type='sustainability_data',
                reporting_year=year
            )
            
            logger.info(f"{'Added new' if is_new else 'Updated existing'} document with ID: {document_id}")
            
            # Read all sheets from Excel file
            logger.info("Reading Excel sheets...")
            xl = pd.ExcelFile(file_path)
            
            # Process each sheet and combine metrics
            all_metrics = {}
            for sheet_name in xl.sheet_names:
                logger.info(f"Processing sheet: {sheet_name}")
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Skip empty sheets
                if df.empty:
                    continue
                    
                # Extract metrics from this sheet
                sheet_metrics = self.extract_metrics_from_sheet(df, sheet_name)
                
                # Update all_metrics with non-null values
                all_metrics.update({k: v for k, v in sheet_metrics.items() if v is not None})
            
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

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python process_excel_data.py <excel_file> <company_name> <company_code>")
        sys.exit(1)
        
    processor = ExcelDataProcessor()
    metrics = processor.process_excel(sys.argv[1], sys.argv[2], sys.argv[3])
    if metrics:
        print("Successfully extracted metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}") 