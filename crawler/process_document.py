from pathlib import Path
import re
from typing import Dict, Optional
from utils.db_manager import ESGDatabaseManager
from utils.scraper_utils import get_llm_strategy
from langchain_community.document_loaders import PyPDFLoader
import logging
import json
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, db_manager: ESGDatabaseManager = None):
        self.db_manager = db_manager or ESGDatabaseManager()
        self.llm_strategy = get_llm_strategy()
        
    def extract_year_from_filename(self, filename: str) -> Optional[int]:
        """Extract year from filename"""
        year_match = re.search(r'20\d{2}', filename)
        return int(year_match.group(0)) if year_match else None
        
    def determine_document_type(self, filename: str) -> str:
        """Determine document type from filename"""
        filename = filename.lower()
        if 'climate' in filename:
            return 'climate_report'
        elif 'sustainability' in filename or 'esg' in filename:
            return 'sustainability_report'
        elif 'annual' in filename:
            return 'annual_report'
        else:
            return 'other'
            
    def process_pdf(self, file_path: str, company_name: str, company_code: str) -> Dict:
        """Process a PDF document and extract ESG metrics"""
        file_path = Path(file_path)
        logger.info(f"Processing document: {file_path}")
        
        try:
            # Add company to database
            company_id = self.db_manager.add_company(company_name, company_code)
            
            # Extract year and document type
            year = self.extract_year_from_filename(file_path.name)
            doc_type = self.determine_document_type(file_path.name)
            
            if not year:
                logger.warning(f"Could not extract year from filename: {file_path.name}")
                year = 2024  # Default to current year
                
            # Add or update document in database
            document_id, is_new = self.db_manager.add_or_update_document(
                company_id=company_id,
                file_name=file_path.name,
                file_path=str(file_path),
                document_type=doc_type,
                reporting_year=year
            )
            
            logger.info(f"{'Added new' if is_new else 'Updated existing'} document with ID: {document_id}")
            
            # Load and process PDF
            logger.info("Loading PDF content...")
            loader = PyPDFLoader(str(file_path))
            pages = loader.load()
            
            # Skip table of contents (first few pages) and find relevant sections
            metrics_section = ""
            current_section = ""
            for i, page in enumerate(pages):
                # Skip first few pages (table of contents)
                if i < 3:
                    continue
                    
                content = page.page_content
                logger.info(f"Processing page {i + 1}")
                logger.info(f"Page content preview: {content[:200]}")
                
                # Look for section headers
                if "Metrics and targets" in content:
                    current_section = "metrics"
                    logger.info("Found metrics section")
                elif "Performance summary" in content:
                    current_section = "performance"
                    logger.info("Found performance section")
                elif "Sector decarbonisation" in content:
                    current_section = "decarbonisation"
                    logger.info("Found decarbonisation section")
                elif "Environmental finance" in content:
                    current_section = "finance"
                    logger.info("Found finance section")
                    
                # Collect content from relevant sections
                if current_section in ["metrics", "performance", "decarbonisation", "finance"]:
                    metrics_section += content + "\n"
            
            # Use the metrics section if found, otherwise use full text
            text = metrics_section if metrics_section else "\n".join(page.page_content for page in pages[3:])
            
            # Debug: Print first 2000 characters of content
            logger.info("First 2000 characters of processed content:")
            logger.info(text[:2000])
            
            # Extract metrics using LLM
            logger.info("Extracting metrics using LLM...")
            metrics_json = self.llm_strategy.extract(text)[0]
            
            # Parse the JSON string into a dictionary
            try:
                # Clean up the JSON string if needed
                metrics_json = metrics_json.strip()
                if metrics_json.startswith("```json"):
                    metrics_json = metrics_json[7:]
                if metrics_json.endswith("```"):
                    metrics_json = metrics_json[:-3]
                    
                metrics = json.loads(metrics_json)  # Use json.loads instead of eval
                
                # Update metrics in database
                logger.info("Storing metrics in database...")
                if is_new:
                    self.db_manager.add_metrics(document_id, metrics)
                else:
                    self.db_manager.update_metrics(document_id, metrics)
                
                # Log success
                self.db_manager.log_processing(
                    document_id,
                    "success",
                    f"Document {'processed' if is_new else 'updated'} successfully"
                )
                
                return metrics
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing metrics JSON: {e}")
                logger.error(f"Raw JSON string: {metrics_json}")
                self.db_manager.log_processing(
                    document_id, 
                    "error",
                    f"Failed to parse metrics JSON: {str(e)}"
                )
                return None
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            if 'document_id' in locals():
                self.db_manager.log_processing(
                    document_id,
                    "error",
                    f"Processing failed: {str(e)}"
                )
            return None

def process_cba_report():
    """Process the CBA 2024 Climate Report"""
    processor = DocumentProcessor()
    
    # Process the CBA report
    file_path = Path(__file__).parent / 'downloads' / 'CBA-2024-Climate-Report.pdf'
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return
        
    metrics = processor.process_pdf(
        file_path=str(file_path),
        company_name="Commonwealth Bank of Australia",
        company_code="CBA"
    )
    
    if metrics:
        logger.info("Successfully processed CBA report")
        logger.info("Extracted metrics:")
        for key, value in metrics.items():
            logger.info(f"{key}: {value}")
    else:
        logger.error("Failed to process CBA report")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python process_document.py <pdf_file> <company_name> <company_code>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    company_name = sys.argv[2]
    company_code = sys.argv[3]
    
    processor = DocumentProcessor()
    metrics = processor.process_pdf(
        file_path=file_path,
        company_name=company_name,
        company_code=company_code
    )
    
    if metrics:
        print("Successfully processed document")
        print("Extracted metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")
    else:
        print("Failed to process document") 