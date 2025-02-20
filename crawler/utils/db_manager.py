import sqlite3
from pathlib import Path
from typing import Dict, Optional, Union, Tuple
import json

class ESGDatabaseManager:
    def __init__(self, db_path: Union[str, Path] = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent / 'data' / 'esg_metrics.db'
        self.db_path = Path(db_path)
        
    def get_connection(self):
        """Get a database connection"""
        return sqlite3.connect(str(self.db_path))
        
    def add_company(self, name: str, code: str) -> int:
        """Add a company to the database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO companies (name, code) VALUES (?, ?)",
                (name, code)
            )
            conn.commit()
            
            # Get the company ID (whether newly inserted or existing)
            cursor.execute(
                "SELECT id FROM companies WHERE code = ?",
                (code,)
            )
            return cursor.fetchone()[0]
            
    def get_document_id(self, file_path: str) -> Optional[int]:
        """Get document ID if it exists"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM documents WHERE file_path = ?",
                (file_path,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
            
    def add_or_update_document(self, 
                             company_id: int,
                             file_name: str,
                             file_path: str,
                             document_type: str,
                             reporting_year: int) -> Tuple[int, bool]:
        """Add or update a document in the database. Returns (document_id, is_new)"""
        existing_id = self.get_document_id(file_path)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if existing_id:
                # Update existing document
                cursor.execute("""
                    UPDATE documents 
                    SET company_id = ?, file_name = ?, document_type = ?, 
                        reporting_year = ?, processed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (company_id, file_name, document_type, reporting_year, existing_id))
                conn.commit()
                return existing_id, False
            else:
                # Insert new document
                cursor.execute("""
                    INSERT INTO documents 
                    (company_id, file_name, file_path, document_type, reporting_year)
                    VALUES (?, ?, ?, ?, ?)
                """, (company_id, file_name, file_path, document_type, reporting_year))
                conn.commit()
                return cursor.lastrowid, True
            
    def update_metrics(self, document_id: int, metrics: Dict, confidence_score: float = None) -> int:
        """Update metrics for a document"""
        # First, remove old metrics
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM esg_metrics WHERE document_id = ?", (document_id,))
            conn.commit()
        
        # Then add new metrics
        return self.add_metrics(document_id, metrics, confidence_score)
            
    def add_metrics(self, document_id: int, metrics: Dict, confidence_score: float = None) -> int:
        """Add ESG metrics to the database"""
        # Prepare the metrics data
        metric_data = {
            'document_id': document_id,
            'scope1_emissions': metrics.get('scope1_emissions'),
            'scope2_emissions': metrics.get('scope2_emissions'),
            'scope3_emissions': metrics.get('scope3_emissions'),
            'emissions_unit': metrics.get('emissions_unit'),
            'emissions_base_year': metrics.get('emissions_base_year'),
            'renewable_energy_percentage': metrics.get('renewable_energy_percentage'),
            'renewable_energy_target': metrics.get('renewable_energy_target'),
            'target_year': metrics.get('target_year'),
            'emission_reduction_target': metrics.get('emission_reduction_target'),
            'emission_reduction_base_year': metrics.get('emission_reduction_base_year'),
            'current_reduction_percentage': metrics.get('current_reduction_percentage'),
            'net_zero_commitment_year': metrics.get('net_zero_commitment_year'),
            'carbon_neutral_certified': metrics.get('carbon_neutral_certified'),
            'internal_carbon_price': metrics.get('internal_carbon_price'),
            'sustainable_finance_target': metrics.get('sustainable_finance_target'),
            'climate_related_investment': metrics.get('climate_related_investment'),
            'confidence_score': confidence_score
        }
        
        # Build the SQL query dynamically based on available metrics
        fields = [k for k, v in metric_data.items() if v is not None]
        placeholders = ['?' for _ in fields]
        values = [metric_data[f] for f in fields]
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                INSERT INTO esg_metrics 
                ({', '.join(fields)})
                VALUES ({', '.join(placeholders)})
            """, values)
            conn.commit()
            return cursor.lastrowid
            
    def log_processing(self, document_id: int, status: str, message: str = None):
        """Log document processing status"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO processing_log 
                (document_id, status, message)
                VALUES (?, ?, ?)
            """, (document_id, status, message))
            conn.commit()
            
    def get_company_metrics(self, company_code: str) -> Dict:
        """Get all metrics for a company"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    c.name as company_name,
                    c.code as company_code,
                    d.document_type,
                    d.reporting_year,
                    m.*
                FROM companies c
                JOIN documents d ON c.id = d.company_id
                JOIN esg_metrics m ON d.id = m.document_id
                WHERE c.code = ?
                ORDER BY d.reporting_year DESC
            """, (company_code,))
            
            columns = [col[0] for col in cursor.description]
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            return results
            
    def get_latest_metrics(self, company_code: str) -> Optional[Dict]:
        """Get the most recent metrics for a company"""
        metrics = self.get_company_metrics(company_code)
        return metrics[0] if metrics else None 