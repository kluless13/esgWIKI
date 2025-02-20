import sqlite3
import os
from pathlib import Path

def setup_database():
    """Create SQLite database and tables for ESG metrics storage"""
    # Get the path to the database file
    db_path = Path(__file__).parent.parent / 'data' / 'esg_metrics.db'
    db_path.parent.mkdir(exist_ok=True)
    
    # Connect to SQLite database (creates it if it doesn't exist)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Create tables
    cursor.executescript("""
        -- Companies table
        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            code TEXT UNIQUE,  -- ASX code
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Documents table
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            document_type TEXT NOT NULL,  -- 'climate_report', 'annual_report', 'sustainability_report'
            reporting_year INTEGER NOT NULL,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (company_id) REFERENCES companies(id),
            UNIQUE(file_path)
        );

        -- ESG Metrics table
        CREATE TABLE IF NOT EXISTS esg_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            -- Emissions data
            scope1_emissions REAL,
            scope2_emissions REAL,
            scope3_emissions REAL,
            emissions_unit TEXT,
            emissions_base_year TEXT,
            
            -- Energy & Renewables
            renewable_energy_percentage REAL,
            renewable_energy_target REAL,
            target_year TEXT,
            
            -- Climate Targets
            emission_reduction_target REAL,
            emission_reduction_base_year TEXT,
            current_reduction_percentage REAL,
            net_zero_commitment_year TEXT,
            carbon_neutral_certified BOOLEAN,
            
            -- Financial Metrics
            internal_carbon_price REAL,
            sustainable_finance_target REAL,
            climate_related_investment REAL,
            
            -- Metadata
            confidence_score REAL,
            extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        );

        -- Processing Log table
        CREATE TABLE IF NOT EXISTS processing_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            status TEXT NOT NULL,  -- 'success', 'error', 'warning'
            message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        );
    """)
    
    conn.commit()
    conn.close()
    
    print(f"Database created successfully at {db_path}")
    return str(db_path)

if __name__ == "__main__":
    db_path = setup_database() 