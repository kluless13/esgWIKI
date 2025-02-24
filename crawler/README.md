# ESG Data Crawler

# Instructions

Before running any commands, make sure you're in the project directory:
```bash
cd /path/to/esgWIKI/crawler
```

1. Install Requirements:
```bash
python -m pip install -r requirements.txt
```

2. Run Test Crawler (processes first 3 companies):
```bash
# This will create test output files in tests/test_output/
python -m tests.test_main collect
```

3. Run Test Downloader:
```bash
# Download all files to tests/test_output/downloads/
python -m tests.test_downloader

# Or limit to first 10 files per company
python -m tests.test_downloader 10
```

All test outputs will be stored in the `tests/test_output/` directory to keep test artifacts separate from production data.

---

This project crawls and collects ESG (Environmental, Social, and Governance) reports from Australian Stock Exchange (ASX) listed companies.

## Overview

The crawler uses a two-phase approach:
1. Collection Phase: Identifies and collects links to ESG-related reports and documents
2. Processing Phase: Downloads and processes the collected documents

## Project Structure

```
crawler/
├── main.py                     # Main crawler script for all companies
├── download_reports.py         # Downloads reports from collected URLs
├── process_document.py         # Processes downloaded PDFs
├── process_excel_data.py       # Extracts data from Excel files
├── initialize_vector_store.py  # Sets up vector storage for document processing
├── config.py                   # Configuration settings
├── requirements.txt           # Python dependencies
├── companies-list.csv        # List of companies to crawl
│
├── output/                   # Production output directory
│   ├── downloads/           # Downloaded reports
│   ├── sustainability_data.json
│   ├── sustainability_summary.json
│   └── report_link.txt
│
├── tests/                   # Test files and test data
│   ├── test_main.py        # Test crawler (first 3 companies)
│   ├── test_downloader.py  # Test report downloader
│   ├── test_excel_download.py
│   └── test_output/        # Test artifacts directory
│       ├── downloads/      # Test downloads
│       ├── sustainability_data.json
│       ├── sustainability_summary.json
│       └── report_link.txt
│
└── utils/                  # Utility modules
    ├── scraper_utils.py   # Crawler utility functions
    ├── download_manager.py # Download handling
    ├── vector_store.py    # Vector storage utilities
    ├── db_manager.py      # Database management
    └── db_setup.py        # Database initialization

```

## Output Structure

### Production Output (`/output/`)
- `downloads/`: Downloaded PDF and Excel files
- `sustainability_data.json`: Detailed data about each company and their reports
- `sustainability_summary.json`: Summary statistics for each company
- `report_link.txt`: List of all report URLs found

### Test Output (`/tests/test_output/`)
- Same structure as production output but for test data
- Contains results from processing first 3 companies
- Used for testing and validation

## Current Status and TODOs

### Implemented Features
- ✅ Company website crawling
- ✅ Report URL collection
- ✅ PDF and Excel file downloading
- ✅ Company-wise download limiting
- ✅ Duplicate file detection
- ✅ Basic error handling

### In Progress
- 🔄 Excel file processing
- 🔄 PDF text extraction
- 🔄 ESG metrics collection

### TODO List
1. Document Processing
   - [ ] Implement PDF text extraction pipeline
   - [ ] Add OCR support for scanned PDFs
   - [ ] Create structured data from extracted text
   - [ ] Implement table extraction from PDFs

2. Excel Processing
   - [ ] Add support for various Excel formats
   - [ ] Create standardized data extraction
   - [ ] Handle different sheet structures
   - [ ] Implement data validation

3. Performance Improvements
   - [ ] Add parallel processing for downloads
   - [ ] Implement rate limiting
   - [ ] Add download resume capability
   - [ ] Optimize memory usage for large files

4. Error Handling
   - [ ] Add comprehensive error logging
   - [ ] Implement retry mechanisms
   - [ ] Add error recovery for interrupted processes
   - [ ] Create error reports

5. Testing
   - [ ] Add unit tests for downloaders
   - [ ] Add integration tests
   - [ ] Create test data sets
   - [ ] Add performance benchmarks

6. Documentation
   - [ ] Add API documentation
   - [ ] Create troubleshooting guide
   - [ ] Add examples for common use cases
   - [ ] Document configuration options

## Usage Guide

### 1. Environment Setup
Create a `.env` file in the crawler directory:
```bash
cp .env.example .env
```

Add your API keys:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 2. Testing the Pipeline
1. Run test crawler (first 3 companies):
```bash
python -m tests.test_main collect
```

2. Test downloading reports:
```bash
# Download 10 files per company
python -m tests.test_downloader 10
```

### 3. Production Run
1. Collect all company reports:
```bash
python main.py collect
```

2. Download reports:
```bash
# Download 10 files per company
python download_reports.py 10

# Or download all files
python download_reports.py
```

### 4. Data Processing
Process Excel files:
```bash
python process_excel_data.py <excel_file> <company_name> <company_code>
```

Example:
```bash
python process_excel_data.py "output/downloads/2024-sustainability-data-pack.xlsx" "Commonwealth Bank of Australia" "CBA"
```

## Configuration

The crawler can be configured through:
- `config.py`: Core settings
- `.env`: API keys and sensitive data
- Command-line arguments: Runtime options (e.g., download limits)

## Key Features

- Crawls company websites through ListCorp.com
- Identifies corporate governance, sustainability, and ESG sections
- Collects links to:
  - Sustainability reports
  - ESG reports
  - Annual reports
  - Climate reports
  - Environmental reports
  - Carbon disclosure documents
  - ESG data spreadsheets

## Dependencies

Required Python packages are listed in `requirements.txt`. Install using:
```bash
pip install -r requirements.txt
```

## Testing

The test version (`test_main.py`) mirrors the functionality of `main.py` but only processes the first three companies from the list. This is useful for:
- Verifying crawler functionality
- Testing changes
- Debugging issues
- Quick validation of the crawling process 