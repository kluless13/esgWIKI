# ESG Data Crawler

This project crawls and collects ESG (Environmental, Social, and Governance) reports from Australian Stock Exchange (ASX) listed companies.

## Overview

The crawler uses a two-phase approach:
1. Collection Phase: Identifies and collects links to ESG-related reports and documents
2. Processing Phase: Downloads and processes the collected documents

## Project Structure

```
crawler/
├── main.py                 # Main crawler script for all companies
├── tests/
│   └── test_main.py       # Test crawler for first 3 companies
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── companies-list.csv     # List of companies to crawl
└── utils/
    └── scraper_utils.py   # Utility functions for crawling
```

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

## Usage

### Collection Phase

1. For testing (first 3 companies):
```bash
python tests/test_main.py collect
```

2. For full crawl (all companies):
```bash
python main.py collect
```

### Output Files

The crawler generates several output files:

- `sustainability_data.json`: Detailed information about found reports and pages
- `sustainability_summary.json`: Summary statistics of collected reports
- `report_link.txt`: List of report URLs (filtered to last 3 years)

## How It Works

1. **Company List Processing**
   - Reads company information from `companies-list.csv`
   - Extracts company names and URLs

2. **Initial Crawl**
   - Visits each company's ListCorp page
   - Identifies the Corporate Governance section

3. **Deep Crawl**
   - From the Corporate Governance page, finds:
     - Sustainability sections
     - ESG sections
     - Environmental sections
     - Climate sections

4. **Document Collection**
   - Scans all identified sections for:
     - PDF reports
     - Excel data files
   - Collects metadata including:
     - Document type
     - Year
     - Source page
     - File type

5. **Data Organization**
   - Filters documents to last 3 years
   - Categorizes documents by type
   - Generates summary statistics

## Configuration

The crawler can be configured through `config.py` and environment variables:
- Browser settings
- Timeouts
- Cache settings
- Required fields for data extraction

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