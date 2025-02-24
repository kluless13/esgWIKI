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
├── main.py                 # Main crawler script for all companies
├── tests/
│   └── test_main.py       # Test crawler for first 3 companies
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── companies-list.csv     # List of companies to crawl
└── utils/
    └── scraper_utils.py   # Utility functions for crawling
```

## Step-by-Step Guide to Using the ESG Crawler

### 1. Initial Setup

First, ensure you have all dependencies installed:
```bash
cd /Users/kluless/esgWIKI/crawler
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file in the crawler directory with necessary configurations:
```bash
cp .env.example .env
```

Edit the `.env` file with your API keys:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Data Collection

#### 3.1 Test Run (First 3 Companies)
```bash
python tests/test_main.py collect
```

This will:
- Read the first 3 companies from `companies-list.csv`
- Crawl their websites
- Generate:
  - `sustainability_data.json`
  - `sustainability_summary.json`
  - `report_link.txt`

#### 3.2 Full Run (All Companies)
```bash
python main.py collect
```

This will do the same as the test run but for all companies.

### 4. Report Download

To download reports:
```bash
python download_reports.py
```

This will:
- Read URLs from `report_link.txt`
- Download files to the `downloads` directory
- Handle both PDF and Excel files
- Organize downloads by company

### 5. Processing Excel Files

To process a specific Excel file:
```bash
python process_excel_data.py <excel_file> <company_name> <company_code>
```

Example:
```bash
python process_excel_data.py "downloads/2024-sustainability-data-pack.xlsx" "Commonwealth Bank of Australia" "CBA"
```

This will:
1. Load the Excel file
2. Process each relevant sheet (GHG Emissions, Energy, Position)
3. Extract metrics such as:
   - Scope 1, 2, and 3 emissions
   - Renewable energy percentage
   - Energy consumption data
   - Net zero commitments
   - Emission reduction targets
4. Print the extracted metrics

The script looks for specific sheets and patterns:
- "GHG Emissions" - for emissions data
- "Energy" - for energy consumption and renewable energy data
- "Position" - for strategic targets and commitments

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