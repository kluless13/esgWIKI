# ASX ESG Data Collection and Analysis System

This repository combines two powerful components for ESG (Environmental, Social, and Governance) data analysis:

1. **ASX Company Sustainability Crawler**: Automatically collects ESG reports and sustainability data from ASX-listed companies
2. **ESG RAG System**: Analyzes the collected reports using DeepSeek's R1 model for intelligent querying and analysis

## System Components

### 1. ASX Sustainability Crawler

Automatically crawls ASX-listed companies' websites to collect:

**Sustainability Sections**:
- Sustainability pages
- ESG sections
- Environmental pages
- Climate-related sections

**Report Types**:
- Sustainability reports
- ESG reports
- Climate reports
- Environmental reports
- Carbon disclosure reports
- Emissions reports

### 2. ESG RAG System

Processes and analyzes the collected reports using:
- DeepSeek's R1 model for reasoning
- Efficient document retrieval
- Context-aware question-answering

## Setup

1. Clone the repository:
```bash
git clone https://github.com/kluless13/esgWIKI.git
cd esgWIKI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Add required API keys:
# - GROQ_API_KEY for PDF extraction
# - HUGGINGFACE_API_TOKEN for RAG system
```

4. Prepare your `companies-list.csv` file with:
   - `Link`: ListCorp URL for each company
   - `Company Name`: Company name

## Usage

### 1. Data Collection

Collect sustainability reports:
```bash
python main.py collect  # Gathers report links
python main.py crawl    # Processes and extracts data from PDFs
```

### 2. Data Analysis

Process and analyze the reports:
```bash
# Ingest collected PDFs into the vector database
python ingest_pdfs.py

# Start the RAG analysis system
python r1_smolagent_rag.py
```

## Output Files

### Crawler Outputs
1. `sustainability_data.json` - Detailed company ESG data
2. `sustainability_summary.json` - High-level ESG reporting summary
3. `report_link.txt` - List of all sustainability report URLs

### RAG System Outputs
- Processed documents in `chroma_db`
- Interactive analysis through Gradio interface

## File Structure

```
├── Crawler Components
│   ├── main.py              # Main crawler script
│   ├── config.py            # Crawler configuration
│   └── utils/
│       └── scraper_utils.py # Browser configuration
│
├── RAG Components
│   ├── ingest_pdfs.py       # PDF ingestion script
│   ├── r1_smolagent_rag.py  # RAG system
│   └── streamlit.py         # Web interface
│
├── data/                    # Collected PDFs
├── chroma_db/              # Vector database
├── requirements.txt        # Dependencies
└── .env.example           # Environment variables template
```

## How It Works

1. **Data Collection Phase**:
   - Crawler reads company URLs from `companies-list.csv`
   - Finds and downloads sustainability reports
   - Catalogs metadata (report type, year, source)

2. **Analysis Phase**:
   - RAG system ingests collected PDFs
   - Creates embeddings using sentence-transformers
   - Enables intelligent querying of ESG data

## Technical Notes

- Crawler uses 120-second timeout for reliable data collection
- RAG system uses chunking (1000 chars, 200 char overlap)
- Automatic deduplication of reports
- Efficient vector storage and retrieval

## Requirements

- Python 3.8+
- GROQ API key (PDF extraction)
- HuggingFace API token (RAG system)
- Sufficient storage for PDFs and vector database
