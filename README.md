## Running the Pipeline

### 1. Process Excel Data
```bash
# Put your Excel file in input/excel/
# Example: input/excel/2024-sustainability-data-pack.xlsx
python excel.py --energy --scope1 --scope2 --scope3 --renewable --emissions --debug

# This will create: output/extracted/excel_data.json
```

### 2. Process PDF Reports
```bash
# Put your PDF files in input/pdf/
# Example: input/pdf/nab-climate-report.pdf

# Extract all metrics (recommended)
python pdf_extract.py --debug

# Or extract specific metrics:
python pdf_extract.py --metric net_zero
python pdf_extract.py --metric emission_targets
python pdf_extract.py --metric renewable_targets
python pdf_extract.py --metric carbon_neutral
python pdf_extract.py --metric carbon_price
python pdf_extract.py --metric sustainable_finance
python pdf_extract.py --metric climate_investment

# Specify a different PDF file:
python pdf_extract.py --file input/pdf/custom_report.pdf

# This will create: output/extracted/pdf_data.json
```

Available metrics:
- `net_zero`: Net zero commitment year and interim targets
- `emission_targets`: Emission reduction targets and base years
- `renewable_targets`: Renewable energy targets and current percentage
- `carbon_neutral`: Carbon neutral certification status
- `carbon_price`: Internal carbon price used for decision making
- `sustainable_finance`: Sustainable finance commitments and progress
- `climate_investment`: Climate-related investments and initiatives

### 3. Analyze ESG Data
```bash
# Uses the extracted data from steps 1 and 2
python esg_analyzer.py --excel output/extracted/excel_data.json --pdf output/extracted/pdf_data.json --debug

# This will create: output/analyzed/esg_analysis_[timestamp].json
```

### 4. Generate Report
```bash
# Uses the analyzed data from step 3
python ai_report_generator.py --input output/analyzed/esg_analysis_[timestamp].json

# This will create: output/reports/esg_report_[timestamp].md
```

## File Structure
```
input/
  ├── excel/
  │   └── 2024-sustainability-data-pack.xlsx  # Put your Excel files here
  └── pdf/
      └── nab-climate-report.pdf  # Put your PDF files here

output/
  ├── extracted/
  │   ├── excel_data.json          # Created by excel.py
  │   └── pdf_data.json           # Created by pdf_extract.py
  ├── analyzed/
  │   └── esg_analysis_*.json    # Created by esg_analyzer.py
  └── reports/
      └── esg_report_*.md       # Created by ai_report_generator.py
```

TODO:
- Generalise scripts for different companies