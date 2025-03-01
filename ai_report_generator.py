import json
import os
from datetime import datetime
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
import time
from functools import wraps
import signal

# Load environment variables
load_dotenv()

# Set OpenAI API key
OPENAI_API_KEY = "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"
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def with_timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set the signal handler
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                # Disable the alarm
                signal.alarm(0)
            return result
        return wrapper
    return decorator

class ESGReportGenerator:
    def __init__(self):
        print("Initializing ESG Report Generator...")
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.2,
            request_timeout=60
        )

    def format_raw_data(self, data: Dict[str, Any]) -> str:
        """Format raw data into a clear markdown section."""
        sections = data['sections']
        emissions = sections['current_environmental_impact']['emissions']
        
        output = []
        output.append("# Raw ESG Data Analysis\n")
        
        # Emissions Data
        output.append("## Current Emissions Data\n")
        for scope in ['scope1', 'scope2', 'scope3']:
            if scope in emissions:
                current = emissions[scope]['current']
                output.append(f"### {scope.upper()}")
                output.append(f"- Current Value: {current['value']} {current['unit']}")
                output.append(f"- Year: {current['year']}")
                if 'historical' in emissions[scope]:
                    output.append("\nHistorical Data:")
                    for year, data in emissions[scope]['historical'].items():
                        output.append(f"- {year}: {data['value']} {data['unit']}")
                output.append("")
        
        # Climate Commitments
        output.append("## Climate Commitments\n")
        climate = sections['climate_commitments']
        if 'net_zero' in climate:
            output.append("### Net Zero Target")
            output.append(f"- Commitment: {climate['net_zero']['commitment']}")
            output.append(f"- Scope: {climate['net_zero']['scope']}\n")
        
        # Financial Commitments
        output.append("## Financial Commitments\n")
        finance = sections['financial_commitment']
        if 'sustainable_finance' in finance:
            output.append("### Sustainable Finance")
            output.append(f"- Target: {finance['sustainable_finance']['target']}")
            output.append(f"- Current: {finance['sustainable_finance']['current']}")
            output.append(f"- Progress: {finance['sustainable_finance']['progress']}\n")
        
        return "\n".join(output)

    def generate_analysis(self, data: Dict[str, Any]) -> str:
        """Generate concise analysis of strengths and shortcomings."""
        prompt = ChatPromptTemplate.from_template("""
        Analyze the following ESG data and provide:
        1. Key strengths (3-4 bullet points)
        2. Major shortcomings (3-4 bullet points)
        3. One-line summary for each major metric

        Data:
        {data}

        Keep the analysis concise and factual.
        Format in markdown.
        """)

        response = self.llm.invoke(prompt.format(
            data=json.dumps(data['sections'], indent=2)
        ))
        return response.content

    def identify_commercial_opportunities(self, data: Dict[str, Any]) -> str:
        """Identify specific commercial opportunities based on the analysis."""
        prompt = ChatPromptTemplate.from_template("""
        Based on the ESG data provided, identify 3-5 specific commercial opportunities:

        Data:
        {data}

        For each opportunity:
        1. Brief description (one line)
        2. Why it's valuable (one line)
        3. Estimated timeline to implement

        Format in markdown, keep it concise and actionable.
        """)

        response = self.llm.invoke(prompt.format(
            data=json.dumps(data['sections'], indent=2)
        ))
        return response.content

    def generate_report(self, analysis_path: str) -> str:
        """Generate the complete report."""
        print("Generating ESG report...")
        
        # Load and parse the JSON data
        with open(analysis_path, 'r') as f:
            data = json.load(f)
        
        # Generate each section
        raw_data = self.format_raw_data(data)
        analysis = self.generate_analysis(data)
        opportunities = self.identify_commercial_opportunities(data)
        
        # Combine sections
        report = f"""
        # ESG Analysis Report
        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        {raw_data}

        # Analysis Summary
        {analysis}

        # Commercial Opportunities
        {opportunities}
        """
        
        return report

def main():
    generator = ESGReportGenerator()
    report = generator.generate_report('output/esg_analysis_20250301_135322.json')
    
    # Save the report
    output_path = f"output/esg_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"Report generated successfully: {output_path}")

if __name__ == "__main__":
    main() 