import asyncio
import sys
import os
import json
import re
import csv
from urllib.parse import urlparse, urljoin

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_URL, CSS_SELECTOR, REQUIRED_KEYS
from utils.scraper_utils import get_browser_config, get_llm_strategy, ESGExtractionStrategy


load_dotenv()

# Output directory configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output file paths
SUSTAINABILITY_DATA = os.path.join(OUTPUT_DIR, 'sustainability_data.json')
SUSTAINABILITY_SUMMARY = os.path.join(OUTPUT_DIR, 'sustainability_summary.json')
REPORT_LINKS = os.path.join(OUTPUT_DIR, 'report_link.txt')
DOWNLOADS_DIR = os.path.join(OUTPUT_DIR, 'downloads')
os.makedirs(DOWNLOADS_DIR, exist_ok=True)


def convert_company_name_to_url(company_name: str) -> str:
    """
    Converts a company name to a company URL for listcorp.com. 
    Expected URL structure: https://www.listcorp.com/asx/ask/<slug>/news
    """
    slug = company_name.lower().strip()
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = re.sub(r'\s+', '-', slug)
    slug = re.sub(r'-+', '-', slug)
    return f"https://www.listcorp.com/asx/ask/{slug}/news"


def get_valid_years():
    """Helper function to get valid years for filtering"""
    current_year = 2024  # You can also use: from datetime import datetime; current_year = datetime.now().year
    return {str(year) for year in range(current_year-2, current_year+1)}  # Last 3 years


def is_document_url(url: str, html_content: str = None) -> tuple[bool, str]:
    """
    Helper function to determine if a URL points to a document and what type.
    Returns (is_document, file_type)
    """
    url_lower = url.lower()
    
    # Check for common document paths and keywords
    doc_indicators = [
        '/media/',
        '/documents/',
        '/reports/',
        '/publications/',
        '/annual-reports/'
    ]
    
    # Check if URL contains document indicators
    has_doc_indicators = any(indicator in url_lower for indicator in doc_indicators)
    
    # Check for explicit file extensions
    if url_lower.endswith(('.pdf', '.xlsx', '.xls')):
        return True, 'pdf' if url_lower.endswith('.pdf') else 'excel'
    
    # Check for Excel file indicators without extension
    if has_doc_indicators and (
        'databook' in url_lower or
        ('esg' in url_lower and 'data' in url_lower) or
        ('standards' in url_lower and 'data' in url_lower) or
        'metrics' in url_lower
    ):
        return True, 'excel'
    
    # Check for PDF indicators without extension
    if has_doc_indicators and (
        'report' in url_lower or
        'statement' in url_lower or
        'disclosure' in url_lower or
        'methodology' in url_lower or
        'definitions' in url_lower
    ):
        return True, 'pdf'
    
    # If we have HTML content, check for pdf-viewer elements
    if html_content:
        pdf_viewer_pattern = r'<pdf-viewer[^>]*?(?:data-url|src)=["\'](.*?)["\']'
        pdf_matches = re.finditer(pdf_viewer_pattern, html_content, re.IGNORECASE)
        for pdf_match in pdf_matches:
            pdf_url = pdf_match.group(1)
            if pdf_url in url or url in pdf_url:  # Check if URLs are related
                return True, 'pdf'
    
    return False, ''


def is_url_from_company(url: str, company_base_url: str) -> bool:
    """
    Check if a URL belongs to the company's domain.
    Handles both exact domain matches and subdomains.
    """
    try:
        url_domain = urlparse(url).netloc.lower()
        company_domain = urlparse(company_base_url).netloc.lower()
        
        # Handle www prefix
        url_domain = url_domain.replace('www.', '')
        company_domain = company_domain.replace('www.', '')
        
        # Check if it's an exact match or a subdomain
        return url_domain == company_domain or url_domain.endswith('.' + company_domain)
    except:
        return False


async def collect_report_links():
    """
    Reads all companies from companies-list.csv, collects sustainability reports and related data.
    """
    report_data = []
    browser_config = get_browser_config()
    
    if not os.path.exists("companies-list.csv"):
        print("Error: companies-list.csv not found!")
        return

    # Read all company URLs
    company_urls = []
    with open("companies-list.csv", "r", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'Link' in row and row['Link'].strip():
                name = row.get('Company Name', '').strip()
                if not name:  # If Company Name is empty, try to extract from Link
                    parsed_url = urlparse(row['Link'])
                    path_parts = parsed_url.path.split('/')
                    if len(path_parts) >= 3:
                        name = path_parts[3].replace('-', ' ').title()
                company_urls.append((row['Link'].strip(), name))
                print(f"Found company in CSV: {name} - {row['Link'].strip()}")

    if not company_urls:
        print("No valid company URLs found in companies-list.csv")
        return

    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Process companies concurrently in batches
        batch_size = 5  # Process 5 companies at a time
        for i in range(0, len(company_urls), batch_size):
            batch = company_urls[i:i + batch_size]
            tasks = [process_company(crawler, url, name) for url, name in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in batch_results:
                if isinstance(result, dict) and (result.get('sustainability_pages') or result.get('sustainability_reports')):
                    report_data.append(result)
            await asyncio.sleep(1)  # Reduced delay between batches

    # Save results
    with open(SUSTAINABILITY_DATA, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nCollected data for {len(report_data)} companies and saved to '{SUSTAINABILITY_DATA}'")
    
    # Create a summary of sustainability reports
    sustainability_summary = []
    for company in report_data:
        if company['sustainability_reports']:
            summary = {
                'company_name': company['company_name'],
                'company_url': company['company_url'],
                'total_reports': len(company['sustainability_reports']),
                'report_types': {},
                'years_covered': set(),
                'file_types': {
                    'pdf': 0,
                    'excel': 0
                }
            }
            
            for report in company['sustainability_reports']:
                # Count report types
                report_type = report['type']
                summary['report_types'][report_type] = summary['report_types'].get(report_type, 0) + 1
                
                # Track years
                if report['year']:
                    summary['years_covered'].add(report['year'])
                
                # Count file types
                file_type = report.get('file_type', 'pdf')
                summary['file_types'][file_type] += 1
            
            # Convert year set to sorted list
            summary['years_covered'] = sorted(list(summary['years_covered']))
            sustainability_summary.append(summary)
    
    # Save sustainability summary
    with open(SUSTAINABILITY_SUMMARY, "w") as f:
        json.dump(sustainability_summary, f, indent=2)
    print(f"\nSaved sustainability summary for {len(sustainability_summary)} companies to '{SUSTAINABILITY_SUMMARY}'")
    
    # Save report URLs to report_link.txt (only recent reports)
    report_count = 0
    with open(REPORT_LINKS, "w") as f:
        for company in report_data:
            # Add company header
            f.write(f"\n=== {company['company_name']} ===\n")
            for report in company['sustainability_reports']:
                if report.get('year') in get_valid_years():
                    f.write(f"{report['type']} ({report['year']}): {report['url']}\n")
                    report_count += 1
    
    print(f"Saved {report_count} recent sustainability report URLs to '{REPORT_LINKS}'")


async def process_company(crawler, company_url, company_name):
    """
    Process a single company's data
    """
    print(f"\nProcessing company URL: {company_url}")
    print(f"Company name: {company_name}")
    company_data = {
        'company_url': company_url,
        'company_name': company_name,
        'sustainability_pages': [],
        'sustainability_reports': []
    }
    
    # Track processed URLs to avoid duplicates
    processed_urls = set()
    processed_pdfs = set()
    valid_years = get_valid_years()
    
    try:
        # Load the company page with timeout
        result = await crawler.arun(
            url=company_url,
            config=CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                wait_until='networkidle',
                page_timeout=30000  # Reduced timeout to 30 seconds
            )
        )
        
        if not result.success:
            print(f"Failed to load URL: {company_url}")
            return company_data
        
        # Debug print to see what HTML we're getting
        print(f"Page HTML length: {len(result.cleaned_html)}")
        
        # Debug: Print HTML around Corporate Governance text
        context_pattern = r'(?s).{0,500}Corporate Governance.{0,500}'
        context_match = re.search(context_pattern, result.cleaned_html)
        if context_match:
            print("\nHTML context around Corporate Governance:")
            print(context_match.group(0))
            print("\n")
        
        # Find the Corporate Governance link using ListCorp's exact HTML structure
        governance_pattern = r'<a[^>]*?href="([^"]*?)"[^>]*?>Corporate Governance</a>'
        governance_match = re.search(governance_pattern, result.cleaned_html, re.IGNORECASE | re.DOTALL)
        
        if not governance_match:
            print("No governance link found with exact ListCorp pattern")
            return company_data
            
        governance_url = governance_match.group(1)
        if not governance_url.startswith('http'):
            parsed_url = urlparse(company_url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            governance_url = urljoin(base_url, governance_url)
        
        print(f"Found corporate governance URL: {governance_url}")
        company_data['governance_url'] = governance_url
        
        # Extract base company URL
        parsed_url = urlparse(governance_url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        print(f"Company base URL: {base_url}")
        company_data['base_url'] = base_url
        
        # Load the governance page
        try:
            gov_result = await crawler.arun(
                url=governance_url,
                config=CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    wait_until='networkidle',
                    page_timeout=120000  # Increased timeout to 120 seconds
                )
            )
            
            if not gov_result.success:
                print(f"Failed to load governance URL: {governance_url}")
                return company_data
            
            # Core sustainability section patterns
            sustainability_section_patterns = [
                (r'href="([^"]*?/sustainability[^"]*?)"', 'sustainability'),
                (r'href="([^"]*?/esg[^"]*?)"', 'esg'),
                (r'href="([^"]*?/environment[^"]*?)"', 'environment'),
                (r'href="([^"]*?/climate[^"]*?)"', 'climate')
            ]
            
            for pattern, section_type in sustainability_section_patterns:
                matches = re.finditer(pattern, gov_result.cleaned_html, re.IGNORECASE)
                for match in matches:
                    section_url = match.group(1)
                    if not section_url.startswith('http'):
                        section_url = urljoin(base_url, section_url)
                    if section_url not in [p['url'] for p in company_data['sustainability_pages']]:
                        print(f"Found {section_type} section: {section_url}")
                        company_data['sustainability_pages'].append({
                            'url': section_url,
                            'type': section_type
                        })
                        
                        # Load each sustainability section
                        try:
                            sus_result = await crawler.arun(
                                url=section_url,
                                config=CrawlerRunConfig(
                                    cache_mode=CacheMode.BYPASS,
                                    wait_until='networkidle',
                                    page_timeout=120000  # Increased timeout to 120 seconds
                                )
                            )
                            
                            if sus_result.success:
                                # Core sustainability report patterns
                                sustainability_pdf_patterns = [
                                    (r'href="([^"]*?sustainability[^"]*?report[^"]*?\.pdf)"', 'sustainability-report'),
                                    (r'href="([^"]*?esg[^"]*?report[^"]*?\.pdf)"', 'esg-report'),
                                    (r'href="([^"]*?climate[^"]*?report[^"]*?\.pdf)"', 'climate-report'),
                                    (r'href="([^"]*?environmental[^"]*?report[^"]*?\.pdf)"', 'environmental-report'),
                                    (r'href="([^"]*?carbon[^"]*?disclosure[^"]*?\.pdf)"', 'carbon-disclosure'),
                                    (r'href="([^"]*?emissions[^"]*?report[^"]*?\.pdf)"', 'emissions-report'),
                                    
                                    # Additional patterns
                                    (r'href="([^"]*?annual[^"]*?report[^"]*?\.pdf)"', 'annual-report'),
                                    (r'href="([^"]*?integrated[^"]*?report[^"]*?\.pdf)"', 'integrated-report'),
                                    (r'href="([^"]*?tcfd[^"]*?report[^"]*?\.pdf)"', 'tcfd-report'),
                                    (r'href="([^"]*?social[^"]*?report[^"]*?\.pdf)"', 'social-report'),
                                    (r'href="([^"]*?governance[^"]*?report[^"]*?\.pdf)"', 'governance-report'),
                                    (r'href="([^"]*?responsibility[^"]*?report[^"]*?\.pdf)"', 'responsibility-report')
                                ]
                                
                                for pattern, report_type in sustainability_pdf_patterns:
                                    matches = re.finditer(pattern, sus_result.cleaned_html, re.IGNORECASE)
                                    for match in matches:
                                        report_url = match.group(1)
                                        if not report_url.startswith('http'):
                                            report_url = urljoin(section_url, report_url)
                                        
                                        # Extract year if present in the URL or filename
                                        year_match = re.search(r'20\d{2}', report_url)
                                        year = year_match.group(0) if year_match else None
                                        
                                        report_data = {
                                            'url': report_url,
                                            'type': report_type,
                                            'year': year,
                                            'source_page': section_url,
                                            'source_section': section_type
                                        }
                                        
                                        # Check if we already have this report
                                        if not any(r['url'] == report_url for r in company_data['sustainability_reports']):
                                            print(f"Found {report_type} ({year if year else 'year unknown'}): {report_url}")
                                            company_data['sustainability_reports'].append(report_data)
                                            
                        except Exception as e:
                            print(f"Error accessing sustainability section {section_url}: {str(e)}")
                            continue
                    
        except Exception as e:
            print(f"Error accessing governance page: {str(e)}")
            return company_data
            
    except Exception as e:
        print(f"Error processing {company_url}: {str(e)}")
    
    return company_data


async def process_sustainability_section(crawler, section_url, company_data, processed_urls, processed_pdfs):
    """Helper function to process a sustainability section"""
    if section_url in processed_urls:
        return
    
    processed_urls.add(section_url)
    valid_years = get_valid_years()
    
    try:
        sus_result = await crawler.arun(
            url=section_url,
            config=CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                wait_until='networkidle',
                page_timeout=180000  # Increased timeout
            )
        )
        
        if sus_result.success:
            section_type = 'sustainability'
            if 'esg' in section_url.lower():
                section_type = 'esg'
            elif 'environment' in section_url.lower():
                section_type = 'environment'
            elif 'climate' in section_url.lower():
                section_type = 'climate'
            
            # Process the main section page for PDFs
            await process_page_for_pdfs(sus_result.cleaned_html, section_url, section_type, company_data, processed_pdfs)
            
            # Enhanced patterns for finding subsections
            subsection_patterns = [
                # Common report repository sections
                r'href="([^"]*?/reports?[^"]*?)"',
                r'href="([^"]*?/publications?[^"]*?)"',
                r'href="([^"]*?/documents?[^"]*?)"',
                r'href="([^"]*?/downloads?[^"]*?)"',
                r'href="([^"]*?/resources?[^"]*?)"',
                # ESG and sustainability specific sections
                r'href="([^"]*?/sustainability[^"]*?)"',
                r'href="([^"]*?/esg[^"]*?)"',
                r'href="([^"]*?/environment[^"]*?)"',
                r'href="([^"]*?/climate[^"]*?)"',
                # Annual report sections
                r'href="([^"]*?/annual[^"]*?report[^"]*?)"',
                r'href="([^"]*?/financial[^"]*?report[^"]*?)"',
                # Additional common patterns
                r'href="([^"]*?/performance[^"]*?)"',
                r'href="([^"]*?/governance[^"]*?)"',
                r'href="([^"]*?/policies[^"]*?)"'
            ]
            
            for subsection_pattern in subsection_patterns:
                subsection_matches = re.finditer(subsection_pattern, sus_result.cleaned_html, re.IGNORECASE)
                for subsection_match in subsection_matches:
                    subsection_url = subsection_match.group(1)
                    if not subsection_url.startswith('http'):
                        subsection_url = urljoin(section_url, subsection_url)
                    
                    if subsection_url not in processed_urls and subsection_url not in processed_pdfs:
                        # Check if URL belongs to company
                        if not is_url_from_company(subsection_url, company_data['base_url']):
                            continue
                            
                        # Check if it's a document first
                        is_doc, file_type = is_document_url(subsection_url)
                        if is_doc:
                            year_match = re.search(r'20\d{2}', subsection_url)
                            year = year_match.group(0) if year_match else None
                            
                            if year and year in valid_years:
                                processed_pdfs.add(subsection_url)
                                report_type = determine_report_type(subsection_url, file_type)
                                
                                report_data = {
                                    'url': subsection_url,
                                    'type': report_type,
                                    'year': year,
                                    'source_page': section_url,
                                    'source_section': section_type,
                                    'file_type': file_type
                                }
                                
                                if not any(r['url'] == subsection_url for r in company_data['sustainability_reports']):
                                    print(f"Found {report_type} ({year}) [{file_type}]: {subsection_url}")
                                    company_data['sustainability_reports'].append(report_data)
                        else:
                            processed_urls.add(subsection_url)
                            print(f"Found subsection URL: {subsection_url}")
                            await process_subsection(crawler, subsection_url, section_type, company_data, processed_pdfs)
                        
    except Exception as e:
        print(f"Error accessing sustainability section {section_url}: {str(e)}")


def determine_report_type(url, file_type):
    """Helper function to determine report type based on URL and file type"""
    url_lower = url.lower()
    
    # ESG Reports
    if 'esg' in url_lower:
        return 'esg-data' if file_type == 'excel' else 'esg-report'
    
    # Sustainability Reports
    if 'sustainability' in url_lower:
        return 'sustainability-data' if file_type == 'excel' else 'sustainability-report'
    
    # Climate Reports
    if 'climate' in url_lower or 'tcfd' in url_lower:
        return 'climate-report'
    
    # Environmental Reports
    if 'environment' in url_lower:
        return 'environmental-report'
    
    # Annual Reports
    if 'annual' in url_lower:
        return 'annual-report'
    
    # Performance Data
    if 'performance' in url_lower and file_type == 'excel':
        return 'performance-data'
    
    # Metrics Data
    if 'metrics' in url_lower and file_type == 'excel':
        return 'metrics-data'
    
    # Default types based on file type
    return 'excel-data' if file_type == 'excel' else 'sustainability-report'


async def process_subsection(crawler, subsection_url, section_type, company_data, processed_pdfs):
    """Helper function to process a subsection"""
    try:
        # Check if this is a document using our helper function
        is_doc, file_type = is_document_url(subsection_url)
        if is_doc:
            # Extract year if present in the URL or filename
            year_match = re.search(r'20\d{2}', subsection_url)
            year = year_match.group(0) if year_match else None
            
            # Skip if report is older than 3 years
            valid_years = get_valid_years()
            if not year or year not in valid_years:
                return
            
            # Determine report type based on URL and file type
            if file_type == 'excel':
                if 'esg' in subsection_url.lower():
                    report_type = 'esg-data'
                elif 'sustainability' in subsection_url.lower():
                    report_type = 'sustainability-data'
                elif 'performance' in subsection_url.lower():
                    report_type = 'performance-data'
                elif 'metrics' in subsection_url.lower():
                    report_type = 'metrics-data'
                else:
                    report_type = 'excel-data'
            else:
                report_type = 'sustainability-report' if 'sustainability' in subsection_url.lower() else 'annual-report'
            
            report_data = {
                'url': subsection_url,
                'type': report_type,
                'year': year,
                'source_page': subsection_url,
                'source_section': section_type,
                'file_type': file_type
            }
            
            if not any(r['url'] == subsection_url for r in company_data['sustainability_reports']):
                print(f"Found {report_type} ({year if year else 'year unknown'}) [{file_type}]: {subsection_url}")
                company_data['sustainability_reports'].append(report_data)
            return
        
        subsection_result = await crawler.arun(
            url=subsection_url,
            config=CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                wait_until='networkidle',  # Changed to ensure dynamic content loads
                page_timeout=180000  # Increased timeout
            )
        )
        
        if subsection_result.success:
            await process_page_for_pdfs(subsection_result.cleaned_html, subsection_url, section_type, company_data, processed_pdfs)
            
    except Exception as e:
        print(f"Error accessing subsection {subsection_url}: {str(e)}")


async def process_page_for_pdfs(html_content, base_url, section_type, company_data, processed_pdfs):
    """Helper function to process a page for PDF links and Excel files"""
    valid_years = get_valid_years()
    
    # Combined pattern for both PDF and Excel files
    file_patterns = [
        # PDF Report Patterns
        (r'href="([^"]*?sustainability[^"]*?report[^"]*?\.pdf)"', 'sustainability-report'),
        (r'href="([^"]*?esg[^"]*?report[^"]*?\.pdf)"', 'esg-report'),
        (r'href="([^"]*?climate[^"]*?report[^"]*?\.pdf)"', 'climate-report'),
        (r'href="([^"]*?environmental[^"]*?report[^"]*?\.pdf)"', 'environmental-report'),
        (r'href="([^"]*?carbon[^"]*?disclosure[^"]*?\.pdf)"', 'carbon-disclosure'),
        (r'href="([^"]*?emissions[^"]*?report[^"]*?\.pdf)"', 'emissions-report'),
        (r'href="([^"]*?annual[^"]*?report[^"]*?\.pdf)"', 'annual-report'),
        (r'href="([^"]*?integrated[^"]*?report[^"]*?\.pdf)"', 'integrated-report'),
        (r'href="([^"]*?tcfd[^"]*?report[^"]*?\.pdf)"', 'tcfd-report'),
        (r'href="([^"]*?social[^"]*?report[^"]*?\.pdf)"', 'social-report'),
        (r'href="([^"]*?governance[^"]*?report[^"]*?\.pdf)"', 'governance-report'),
        (r'href="([^"]*?responsibility[^"]*?report[^"]*?\.pdf)"', 'responsibility-report'),
        # Generic PDF patterns
        (r'href="([^"]*?report[^"]*?\.pdf)"', 'sustainability-report'),
        (r'href="([^"]*?\.pdf)"', 'pdf-document'),
        
        # Excel Data Patterns
        (r'href="([^"]*?esg[^"]*?data[^"]*?\.xlsx?)"', 'esg-data'),
        (r'href="([^"]*?sustainability[^"]*?data[^"]*?\.xlsx?)"', 'sustainability-data'),
        (r'href="([^"]*?performance[^"]*?data[^"]*?\.xlsx?)"', 'performance-data'),
        (r'href="([^"]*?metrics[^"]*?data[^"]*?\.xlsx?)"', 'metrics-data'),
        # Generic Excel patterns
        (r'href="([^"]*?data[^"]*?\.xlsx?)"', 'excel-data'),
        (r'href="([^"]*?\.xlsx?)"', 'excel-document'),
        
        # Additional formats
        (r'href="([^"]*?\.csv)"', 'csv-data'),
        (r'href="([^"]*?\.json)"', 'json-data')
    ]
    
    for pattern, report_type in file_patterns:
        matches = re.finditer(pattern, html_content, re.IGNORECASE)
        for match in matches:
            file_url = match.group(1)
            if not file_url.startswith('http'):
                file_url = urljoin(base_url, file_url)
            
            # Skip if we've already processed this file or if it's not from company domain
            if file_url in processed_pdfs or not is_url_from_company(file_url, company_data['base_url']):
                continue
            
            processed_pdfs.add(file_url)
            
            # Try different year formats
            year = None
            
            # Try fiscal year format (FY22, FY2022, etc)
            fy_match = re.search(r'fy(?:20)?(\d{2})', file_url.lower())
            if fy_match:
                year = f"20{fy_match.group(1)}"
            else:
                # Try direct year format (2022, 2023, etc)
                year_match = re.search(r'20\d{2}', file_url)
                if year_match:
                    year = year_match.group(0)
            
            # Skip if report is older than 3 years
            if not year or year not in valid_years:
                continue
            
            # Determine file type
            file_type = 'excel' if file_url.lower().endswith(('.xlsx', '.xls')) else 'pdf'
            if file_url.lower().endswith('.csv'):
                file_type = 'csv'
            elif file_url.lower().endswith('.json'):
                file_type = 'json'
            
            # Refine report type based on URL content if it's a generic type
            if report_type in ['pdf-document', 'excel-document']:
                report_type = determine_report_type(file_url, file_type)
            
            report_data = {
                'url': file_url,
                'type': report_type,
                'year': year,
                'source_page': base_url,
                'source_section': section_type,
                'file_type': file_type
            }
            
            # Check if we already have this report
            if not any(r['url'] == file_url for r in company_data['sustainability_reports']):
                print(f"Found {report_type} ({year}) [{file_type}]: {file_url}")
                company_data['sustainability_reports'].append(report_data)


async def crawl_report_link(crawler, report_url, llm_strategy, session_id, required_keys, seen_companies):
    """
    Crawls a single report link and extracts report data.
    """
    print(f"Processing report link: {report_url}")
    result = await crawler.arun(
        url=report_url,
        config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            extraction_strategy=llm_strategy,
            css_selector=CSS_SELECTOR,
            session_id=session_id,
        ),
    )
    if not (result.success and result.extracted_content):
        print(f"Error fetching report at {report_url}: {result.error_message}")
        return []
    try:
        extracted_data = json.loads(result.extracted_content)
    except Exception as e:
        print(f"Failed to parse JSON in {report_url}: {str(e)}")
        return []
    if not extracted_data:
        print(f"No data found in report {report_url}")
        return []
    complete_reports = []
    for report in extracted_data:
        # Check completeness based on required_keys
        if not all(key in report and report[key] for key in required_keys):
            continue
        if report["company_name"] in seen_companies:
            print(f"Duplicate report for company {report['company_name']} found. Skipping.")
            continue
        seen_companies.add(report["company_name"])
        complete_reports.append(report)
    if complete_reports:
        print(f"Extracted {len(complete_reports)} complete report(s) from {report_url}.")
    else:
        print(f"No complete report extracted from {report_url}.")
    return complete_reports


async def crawl_reports_from_links():
    """
    Crawls sustainability report pages from links listed in report_link.txt.
    """
    session_id = "report_crawl_links"
    browser_config = get_browser_config()
    llm_strategy = get_llm_strategy()
    seen_companies = set()
    all_reports = []
    if not os.path.exists(REPORT_LINKS):
        print("report_link.txt not found. Please run in 'collect' mode first.")
        return
    with open(REPORT_LINKS, "r") as f:
        links = [line.strip() for line in f if line.strip()]
    async with AsyncWebCrawler(config=browser_config) as crawler:
        for report_url in links:
            reports = await crawl_report_link(crawler, report_url, llm_strategy, session_id, REQUIRED_KEYS, seen_companies)
            all_reports.extend(reports)
            await asyncio.sleep(2)
    if all_reports:
        save_venues_to_csv(all_reports, "sustainability_reports.csv")
        print(f"Saved {len(all_reports)} reports to 'sustainability_reports.csv'.")
    else:
        print("No reports were extracted from the report links.")


async def analyze_sustainability_reports():
    """
    Analyzes sustainability reports from report_link.txt and creates a metrics CSV.
    """
    if not os.path.exists(REPORT_LINKS):
        print("Error: report_link.txt not found! Please run 'collect' mode first.")
        return
        
    metrics_data = []
    browser_config = get_browser_config()
    llm_strategy = get_llm_strategy()
    
    with open(REPORT_LINKS, "r") as f:
        report_links = [line.strip() for line in f if line.strip()]
    
    print(f"\nAnalyzing {len(report_links)} sustainability reports...")
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        for report_url in report_links:
            print(f"\nAnalyzing report: {report_url}")
            try:
                # Handle Excel files differently
                if report_url.lower().endswith(('.xlsx', '.xls')):
                    print("Excel file detected - downloading for separate processing")
                    # TODO: Add Excel file processing
                    continue
                    
                result = await crawler.arun(
                    url=report_url,
                    config=CrawlerRunConfig(
                        cache_mode=CacheMode.BYPASS,
                        extraction_strategy=llm_strategy,
                        wait_until='networkidle',
                        page_timeout=300000  # 5 minutes timeout for PDFs
                    )
                )
                
                if result.success and result.extracted_content:
                    metrics = json.loads(result.extracted_content)
                    metrics_data.append(metrics)
                    print(f"Successfully extracted metrics from report")
                else:
                    print(f"Failed to extract metrics: {result.error_message}")
            except Exception as e:
                print(f"Error processing report {report_url}: {str(e)}")
    
    if not metrics_data:
        print("\nNo metrics were successfully extracted from the reports.")
        return
        
    # Create CSV with metrics
    csv_fields = [
        'company_name', 'year', 'scope1_emissions', 'scope2_emissions', 'scope3_emissions',
        'renewable_energy_percentage', 'renewable_energy_target', 'target_year',
        'emission_reduction_target', 'current_reduction_percentage', 'net_zero_commitment_year',
        'carbon_price_used', 'energy_efficiency_initiatives', 'renewable_projects'
    ]
    
    with open('esg_metrics.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for metrics in metrics_data:
            writer.writerow(metrics)
    
    print(f"\nSuccessfully analyzed {len(metrics_data)} reports and saved metrics to 'esg_metrics.csv'")


async def main():
    """
    Main entry point for the crawler
    """
    if len(sys.argv) < 2:
        print("Please specify a command: collect")
        sys.exit(1)

    command = sys.argv[1]
    if command == "collect":
        await collect_report_links()
    else:
        print("Invalid command. Available commands: collect")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
