import os
import sys
import re
import csv
import json
import asyncio
import aiohttp
from urllib.parse import urljoin, urlparse
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Set
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_URL, CSS_SELECTOR, REQUIRED_KEYS
from utils.scraper_utils import get_browser_config, get_llm_strategy, ESGExtractionStrategy


load_dotenv()

# Test output directory
TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'test_output')
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

# Test output files
TEST_SUSTAINABILITY_DATA = os.path.join(TEST_OUTPUT_DIR, 'sustainability_data.json')
TEST_SUSTAINABILITY_SUMMARY = os.path.join(TEST_OUTPUT_DIR, 'sustainability_summary.json')
TEST_REPORT_LINKS = os.path.join(TEST_OUTPUT_DIR, 'report_link.txt')
TEST_METRICS_CSV = os.path.join(TEST_OUTPUT_DIR, 'metrics.csv')


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


def get_valid_years() -> Set[str]:
    """Return a set of valid years (current year and 2 years prior)"""
    current_year = datetime.now().year
    return {str(year) for year in range(current_year - 2, current_year + 1)}


def is_document_url(url: str) -> Tuple[bool, str]:
    """Check if a URL points to a document and return its type"""
    url_lower = url.lower()
    if url_lower.endswith('.pdf'):
        return True, 'pdf'
    elif url_lower.endswith(('.xlsx', '.xls')):
        return True, 'excel'
    elif url_lower.endswith('.csv'):
        return True, 'csv'
    elif url_lower.endswith('.json'):
        return True, 'json'
    return False, None


def is_url_from_company(url: str, base_url: str) -> bool:
    """Check if a URL belongs to the company's domain"""
    try:
        url_domain = urlparse(url).netloc
        base_domain = urlparse(base_url).netloc
        return url_domain == base_domain or url_domain.endswith('.' + base_domain)
    except:
        return False


def clean_url(url: str) -> str:
    """Clean and normalize a URL"""
    # Remove any fragments
    url = url.split('#')[0]
    # Remove any query parameters
    url = url.split('?')[0]
    # Remove trailing slashes
    url = url.rstrip('/')
    return url


async def collect_report_links():
    """
    Main function to collect sustainability report links from company websites.
    Only processes the first 3 companies for testing purposes.
    """
    print("Starting test collection of sustainability report links...")
    
    # Create test output directory if it doesn't exist
    os.makedirs(os.path.dirname(TEST_SUSTAINABILITY_DATA), exist_ok=True)
    
    # Get browser configuration
    browser_config = get_browser_config()
    
    # Initialize the crawler
    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Read company URLs from CSV
        companies = []
        with open('companies-list.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'Link' in row and row['Link'].strip():
                    name = row.get('Company Name', '').strip()
                    if not name:  # If Company Name is empty, try to extract from Link
                        parsed_url = urlparse(row['Link'])
                        path_parts = parsed_url.path.split('/')
                        if len(path_parts) >= 3:
                            name = path_parts[3].replace('-', ' ').title()
                    companies.append({
                        'url': row['Link'].strip(),
                        'name': name
                    })
                    print(f"Found company in CSV: {name} - {row['Link'].strip()}")
        
        if not companies:
            print("No valid company URLs found in companies-list.csv")
            return
        
        # Process only the first 3 companies for testing
        test_companies = companies[:3]
        print(f"Processing {len(test_companies)} companies for testing...")
        
        # Process companies in batches
        batch_size = 1  # Process one at a time for testing
        sustainability_data = []
        
        for i in range(0, len(test_companies), batch_size):
            batch = test_companies[i:i + batch_size]
            tasks = []
            for company in batch:
                task = asyncio.create_task(
                    process_company(crawler, company['url'], company['name'])
                )
                tasks.append(task)
            
            # Wait for all tasks in the batch to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle any exceptions
            for company, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    print(f"Error processing {company['name']}: {str(result)}")
                    continue
                
                if result['sustainability_reports']:
                    print(f"\nFound {len(result['sustainability_reports'])} reports for {company['name']}")
                    sustainability_data.append(result)
                else:
                    print(f"\nNo sustainability reports found for {company['name']}")
        
        # Save sustainability data to JSON
        with open(TEST_SUSTAINABILITY_DATA, 'w') as f:
            json.dump(sustainability_data, f, indent=2)
        
        # Create summary of findings
        summary = {
            'total_companies': len(test_companies),
            'companies_with_reports': len(sustainability_data),
            'total_reports': sum(len(company['sustainability_reports']) for company in sustainability_data),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save summary to JSON
        with open(TEST_SUSTAINABILITY_SUMMARY, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save report links to text file
        with open(TEST_REPORT_LINKS, 'w') as f:
            for company in sustainability_data:
                f.write(f"\n=== {company['company_name']} ===\n")
                for report in company['sustainability_reports']:
                    f.write(f"{report['type']} ({report['year']}): {report['url']}\n")
        
        print("\nTest collection completed!")
        print(f"Processed {len(test_companies)} companies")
        print(f"Found reports for {len(sustainability_data)} companies")
        print(f"Total reports found: {summary['total_reports']}")
        print(f"\nResults saved to:")
        print(f"- {TEST_SUSTAINABILITY_DATA}")
        print(f"- {TEST_SUSTAINABILITY_SUMMARY}")
        print(f"- {TEST_REPORT_LINKS}")


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
            
            # Common report section patterns
            report_section_patterns = [
                # Investor/Annual Report sections
                (r'href="([^"]*?/investors?[^"]*?/(?:annual-)?reports?[^"]*?)"', 'investor-reports'),
                (r'href="([^"]*?/annual-reports?[^"]*?)"', 'annual-reports'),
                # Sustainability sections
                (r'href="([^"]*?/sustainability[^"]*?)"', 'sustainability'),
                (r'href="([^"]*?/esg[^"]*?)"', 'esg'),
                (r'href="([^"]*?/environment[^"]*?)"', 'environment'),
                (r'href="([^"]*?/climate[^"]*?)"', 'climate'),
                # Document sections
                (r'href="([^"]*?/documents?[^"]*?)"', 'documents'),
                (r'href="([^"]*?/publications?[^"]*?)"', 'publications')
            ]
            
            # Process governance page and find report sections
            for pattern, section_type in report_section_patterns:
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
                        
                        # Process each section
                        await process_sustainability_section(crawler, section_url, company_data, processed_urls, processed_pdfs)
                    
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
        subsection_result = await crawler.arun(
            url=subsection_url,
            config=CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                wait_until='networkidle',
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
    
    if not os.path.exists(TEST_REPORT_LINKS):
        print("report_link.txt not found. Please run in 'collect' mode first.")
        return
        
    with open(TEST_REPORT_LINKS, "r") as f:
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
    if not os.path.exists(TEST_REPORT_LINKS):
        print("Error: report_link.txt not found! Please run 'collect' mode first.")
        return
        
    metrics_data = []
    browser_config = get_browser_config()
    llm_strategy = get_llm_strategy()
    
    with open(TEST_REPORT_LINKS, "r") as f:
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
    
    with open(TEST_METRICS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for metrics in metrics_data:
            writer.writerow(metrics)
    
    print(f"\nSuccessfully analyzed {len(metrics_data)} reports and saved metrics to '{TEST_METRICS_CSV}'")


async def main():
    """
    Test entry point that processes only the first 3 companies
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
