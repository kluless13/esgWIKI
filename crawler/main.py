import asyncio
import sys
import os
import json
import re
import csv
from urllib.parse import urlparse, urljoin

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from dotenv import load_dotenv

from config import BASE_URL, CSS_SELECTOR, REQUIRED_KEYS
from utils.data_utils import save_venues_to_csv
from utils.scraper_utils import get_browser_config, get_llm_strategy
from utils.report_utils import extract_report_link


load_dotenv()


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


async def collect_report_links():
    """
    Reads company URLs from companies-list.csv, collects sustainability reports and related data.
    """
    report_data = []  # Store both report links and additional sustainability data
    browser_config = get_browser_config()
    
    if not os.path.exists("companies-list.csv"):
        print("Error: companies-list.csv not found!")
        return

    # First read all company URLs
    company_urls = []
    with open("companies-list.csv", "r", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'Link' in row and row['Link'].strip():
                company_urls.append((row['Link'].strip(), row.get('Company Name', '')))

    if not company_urls:
        print("No valid company URLs found in companies-list.csv")
        return

    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Process all company URLs
        for company_url, company_name in company_urls:
            print(f"\nProcessing company URL: {company_url}")
            company_data = {
                'company_url': company_url,
                'company_name': company_name,
                'sustainability_pages': [],
                'sustainability_reports': []
            }
            
            try:
                # Load the company page
                result = await crawler.arun(
                    url=company_url,
                    config=CrawlerRunConfig(
                        cache_mode=CacheMode.BYPASS,
                        wait_until='networkidle',
                        page_timeout=120000  # Increased timeout to 120 seconds
                    )
                )
                
                if not result.success:
                    print(f"Failed to load URL: {company_url}")
                    continue
                
                # Find Corporate Governance link in Company Resources section
                governance_pattern = r'<h2>Company Resources</h2>.*?<a[^>]*?href="([^"]*?)"[^>]*?>Corporate Governance</a>'
                governance_match = re.search(governance_pattern, result.cleaned_html, re.DOTALL)
                
                if not governance_match:
                    print("No Corporate Governance link found in Company Resources section")
                    continue
                    
                governance_url = governance_match.group(1)
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
                        continue
                    
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
                                            (r'href="([^"]*?emissions[^"]*?report[^"]*?\.pdf)"', 'emissions-report')
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
                    continue
                    
            except Exception as e:
                print(f"Error processing {company_url}: {str(e)}")
            
            # Add the company data if we found any sustainability content
            if company_data['sustainability_pages'] or company_data['sustainability_reports']:
                report_data.append(company_data)
            await asyncio.sleep(2)

    # Save detailed report data to JSON
    with open("sustainability_data.json", "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nCollected data for {len(report_data)} companies and saved to 'sustainability_data.json'")
    
    # Create a summary of sustainability reports
    sustainability_summary = []
    for company in report_data:
        if company['sustainability_reports']:
            summary = {
                'company_name': company['company_name'],
                'total_reports': len(company['sustainability_reports']),
                'report_types': {},
                'years_covered': set()
            }
            for report in company['sustainability_reports']:
                summary['report_types'][report['type']] = summary['report_types'].get(report['type'], 0) + 1
                if report['year']:
                    summary['years_covered'].add(report['year'])
            summary['years_covered'] = sorted(list(summary['years_covered']))
            sustainability_summary.append(summary)
    
    # Save sustainability summary
    with open("sustainability_summary.json", "w") as f:
        json.dump(sustainability_summary, f, indent=2)
    print(f"Saved sustainability summary for {len(sustainability_summary)} companies to 'sustainability_summary.json'")
    
    # Save report URLs to report_link.txt
    with open("report_link.txt", "w") as f:
        for company in report_data:
            for report in company['sustainability_reports']:
                f.write(f"{report['url']}\n")
    print(f"Saved {sum(len(c['sustainability_reports']) for c in report_data)} sustainability report URLs to 'report_link.txt'")


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
    if not os.path.exists("report_link.txt"):
        print("report_link.txt not found. Please run in 'collect' mode first.")
        return
    with open("report_link.txt", "r") as f:
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


async def main():
    """
    Entry point of the script.
    Modes:
      - 'collect': Collect report links from companies in list.txt and create report_link.txt.
      - 'crawl': Crawl sustainability reports from links in report_link.txt.
      Default: If report_link.txt exists, run 'crawl', else run 'collect'.
    """
    mode = None
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    
    if mode == "collect":
        await collect_report_links()
    elif mode == "crawl":
        await crawl_reports_from_links()
    else:
        if os.path.exists("report_link.txt"):
            await crawl_reports_from_links()
        else:
            await collect_report_links()


if __name__ == "__main__":
    asyncio.run(main())
