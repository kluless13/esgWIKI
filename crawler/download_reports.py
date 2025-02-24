import os
import json
import time
import re
import sys
from typing import Dict, List, Set, Tuple, Optional
from urllib.parse import urlparse, urljoin, unquote
from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from utils.download_manager import DownloadManager

# Output directory configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
DOWNLOADS_DIR = os.path.join(OUTPUT_DIR, 'downloads')
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# Valid years for filtering
VALID_YEARS = {'2022', '2023', '2024'}

def extract_year_from_url(url: str) -> Optional[str]:
    """Extract year from URL if present."""
    year_match = re.search(r'(20\d{2})', url.lower())
    return year_match.group(1) if year_match else None

def is_valid_year(url: str) -> bool:
    """Check if URL contains a valid year (2022-2024)."""
    year = extract_year_from_url(url)
    return year in VALID_YEARS if year else False

def setup_chrome_driver(download_dir: str = DOWNLOADS_DIR, use_headless: bool = False) -> webdriver.Chrome:
    """
    Set up Chrome driver with appropriate options for downloading Excel files.
    Headless mode is optional and disabled by default.
    """
    chrome_options = webdriver.ChromeOptions()
    
    # Headless mode is optional now
    if use_headless:
        chrome_options.add_argument('--headless')
    
    # Basic Chrome settings
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--window-size=1920,1080')
    
    # Enhanced download settings
    chrome_options.add_argument('--disable-software-rasterizer')
    chrome_options.add_argument('--disable-extensions')
    
    # Configure download behavior
    prefs = {
        "download.default_directory": os.path.abspath(download_dir),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
        "browser.helperApps.neverAsk.saveToDisk": (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,"
            "application/vnd.ms-excel,application/msexcel,application/x-msexcel,"
            "application/x-ms-excel,application/x-excel,application/x-dos_ms_excel,"
            "application/xls,application/x-xls,application/excel,"
            "application/pdf,application/x-pdf,application/octet-stream,"
            "application/download,application/force-download"
        ),
        "plugins.always_open_pdf_externally": True,
        "pdfjs.disabled": True
    }
    chrome_options.add_experimental_option("prefs", prefs)
    
    return webdriver.Chrome(options=chrome_options)

def wait_for_download(download_dir: str, timeout: int = 60) -> bool:
    """
    Wait for the download to complete and return True if successful.
    Enhanced with better file detection and longer timeout.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        # Check for both temporary and completed files
        files = os.listdir(download_dir)
        if any(f.endswith(('.xlsx', '.xls', '.pdf', '.csv')) for f in files):
            # Found completed file
            time.sleep(2)  # Wait for file to be fully written
            return True
        elif any(f.endswith('.crdownload') for f in files):
            # Download in progress, keep waiting
            time.sleep(1)
            continue
        time.sleep(1)
    return False

class SmartDownloader:
    def __init__(self, download_dir: str = DOWNLOADS_DIR):
        self.download_manager = DownloadManager(download_dir)
        self.download_dir = download_dir
        self.session = requests.Session()
        self.processed_urls = set()
        self.downloaded_files = set()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def normalize_url(self, url: str) -> str:
        """Normalize URL by removing fragments and query parameters."""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    def get_file_basename(self, url: str, content_type: str = None) -> str:
        """Extract base filename from URL or generate one from content type."""
        # Remove query parameters and fragments
        clean_url = self.normalize_url(url)
        
        # Try to get filename from the URL path
        filename = os.path.basename(unquote(clean_url))
        
        # If no filename in URL, generate one from content type
        if not filename or '.' not in filename:
            ext = None
            if content_type:
                if 'pdf' in content_type.lower():
                    ext = '.pdf'
                elif any(x in content_type.lower() for x in ['excel', 'spreadsheet', 'xls']):
                    ext = '.xlsx'
            
            if ext:
                filename = f"document{ext}"
            else:
                filename = "document.pdf"  # default
        
        return filename
        
    def is_downloadable_file(self, url: str) -> Tuple[bool, Optional[str]]:
        """Check if the URL points directly to a downloadable file."""
        try:
            # Get headers without downloading the full file
            head = self.session.head(url, allow_redirects=True, timeout=5)
            content_type = head.headers.get('content-type', '').lower()
            
            # Check if it's a PDF or Excel file
            is_downloadable = any(ft in content_type for ft in [
                'pdf', 'excel', 'spreadsheet', 'xls', 'application/',
                'octet-stream', 'ms-excel'
            ])
            
            return is_downloadable, content_type
        except requests.RequestException as e:
            print(f"Warning: Failed to check content type for {url}: {str(e)}")
            # If we can't check content type, try to guess from URL
            is_downloadable = any(url.lower().endswith(ext) for ext in ['.pdf', '.xlsx', '.xls', '.csv'])
            return is_downloadable, None

    def extract_download_links(self, url: str) -> List[str]:
        """Extract PDF and Excel download links from a webpage."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all links
            links = set()  # Use set to avoid duplicates
            for a in soup.find_all('a', href=True):
                href = a['href']
                full_url = urljoin(url, href)
                normalized_url = self.normalize_url(full_url)
                
                # Skip javascript and other non-http(s) links
                if not normalized_url.startswith(('http://', 'https://')):
                    continue
                
                # Check if the URL is from a valid year
                if not is_valid_year(normalized_url):
                    continue
                
                # Check if the link points to a PDF or Excel file
                if any(ext in href.lower() for ext in ['.pdf', '.xlsx', '.xls', '.csv']):
                    links.add(normalized_url)
                # Also check content type for links without obvious extensions
                else:
                    is_downloadable, _ = self.is_downloadable_file(normalized_url)
                    if is_downloadable:
                        links.add(normalized_url)
            
            return list(links)
        except requests.RequestException as e:
            print(f"Warning: Failed to extract links from {url}: {str(e)}")
            return []

    def try_download_with_viewer(self, url: str, use_headless: bool = False) -> Tuple[bool, Optional[str]]:
        """
        Attempt to download using Chrome browser. Headless mode is optional and disabled by default.
        """
        driver = None
        try:
            print(f"Attempting to download using Chrome browser for: {url}")
            
            # Get initial files in download directory
            files_before = set(os.listdir(self.download_dir))
            print(f"Files in download directory before: {len(files_before)}")
            
            # Setup and initialize Chrome
            driver = setup_chrome_driver(self.download_dir, use_headless)
            driver.set_page_load_timeout(30)
            
            print("Chrome browser initialized successfully")
            print(f"Using {'headless' if use_headless else 'normal'} mode")
            
            # Navigate to URL
            driver.get(url)
            print("Page loaded successfully")
            
            # Wait for download to complete
            if wait_for_download(self.download_dir):
                # Get new files
                current_files = set(os.listdir(self.download_dir))
                new_files = current_files - files_before
                
                if new_files:
                    new_file_path = os.path.join(
                        self.download_dir,
                        list(new_files)[0]
                    )
                    print(f"Successfully downloaded: {os.path.basename(new_file_path)}")
                    return True, new_file_path
            
            print("Download wait time exceeded. No new files detected.")
            return False, None
            
        except Exception as e:
            print(f"Error in Chrome download attempt: {str(e)}")
            return False, None
        finally:
            if driver:
                try:
                    driver.quit()
                    print("Chrome browser closed successfully")
                except Exception as e:
                    print(f"Error closing Chrome browser: {str(e)}")

    def smart_download(self, url: str, use_headless: bool = False) -> Tuple[bool, List[str]]:
        """
        Smartly handle both direct file downloads and webpage scraping.
        Returns (success, list of downloaded files)
        """
        # Skip if URL is not from valid years
        if not is_valid_year(url):
            print(f"Skipping URL (not from years {VALID_YEARS}): {url}")
            return True, []

        # Skip if URL has already been processed
        normalized_url = self.normalize_url(url)
        if normalized_url in self.processed_urls:
            print(f"Skipping already processed URL: {url}")
            return True, []
        
        self.processed_urls.add(normalized_url)
        downloaded_files = []
        skipped_files = []  # Track skipped files
        
        try:
            # First check if it's a direct download link
            is_direct_download = any(url.lower().endswith(ext) for ext in ['.pdf', '.xlsx', '.xls', '.csv'])
            
            if not is_direct_download:
                is_downloadable, content_type = self.is_downloadable_file(url)
                is_direct_download = is_downloadable
            
            if is_direct_download:
                print("Direct file download detected...")
                filename = self.get_file_basename(url)
                
                if filename in self.downloaded_files:
                    print(f"Skipping download: {filename} (already downloaded from another URL)")
                    skipped_files.append(filename)
                    return True, []
                
                print(f"Attempting to download: {url}")
                
                # Try download with Chrome first for direct files
                success, result = self.try_download_with_viewer(url, use_headless)
                
                # If Chrome download fails, try normal download
                if not success:
                    print("Chrome download failed, trying direct download...")
                    success, result = self.download_manager.download(url)
                
                if success:
                    self.downloaded_files.add(os.path.basename(result))
                    downloaded_files.append(result)
                    print(f"Successfully downloaded: {os.path.basename(result)}")
                    return True, downloaded_files
                else:
                    print(f"Failed to download: {url} (all download methods failed)")
                    return False, []
            
            # If not a direct download, try to find download links in the page
            print("Scanning webpage for downloadable files...")
            download_links = self.extract_download_links(url)
            
            if not download_links:
                # Try Chrome download as last resort
                print("No downloadable links found, trying Chrome download...")
                success, result = self.try_download_with_viewer(url, use_headless)
                if success:
                    self.downloaded_files.add(os.path.basename(result))
                    downloaded_files.append(result)
                    return True, downloaded_files
                    
                print("No downloadable files found in the webpage")
                return False, []
            
            print(f"Found {len(download_links)} potential download links")
            
            # Download all found files
            for link in download_links:
                filename = self.get_file_basename(link)
                if filename in self.downloaded_files:
                    print(f"Skipping download: {filename} (already downloaded from another URL)")
                    skipped_files.append(filename)
                    continue
                
                print(f"Attempting to download: {link}")
                # Try Chrome download first
                success, result = self.try_download_with_viewer(link, use_headless)
                
                # If Chrome download fails, try normal download
                if not success:
                    print("Chrome download failed, trying direct download...")
                    success, result = self.download_manager.download(link)
                
                if success:
                    self.downloaded_files.add(os.path.basename(result))
                    downloaded_files.append(result)
                    print(f"Successfully downloaded: {os.path.basename(result)}")
                else:
                    print(f"Failed to download: {link} (all download methods failed)")
            
            if downloaded_files:
                return True, downloaded_files
            elif skipped_files:
                print(f"All files were already downloaded from other URLs ({', '.join(skipped_files)})")
                return True, []
            else:
                return False, []
            
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            return False, downloaded_files

def test_multiple_urls(urls: List[str], use_headless: bool = False):
    """Test the downloader with multiple URLs."""
    # Initialize downloader with test downloads directory
    downloader = SmartDownloader(DOWNLOADS_DIR)
    
    # Track results
    results = {
        'successful': [],
        'failed': [],
        'skipped': []  # Track skipped URLs separately
    }
    
    # Process each URL
    for i, url in enumerate(urls, 1):
        print(f"\nProcessing URL {i}/{len(urls)}")
        print(f"URL: {url}")
        
        try:
            success, downloaded_files = downloader.smart_download(url, use_headless)
            
            if success:
                if downloaded_files:
                    print(f"Successfully downloaded files:")
                    for file in downloaded_files:
                        print(f"  - {os.path.basename(file)}")
                    results['successful'].extend(downloaded_files)
                else:
                    print("URL processed successfully (files were already downloaded)")
                    results['skipped'].append(url)
            else:
                print("Failed to download from URL (no files downloaded)")
                results['failed'].append(url)
                
        except Exception as e:
            print(f"Error processing URL: {str(e)}")
            results['failed'].append(url)
        
        print("-" * 80)  # Separator between URLs
    
    # Print batch summary
    print("\nDownload Summary:")
    print(f"Total URLs processed: {len(urls)}")
    print(f"New files downloaded: {len(results['successful'])}")
    print(f"URLs skipped (duplicates): {len(results['skipped'])}")
    print(f"Failed URLs: {len(results['failed'])}")
    
    if results['successful']:
        print("\nSuccessfully downloaded new files:")
        for file in results['successful']:
            print(f"  - {os.path.basename(file)}")
    
    if results['skipped']:
        print("\nSkipped URLs (files already downloaded):")
        for url in results['skipped']:
            print(f"  - {url}")
    
    if results['failed']:
        print("\nFailed URLs:")
        for url in results['failed']:
            print(f"  - {url}")
    
    return results

def main():
    # Check for limit argument
    limit_per_company = None
    if len(sys.argv) > 1:
        try:
            limit_per_company = int(sys.argv[1])
            print(f"\nLimiting to {limit_per_company} files per company")
        except ValueError:
            print("Invalid limit argument. Please provide a number (e.g., 'python download_reports.py 10')")
            return

    # Read URLs from report_link.txt
    report_links_file = os.path.join(OUTPUT_DIR, 'report_link.txt')
    if not os.path.exists(report_links_file):
        print(f"Error: {report_links_file} not found!")
        return

    print(f"\nReading report links from {report_links_file}")
    company_urls = {}  # Dictionary to store URLs by company
    current_company = None
    
    # First pass: collect URLs by company
    with open(report_links_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a company header
            if line.startswith('==='):
                current_company = line.strip('= ')
                company_urls[current_company] = []
                print(f"\n{'='*80}")
                print(f"Found {current_company}")
                print(f"{'='*80}")
                continue
            
            # Extract URL from the line (format: "type (year): url")
            if current_company and ': ' in line:
                parts = line.split(': ', 1)
                if len(parts) == 2:
                    url = parts[1].strip()
                    company_urls[current_company].append(url)
    
    # Process URLs by company with optional limit
    all_urls = []
    for company, urls in company_urls.items():
        print(f"\n{'='*80}")
        print(f"Processing {company}")
        print(f"{'='*80}")
        
        # Apply limit if specified
        company_urls_to_process = urls[:limit_per_company] if limit_per_company else urls
        print(f"Processing {len(company_urls_to_process)} out of {len(urls)} URLs for this company")
        all_urls.extend(company_urls_to_process)
    
    total_urls = len(all_urls)
    print(f"\nTotal URLs to process: {total_urls}")
    
    # Process URLs in batches
    batch_size = 10
    for i in range(0, len(all_urls), batch_size):
        batch = all_urls[i:i + batch_size]
        print(f"\nProcessing URLs {i+1}-{min(i+batch_size, total_urls)} of {total_urls}")
        test_multiple_urls(batch, use_headless=False)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}") 