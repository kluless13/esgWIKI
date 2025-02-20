import os
import json
import time
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

def setup_chrome_driver(download_dir: str, use_headless: bool = False) -> webdriver.Chrome:
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
    def __init__(self, download_dir: str):
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
        clean_url = self.normalize_url(url)
        filename = os.path.basename(unquote(clean_url))
        
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
            head = self.session.head(url, allow_redirects=True, timeout=5)
            content_type = head.headers.get('content-type', '').lower()
            
            is_downloadable = any(ft in content_type for ft in [
                'pdf', 'excel', 'spreadsheet', 'xls', 'application/',
                'octet-stream', 'ms-excel'
            ])
            
            return is_downloadable, content_type
        except requests.RequestException as e:
            print(f"Warning: Failed to check content type for {url}: {str(e)}")
            is_downloadable = any(url.lower().endswith(ext) for ext in ['.pdf', '.xlsx', '.xls', '.csv'])
            return is_downloadable, None

    def extract_download_links(self, url: str) -> List[str]:
        """Extract PDF and Excel download links from a webpage."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            links = set()
            for a in soup.find_all('a', href=True):
                href = a['href']
                full_url = urljoin(url, href)
                normalized_url = self.normalize_url(full_url)
                
                if not normalized_url.startswith(('http://', 'https://')):
                    continue
                
                if any(ext in href.lower() for ext in ['.pdf', '.xlsx', '.xls', '.csv']):
                    links.add(normalized_url)
                else:
                    is_downloadable, _ = self.is_downloadable_file(normalized_url)
                    if is_downloadable:
                        links.add(normalized_url)
            
            return list(links)
        except requests.RequestException as e:
            print(f"Warning: Failed to extract links from {url}: {str(e)}")
            return []

    def try_download_with_viewer(self, url: str, use_headless: bool = False) -> Tuple[bool, Optional[str]]:
        """Attempt to download using Chrome browser."""
        driver = None
        try:
            print(f"Attempting to download using Chrome browser for: {url}")
            
            files_before = set(os.listdir(self.download_dir))
            
            driver = setup_chrome_driver(self.download_dir, use_headless)
            driver.set_page_load_timeout(30)
            
            driver.get(url)
            
            if wait_for_download(self.download_dir):
                current_files = set(os.listdir(self.download_dir))
                new_files = current_files - files_before
                
                if new_files:
                    new_file_path = os.path.join(
                        self.download_dir,
                        list(new_files)[0]
                    )
                    print(f"Successfully downloaded: {os.path.basename(new_file_path)}")
                    return True, new_file_path
            
            return False, None
            
        except Exception as e:
            print(f"Error in Chrome download attempt: {str(e)}")
            return False, None
        finally:
            if driver:
                try:
                    driver.quit()
                except Exception as e:
                    print(f"Error closing Chrome browser: {str(e)}")

    def smart_download(self, url: str, use_headless: bool = False) -> Tuple[bool, str]:
        """Smart download handling for both direct and indirect downloads."""
        if url in self.processed_urls:
            print(f"URL already processed: {url}")
            return False, "URL already processed"

        self.processed_urls.add(url)
        
        try:
            # First try direct download
            is_downloadable, content_type = self.is_downloadable_file(url)
            if is_downloadable:
                success, result = self.download_manager.download(url)
                if success:
                    return True, result
            
            # If direct download fails, try browser download
            success, file_path = self.try_download_with_viewer(url, use_headless)
            if success:
                return True, file_path
            
            # If both methods fail, try to find downloadable links on the page
            download_links = self.extract_download_links(url)
            for link in download_links:
                if link not in self.processed_urls:
                    success, result = self.smart_download(link, use_headless)
                    if success:
                        return True, result
            
            return False, "No downloadable content found"
            
        except Exception as e:
            return False, str(e)

def get_domain(url: str) -> str:
    """Extract domain from URL to use in organizing downloads."""
    parsed = urlparse(url)
    return parsed.netloc.replace('www.', '')

def load_urls(filepath: str) -> List[str]:
    """Load URLs from the report_link.txt file."""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def organize_urls_by_company(urls: List[str]) -> Dict[str, Set[str]]:
    """Group URLs by company domain to avoid duplicate downloads."""
    company_urls = {}
    for url in urls:
        domain = get_domain(url)
        if domain not in company_urls:
            company_urls[domain] = set()
        company_urls[domain].add(url)
    return company_urls

def main():
    # Initialize paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    download_dir = os.path.join(current_dir, 'downloads')
    links_file = os.path.join(current_dir, 'report_link.txt')
    
    # Create download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    
    # Create smart downloader
    downloader = SmartDownloader(download_dir)
    
    # Load and organize URLs
    urls = load_urls(links_file)
    company_urls = organize_urls_by_company(urls)
    
    # Track download results
    download_results = {
        'successful': [],
        'failed': []
    }
    
    # Process each company's URLs
    for domain, urls in company_urls.items():
        print(f"\nProcessing files for {domain}")
        company_dir = os.path.join(download_dir, domain)
        os.makedirs(company_dir, exist_ok=True)
        
        for url in urls:
            print(f"\nAttempting to download: {url}")
            success, result = downloader.smart_download(url, use_headless=True)
            
            if success:
                print(f"Successfully downloaded to: {result}")
                download_results['successful'].append({
                    'url': url,
                    'filepath': result,
                    'company': domain
                })
            else:
                print(f"Failed to download: {url}")
                print(f"Error: {result}")
                download_results['failed'].append({
                    'url': url,
                    'error': result,
                    'company': domain
                })
    
    # Save download results
    results_file = os.path.join(download_dir, 'download_results.json')
    with open(results_file, 'w') as f:
        json.dump(download_results, f, indent=2)
    
    print(f"\nDownload Summary:")
    print(f"Successful downloads: {len(download_results['successful'])}")
    print(f"Failed downloads: {len(download_results['failed'])}")
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main() 