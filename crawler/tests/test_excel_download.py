'''
This script is used to test the Excel download functionality of the BHP website.
It uses Selenium to navigate to the URL and download the Excel file.
It then checks if the file was downloaded successfully.
'''

import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

def setup_chrome_driver(download_dir: str) -> webdriver.Chrome:
    """
    Set up Chrome driver with appropriate options for downloading Excel files.
    Always uses headless mode.
    """
    chrome_options = webdriver.ChromeOptions()
    
    # Always use headless mode
    chrome_options.add_argument('--headless')
    
    # Basic Chrome settings
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--window-size=1920,1080')
    
    # Enhanced download settings for headless mode
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
            "application/octet-stream"
        )
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
        if any(f.endswith('.xlsx') for f in files):
            # Found completed file
            time.sleep(2)  # Wait for file to be fully written
            return True
        elif any(f.endswith('.crdownload') for f in files):
            # Download in progress, keep waiting
            time.sleep(1)
            continue
        time.sleep(1)
    return False

def test_excel_download(url: str, download_dir: str) -> bool:
    """
    Test downloading an Excel file from the given URL.
    Always uses headless mode.
    """
    driver = None
    try:
        print(f"\nTesting URL: {url}")
        print("Using headless mode")
        
        # Create download directory if it doesn't exist
        os.makedirs(download_dir, exist_ok=True)
        
        # Clear download directory
        for file in os.listdir(download_dir):
            os.remove(os.path.join(download_dir, file))
        
        # Setup and launch browser
        driver = setup_chrome_driver(download_dir)
        driver.set_page_load_timeout(30)
        
        # Navigate to URL
        print("Navigating to URL...")
        driver.get(url)
        
        # Wait for download to complete
        print("Waiting for download to complete...")
        if wait_for_download(download_dir):
            files = [f for f in os.listdir(download_dir) if f.endswith('.xlsx')]
            if files:
                print(f"Successfully downloaded: {files[0]}")
                return True
            else:
                print("No Excel file found after download")
                return False
        else:
            print("Download timeout")
            return False
            
    except Exception as e:
        print(f"Error during download: {str(e)}")
        return False
    finally:
        if driver:
            driver.quit()

def main():
    # Test configuration
    download_dir = os.path.join(os.path.dirname(__file__), "downloads")
    
    # Test URLs
    urls = [
        # Direct Excel URL
        "https://www.bhp.com/-/media/documents/investors/annual-reports/2024/240827_esgstandardsanddatabook2024.xlsx",
        # Indirect URL that triggers Excel download
        "https://www.bhp.com/-/media/Documents/Investors/Annual-Reports/2024/240827_ESGStandardsandDatabook2024"
    ]
    
    # Test each URL
    for url in urls:
        success = test_excel_download(url, download_dir)
        print(f"Download {'successful' if success else 'failed'} for URL: {url}\n")

if __name__ == "__main__":
    main() 