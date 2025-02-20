import os
import time
import requests
import mimetypes
from urllib.parse import urlparse, unquote
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import Optional, Tuple

class DownloadManager:
    def __init__(self, download_dir: str):
        """
        Initialize the download manager.
        
        Args:
            download_dir (str): Directory where files will be downloaded
        """
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)
        
        # Setup Chrome options for Selenium
        self.chrome_options = Options()
        self.chrome_options.add_argument('--headless')  # Run in headless mode
        self.chrome_options.add_argument('--disable-gpu')
        self.chrome_options.add_experimental_option(
            'prefs',
            {
                'download.default_directory': os.path.abspath(download_dir),
                'download.prompt_for_download': False,
                'download.directory_upgrade': True,
                'plugins.always_open_pdf_externally': True,
                'safebrowsing.enabled': True
            }
        )

    def _get_content_type(self, url: str) -> Optional[str]:
        """
        Get the content type of a URL using a HEAD request.
        
        Args:
            url (str): URL to check
            
        Returns:
            Optional[str]: Content type if available
        """
        try:
            response = requests.head(url, allow_redirects=True, timeout=10)
            return response.headers.get('content-type')
        except requests.RequestException:
            return None

    def _get_filename_from_response(self, response) -> str:
        """
        Extract filename from response headers or URL.
        
        Args:
            response: Requests response object
            
        Returns:
            str: Extracted filename
        """
        # Try to get filename from Content-Disposition header
        cd = response.headers.get('content-disposition')
        if cd:
            if 'filename=' in cd:
                filename = cd.split('filename=')[1].strip('"\'')
                return unquote(filename)

        # Fall back to URL path
        url_path = urlparse(response.url).path
        filename = os.path.basename(url_path)
        
        # If no extension, try to guess from content type
        if '.' not in filename:
            content_type = response.headers.get('content-type', '').split(';')[0]
            ext = mimetypes.guess_extension(content_type)
            if ext:
                filename = f"document{ext}"
            else:
                # Default to PDF if we can't determine the type
                filename = "document.pdf"

        return filename

    def _download_with_requests(self, url: str) -> Tuple[bool, str]:
        """
        Download file using requests library.
        
        Args:
            url (str): URL to download from
            
        Returns:
            Tuple[bool, str]: (Success status, Filepath or error message)
        """
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            filename = self._get_filename_from_response(response)
            filepath = os.path.join(self.download_dir, filename)
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            return True, filepath
        except requests.RequestException as e:
            return False, str(e)

    def _download_with_selenium(self, url: str) -> Tuple[bool, str]:
        """
        Download file using Selenium for browser-triggered downloads.
        
        Args:
            url (str): URL to download from
            
        Returns:
            Tuple[bool, str]: (Success status, Filepath or error message)
        """
        driver = None
        try:
            driver = webdriver.Chrome(options=self.chrome_options)
            driver.get(url)
            
            # Wait for potential download to complete
            time.sleep(5)  # Basic wait for download to start
            
            # Check download directory for new files
            files = os.listdir(self.download_dir)
            new_files = [f for f in files if os.path.getmtime(
                os.path.join(self.download_dir, f)
            ) > time.time() - 10]  # Files modified in last 10 seconds
            
            if new_files:
                return True, os.path.join(self.download_dir, new_files[0])
            return False, "No file downloaded"
            
        except Exception as e:
            return False, str(e)
        finally:
            if driver:
                driver.quit()

    def download(self, url: str) -> Tuple[bool, str]:
        """
        Download a file from the given URL, handling both direct downloads
        and browser-triggered downloads.
        
        Args:
            url (str): URL to download from
            
        Returns:
            Tuple[bool, str]: (Success status, Filepath or error message)
        """
        # First try with requests
        content_type = self._get_content_type(url)
        
        if content_type and ('pdf' in content_type.lower() or 
                           'excel' in content_type.lower() or
                           'spreadsheet' in content_type.lower()):
            return self._download_with_requests(url)
        
        # If content type check fails or isn't a direct file,
        # try with Selenium
        return self._download_with_selenium(url)

    def is_duplicate(self, filepath: str) -> bool:
        """
        Check if a file already exists in the download directory.
        
        Args:
            filepath (str): Path to check
            
        Returns:
            bool: True if file exists
        """
        return os.path.exists(filepath) 