import os
import json
import argparse
import re
import datetime
from typing import Optional, Dict, Any, List, Tuple
import PyPDF2
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import threading

# Set OpenAI API key
OPENAI_API_KEY = "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"
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Metric descriptions for different types of sustainability metrics
metric_descriptions = {
    "net_zero": "Net zero commitment year and interim targets",
    "emission_targets": "Emission reduction targets and base years",
    "renewable_targets": "Renewable energy targets and current percentage",
    "carbon_neutral": "Carbon neutral certification status and achievements",
    "carbon_price": "Internal carbon price used for decision making",
    "sustainable_finance": "Sustainable finance commitments and progress",
    "climate_investment": "Climate-related investments and initiatives"
}

# Add a thread-safe cache for page content
page_cache = {}
page_cache_lock = threading.Lock()

# Add topic mapping with keywords
topic_mapping = {
    'net_zero': {
        'primary': ['net zero', 'carbon neutral', 'net-zero'],
        'secondary': ['emissions target', 'climate ambition']
    },
    'emission_targets': {
        'primary': ['emission', 'scope 1', 'scope 2', 'scope 3'],
        'secondary': ['ghg', 'greenhouse gas']
    },
    'renewable_targets': {
        'primary': ['renewable', 'clean energy'],
        'secondary': ['solar', 'wind']
    },
    'carbon_neutral': {
        'primary': ['carbon neutral', 'neutrality'],
        'secondary': ['carbon offset']
    },
    'carbon_price': {
        'primary': ['carbon price', 'carbon pricing'],
        'secondary': ['price on carbon']
    },
    'sustainable_finance': {
        'primary': ['sustainable finance', 'green finance'],
        'secondary': ['green bonds']
    },
    'climate_investment': {
        'primary': ['climate investment', 'green investment'],
        'secondary': ['sustainable investment']
    }
}

@lru_cache(maxsize=1)
def get_llm_instance():
    """Get or create a singleton LLM instance."""
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4",
        temperature=0
    )

def create_prompt(metric_type: str, data: str) -> str:
    """Create an ultra-simplified prompt for quick extraction."""
    system_message = "You are an expert at analyzing sustainability reports. Return ONLY a one-line summary."
    
    user_message = f"""What is the most important {metric_descriptions.get(metric_type, metric_type)} mentioned in this text?
    Return ONLY the key number/target/commitment in a single line.
    If nothing relevant found, return 'No target found'.

{data}"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

def get_completion(messages: list) -> str:
    """Get completion from OpenAI with retries and error handling."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            llm = get_llm_instance()
            response = llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Warning: Error on attempt {attempt + 1}: {str(e)}, retrying...")
            continue
    
    raise Exception("Failed to get valid response after all retries")

def batch_process_chunks(chunks: List[str], metric_type: str) -> List[Dict]:
    """Process multiple chunks in parallel using thread pool."""
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for chunk in chunks:
            prompt = create_prompt(metric_type, chunk)
            futures.append(executor.submit(get_completion, prompt))
        
        results = []
        for future in futures:
            try:
                response = future.result()
                result = json.loads(response)
                results.append(result)
            except Exception as e:
                print(f"Error in batch processing: {str(e)}")
                continue
        
        return results

def cache_page_content(page_num: int, content: str):
    """Thread-safe caching of page content."""
    with page_cache_lock:
        page_cache[page_num] = content

def get_cached_page(page_num: int) -> Optional[str]:
    """Thread-safe retrieval of cached page content."""
    with page_cache_lock:
        return page_cache.get(page_num)

def extract_text_from_pdf(pdf_path: str, start_page: int = None, end_page: int = None) -> List[str]:
    """Extract text from PDF file with caching, returning a list of page contents."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        
        # Set page range
        start = start_page if start_page is not None else 0
        end = min(end_page if end_page is not None else len(reader.pages), len(reader.pages))
        
        # Extract text from each page with caching
        pages = []
        for page_num in range(start, end):
            # Check cache first
            cached_content = get_cached_page(page_num)
            if cached_content:
                pages.append(cached_content)
                continue
                
            # Extract and cache if not found
            text = reader.pages[page_num].extract_text()
            if text.strip():
                cache_page_content(page_num, text)
                pages.append(text)
        
        return pages

def merge_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple results, keeping the most complete data for each year."""
    merged = {
        "2024": {
            "value": None,
            "unit": None,
            "source_text": None,
            "target_year": None,
            "target_value": None,
            "base_year": None,
            "base_value": None
        },
        "2023": {
            "value": None,
            "unit": None,
            "source_text": None,
            "target_year": None,
            "target_value": None,
            "base_year": None,
            "base_value": None
        },
        "2022": {
            "value": None,
            "unit": None,
            "source_text": None,
            "target_year": None,
            "target_value": None,
            "base_year": None,
            "base_value": None
        }
    }
    
    for result in results:
        if not result:
            continue
        
        for year in ["2024", "2023", "2022"]:
            if year not in result:
                continue
                
            year_data = result[year]
            if not year_data:
                continue
                
            # Update values if they're more complete than what we have
            for field in merged[year]:
                if field in year_data and year_data[field] is not None:
                    if merged[year][field] is None or (
                        field == "source_text" and 
                        len(str(year_data[field])) > len(str(merged[year][field]))
                    ):
                        merged[year][field] = year_data[field]
    
    return merged

def extract_toc_from_pdf(pdf_path: str) -> Dict[str, List[int]]:
    """Extract table of contents from PDF with optimized keyword search."""
    # Initialize results with thread-safe counter
    page_mapping = {topic: set() for topic in topic_mapping.keys()}
    processed_pages = set()
    
    def process_page_content(page_num: int, text: str):
        """Process a single page's content for all topics efficiently."""
        if page_num in processed_pages:
            return
            
        text_lower = text.lower()
        for topic, keywords in topic_mapping.items():
            # Check primary keywords first
            if any(kw in text_lower for kw in keywords['primary']):
                page_mapping[topic].add(page_num)
                continue
                
            # Only check secondary keywords if primary not found
            if any(kw in text_lower for kw in keywords['secondary']):
                page_mapping[topic].add(page_num)
        
        processed_pages.add(page_num)
    
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        
        # First try to get the TOC from PDF metadata
        try:
            toc = reader.outline
            if toc:
                print("\nFound structured TOC in PDF metadata")
                for entry in toc:
                    if hasattr(entry, '/Title') and hasattr(entry, '/Page'):
                        title = entry['/Title'].lower()
                        page_num = entry['/Page']
                        
                        # Process TOC entry for all topics at once
                        for topic, keywords in topic_mapping.items():
                            if any(kw in title for kw in keywords['primary'] + keywords['secondary']):
                                page_mapping[topic].add(page_num)
        except Exception as e:
            print(f"\nError reading PDF outline: {str(e)}")
        
        # Process all pages in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(len(reader.pages)):
                cached_content = get_cached_page(i)
                if cached_content:
                    futures.append(executor.submit(process_page_content, i, cached_content))
                else:
                    text = reader.pages[i].extract_text()
                    if text.strip():
                        cache_page_content(i, text)
                        futures.append(executor.submit(process_page_content, i, text))
            
            # Wait for all processing to complete
            for future in futures:
                future.result()
    
    # Convert sets to sorted lists for final output
    return {topic: sorted(list(pages)) for topic, pages in page_mapping.items()}

def get_relevant_pages(pdf_path: str, metric_type: str) -> Tuple[int, int]:
    """
    Get the relevant page range for a specific metric type.
    Returns a tuple of (start_page, end_page).
    """
    page_mapping = extract_toc_from_pdf(pdf_path)
    
    if metric_type in page_mapping and page_mapping[metric_type]:
        pages = sorted(page_mapping[metric_type])
        # Include a buffer of 2 pages before and after
        start_page = max(0, min(pages) - 2)
        end_page = max(pages) + 2
        return start_page, end_page
    
    return None, None

def chunk_text(text: str, max_chars: int = 3000) -> List[str]:
    """Split text into chunks that respect sentence boundaries and token limits."""
    # Split into sentences (simple approach)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        # Estimate tokens (rough approximation: 4 chars per token)
        sentence_length = len(sentence)
        
        if current_length + sentence_length > max_chars and current_chunk:
            # Add current chunk to chunks and start a new one
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def extract_metric(pdf_path: str, pages: List[str], metric_type: str, chunk_size: int = 3) -> str:
    """Extract quick summary for specific metric."""
    # Get relevant pages for this metric type
    start_page, end_page = get_relevant_pages(pdf_path, metric_type)
    
    # If we found relevant pages, only process those
    if start_page is not None and end_page is not None:
        pages = pages[start_page:end_page + 1]
    
    # Take only the most relevant sentences
    relevant_sentences = []
    for page in pages:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', page) if s.strip()]
        for sentence in sentences:
            # Check if sentence contains relevant keywords
            if any(keyword in sentence.lower() for keyword in topic_mapping[metric_type]['primary']):
                relevant_sentences.append(sentence)
    
    if not relevant_sentences:
        return "No relevant information found"
    
    # Take only the most relevant sentences (up to 5)
    text = " ".join(relevant_sentences[:5])
    
    # Get quick summary
    prompt = create_prompt(metric_type, text)
    try:
        response = get_completion(prompt)
        return response.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract sustainability metrics from PDF report')
    parser.add_argument('--metric', type=str, choices=list(metric_descriptions.keys()), help='Metric type to extract')
    parser.add_argument('--file', type=str, default='input/pdf/nab-climate-report.pdf', help='Path to PDF file')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs('output/extracted', exist_ok=True)

    # Extract text from PDF
    try:
        pages = extract_text_from_pdf(args.file)
        if not pages:
            print(f"Error: No text could be extracted from {args.file}")
            return

        if args.debug:
            print(f"Successfully extracted {len(pages)} pages from {args.file}")

        # Process all metrics if none specified
        metrics_to_process = [args.metric] if args.metric else list(metric_descriptions.keys())
        
        results = {}
        for metric in metrics_to_process:
            try:
                result = extract_metric(args.file, pages, metric)
                results[metric] = result
                if args.debug:
                    print(f"\nResults for {metric}:")
                    print(json.dumps(result, indent=2))
            except Exception as e:
                print(f"Error processing metric {metric}: {str(e)}")
                continue

        # Save results to JSON file
        output_file = 'output/extracted/pdf_data.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        if args.debug:
            print(f"\nResults saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: PDF file not found at {args.file}")
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")

if __name__ == '__main__':
    main() 