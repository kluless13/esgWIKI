from crawl4ai import BrowserConfig, ExtractionStrategy

def get_browser_config() -> BrowserConfig:
    return BrowserConfig(
        headless=True,
        ignore_https_errors=True
    )

def get_llm_strategy() -> ExtractionStrategy:
    return ExtractionStrategy(
        model="gpt-3.5-turbo",
        temperature=0.7
    ) 