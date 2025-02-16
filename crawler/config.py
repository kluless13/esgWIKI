# config.py

BASE_URL = "https://www.listcorp.com"
CSS_SELECTOR = "div.content-body"  # Assuming the sustainability report content is within this div
REPORT_URL_KEYWORDS = [
    "sustainability-report",
    "sustainability-paper",
]
REQUIRED_KEYS = [
    "company_name",
    "current_energy_status",
    "target_energy_status",
    "gap",
    "scope1_status",
    "scope2_status",
    "scope3_status",
]
