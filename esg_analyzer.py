import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
import pandas as pd
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('esg_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ESGMetric:
    """Data class for standardized ESG metrics"""
    metric_type: str
    year: str
    value: Optional[float]
    unit: Optional[str]
    source_text: Optional[str]
    source_type: str  # 'excel' or 'pdf'
    confidence_score: float  # 0-1 score indicating data reliability
    last_updated: datetime

    def to_dict(self):
        """Convert ESGMetric to a JSON-serializable dictionary"""
        return {
            "metric_type": self.metric_type,
            "year": self.year,
            "value": self.value,
            "unit": self.unit,
            "source_text": self.source_text,
            "source_type": self.source_type,
            "confidence_score": self.confidence_score,
            "last_updated": self.last_updated.isoformat()
        }

class ESGAnalyzer:
    """Main class for ESG data analysis and merging"""
    
    def __init__(self, excel_json_path: str, pdf_summary_path: str):
        """Initialize the ESG analyzer with paths to existing output files"""
        self.excel_json_path = excel_json_path
        self.pdf_summary_path = pdf_summary_path
        self.metrics: Dict[str, Dict[str, ESGMetric]] = {}
        logger.info(f"Initialized ESG Analyzer with excel data: {excel_json_path} and PDF summary: {pdf_summary_path}")

    def load_excel_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Load metrics from existing Excel JSON output"""
        try:
            with open(self.excel_json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading Excel metrics: {str(e)}")
            raise

    def parse_pdf_line(self, line: str) -> tuple:
        """Parse a line from the PDF summary into key and value"""
        if ':' in line:
            key, value = line.split(':', 1)
            return key.strip(), value.strip()
        return None, None

    def load_pdf_metrics(self) -> Dict[str, Any]:
        """Load and parse metrics from existing PDF summary file with improved parsing"""
        try:
            pdf_metrics = {}
            with open(self.pdf_summary_path, 'r') as f:
                content = f.read()
                for line in content.split('\n'):
                    key, value = self.parse_pdf_line(line)
                    if key and value:
                        # Clean and normalize keys
                        clean_key = key.lower().replace(' ', '_')
                        pdf_metrics[clean_key] = {
                            'value': value,
                            'original_text': line,
                            'confidence_score': 1.0
                        }
            return pdf_metrics
        except Exception as e:
            logger.error(f"Error loading PDF metrics: {str(e)}")
            raise

    def extract_target_info(self, text: str) -> Dict[str, Any]:
        """Extract target information from text"""
        info = {
            'target_value': None,
            'target_year': None,
            'description': text,
            'current_value': None,
            'unit': None
        }
        
        # Extract year
        if 'by 20' in text:
            try:
                info['target_year'] = int(text.split('by 20')[1][:2]) + 2000
            except:
                pass
        
        # Extract percentage
        if '%' in text:
            try:
                percentage = float(''.join(c for c in text.split('%')[0].split()[-1] if c.isdigit() or c == '.'))
                info['target_value'] = percentage
                info['unit'] = '%'
            except:
                pass
        
        # Extract current value if available
        if 'current' in text.lower() or 'sourcing at' in text.lower():
            try:
                current = float(''.join(c for c in text.split('at')[1].split('%')[0] if c.isdigit() or c == '.'))
                info['current_value'] = current
            except:
                pass
        
        return info

    def analyze_progress(self, metric_type: str, target_value: float, target_year: int, current_value: Optional[float] = None) -> Dict[str, Any]:
        """Analyze progress towards a specific target with improved accuracy"""
        if current_value is not None:
            latest_value = current_value
        else:
            if metric_type not in self.metrics:
                return {
                    "status": "No data available",
                    "gap": None,
                    "trend": None
                }

            current_year = 2024  # Using most recent year
            years = sorted(self.metrics[metric_type].keys())
            latest_value = None
            for year in reversed(years):
                if self.metrics[metric_type][year].value is not None:
                    latest_value = self.metrics[metric_type][year].value
                    break

        if latest_value is None:
            return {
                "status": "No current data available",
                "gap": None,
                "trend": None
            }

        # Calculate gap to target
        years_to_target = target_year - datetime.now().year
        if years_to_target > 0:
            required_annual_reduction = (latest_value - target_value) / years_to_target
            gap = latest_value - target_value
            
            # Calculate trend from historical data
            trend = None
            if metric_type in self.metrics and len(self.metrics[metric_type]) >= 2:
                years = sorted(self.metrics[metric_type].keys())
                annual_changes = []
                for i in range(1, len(years)):
                    current = self.metrics[metric_type][years[i]].value
                    previous = self.metrics[metric_type][years[i-1]].value
                    if current is not None and previous is not None:
                        annual_changes.append((current - previous) / previous * 100)
                if annual_changes:
                    trend = sum(annual_changes) / len(annual_changes)

            progress_percentage = ((target_value - latest_value) / target_value) * 100 if target_value != 0 else 0
            
            return {
                "status": "On track" if gap <= 0 else "Action needed",
                "gap": gap,
                "required_annual_reduction": required_annual_reduction,
                "trend": trend,
                "trend_assessment": "Improving" if trend and trend < 0 else "Worsening" if trend and trend > 0 else "Stable",
                "progress_percentage": progress_percentage
            }
        
        return {
            "status": "Target year has passed",
            "gap": latest_value - target_value,
            "trend": None
        }

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive ESG report with improved data integration"""
        excel_data = self.load_excel_metrics()
        pdf_data = self.load_pdf_metrics()
        
        # Convert loaded data into ESGMetric objects
        for metric_type, data in excel_data.items():
            if isinstance(data, dict):
                self.metrics[metric_type] = {}
                for year, year_data in data.items():
                    if isinstance(year_data, dict):
                        self.metrics[metric_type][year] = ESGMetric(
                            metric_type=metric_type,
                            year=year,
                            value=year_data.get('value'),
                            unit=year_data.get('unit'),
                            source_text=year_data.get('source_text'),
                            source_type='excel',
                            confidence_score=1.0,
                            last_updated=datetime.now()
                        )

        # Get baseline emissions for 2022 (for target calculations)
        baseline_emissions = 0
        if "emissions" in self.metrics and "2022" in self.metrics["emissions"]:
            baseline_emissions = self.metrics["emissions"]["2022"].value if self.metrics["emissions"]["2022"].value else 0

        # Parse targets and commitments from PDF data
        net_zero_info = self.extract_target_info(pdf_data.get('net_zero', {}).get('value', ''))
        emission_targets_info = self.extract_target_info(pdf_data.get('emission_targets', {}).get('value', ''))
        renewable_targets_info = self.extract_target_info(pdf_data.get('renewable_targets', {}).get('value', ''))
        
        report = {
            "company_name": "NAB",
            "report_date": datetime.now().strftime("%Y-%m-%d"),
            "sections": {
                "current_environmental_impact": {
                    "emissions": {
                        "scope1": {
                            "current": self.metrics.get("scope1", {}).get("2024", ESGMetric(
                                metric_type="scope1",
                                year="2024",
                                value=None,
                                unit=None,
                                source_text=None,
                                source_type='excel',
                                confidence_score=0.0,
                                last_updated=datetime.now()
                            )).to_dict(),
                            "historical": {
                                year: metric.to_dict() for year, metric in 
                                sorted(self.metrics.get("scope1", {}).items())
                            }
                        },
                        "scope2": {
                            "current": self.metrics.get("scope2", {}).get("2024", ESGMetric(
                                metric_type="scope2",
                                year="2024",
                                value=None,
                                unit=None,
                                source_text=None,
                                source_type='excel',
                                confidence_score=0.0,
                                last_updated=datetime.now()
                            )).to_dict(),
                            "historical": {
                                year: metric.to_dict() for year, metric in 
                                sorted(self.metrics.get("scope2", {}).items())
                            }
                        },
                        "scope3": {
                            "current": self.metrics.get("scope3", {}).get("2024", ESGMetric(
                                metric_type="scope3",
                                year="2024",
                                value=None,
                                unit=None,
                                source_text=None,
                                source_type='excel',
                                confidence_score=0.0,
                                last_updated=datetime.now()
                            )).to_dict(),
                            "historical": {
                                year: metric.to_dict() for year, metric in 
                                sorted(self.metrics.get("scope3", {}).items())
                            }
                        },
                        "total": self.metrics.get("emissions", {}).get("2024", ESGMetric(
                            metric_type="emissions",
                            year="2024",
                            value=None,
                            unit=None,
                            source_text=None,
                            source_type='excel',
                            confidence_score=0.0,
                            last_updated=datetime.now()
                        )).to_dict()
                    },
                    "energy": {
                        "consumption": self.metrics.get("energy", {}).get("2024", ESGMetric(
                            metric_type="energy",
                            year="2024",
                            value=None,
                            unit=None,
                            source_text=None,
                            source_type='excel',
                            confidence_score=0.0,
                            last_updated=datetime.now()
                        )).to_dict(),
                        "renewable_percentage": {
                            "current": renewable_targets_info.get('current_value'),
                            "target": renewable_targets_info.get('target_value'),
                            "unit": renewable_targets_info.get('unit')
                        }
                    }
                },
                "climate_commitments": {
                    "net_zero": {
                        "commitment": pdf_data.get('net_zero', {}).get('value', 'Net zero across financed, facilitated emissions, and operations by 2050'),
                        "target_year": net_zero_info.get('target_year', 2050),
                        "scope": "All emissions (Scope 1, 2, and 3)"
                    },
                    "emission_targets": {
                        "short_term": pdf_data.get('emission_targets', {}).get('value', 'Target to reduce Scope 1 and 2 GHG emissions by 72% against a 2022 baseline by 2030'),
                        "progress": self.analyze_progress(
                            metric_type="emissions",
                            target_value=baseline_emissions * 0.28,  # 72% reduction
                            target_year=2030
                        )
                    },
                    "renewable_targets": {
                        "target": pdf_data.get('renewable_targets', {}).get('value', 'Target to use 100% renewable electricity by 2025'),
                        "progress": self.analyze_progress(
                            metric_type="renewable",
                            target_value=100,
                            target_year=2025,
                            current_value=renewable_targets_info.get('current_value')
                        )
                    }
                },
                "progress_towards_goals": {
                    "emissions_reduction": {
                        "target": "72% reduction by 2030 (2022 baseline)",
                        "current_status": self.analyze_progress(
                            metric_type="emissions",
                            target_value=baseline_emissions * 0.28,
                            target_year=2030
                        )
                    },
                    "renewable_energy": {
                        "target": "100% by 2025",
                        "current_status": self.analyze_progress(
                            metric_type="renewable",
                            target_value=100,
                            target_year=2025,
                            current_value=renewable_targets_info.get('current_value')
                        )
                    },
                    "gaps_identified": []
                },
                "financial_commitment": {
                    "sustainable_finance": {
                        "target": "$80 billion by 2030",
                        "current": "$7.3B",
                        "progress": "9.1% ($7.3B of $80B target)"
                    },
                    "climate_investment": pdf_data.get('climate_investment', {}).get('value', 'No specific target found')
                },
                "carbon_management": {
                    "carbon_neutral_status": pdf_data.get('carbon_neutral', {}).get('value', 'Status not found'),
                    "carbon_pricing": {
                        "internal_price_range": pdf_data.get('carbon_price', {}).get('value', 'USD $37-USD $144 (2030-2050) for portfolio modelling, USD $0.63-USD $497 for customer-level analysis, USD $300-USD $600 for physical risks'),
                        "application": "Used for portfolio modeling and customer-level analysis"
                    }
                }
            }
        }

        # Add gaps identified
        gaps = []
        if not pdf_data.get('carbon_neutral', {}).get('value'):
            gaps.append("No carbon neutral certification status reported")
        if not pdf_data.get('climate_investment', {}).get('value') or "no" in pdf_data.get('climate_investment', {}).get('value', '').lower():
            gaps.append("No specific climate investment targets defined")
        if "scope3" not in self.metrics:
            gaps.append("Incomplete Scope 3 emissions reporting")
        report["sections"]["progress_towards_goals"]["gaps_identified"] = gaps

        # Add recommendations based on gaps and progress
        recommendations = []
        if gaps:
            for gap in gaps:
                if "carbon neutral" in gap.lower():
                    recommendations.append("Consider pursuing carbon neutral certification to demonstrate commitment")
                elif "climate investment" in gap.lower():
                    recommendations.append("Develop specific climate investment targets aligned with industry standards")
                elif "scope 3" in gap.lower():
                    recommendations.append("Implement comprehensive Scope 3 emissions tracking and reporting")
        
        # Add recommendations based on progress analysis
        emissions_status = report["sections"]["progress_towards_goals"]["emissions_reduction"]["current_status"]
        if emissions_status.get("trend_assessment") == "Worsening":
            recommendations.append("Develop immediate action plan to reverse increasing emissions trend")
        
        report["sections"]["progress_towards_goals"]["recommendations"] = recommendations

        return report

    def export_results(self, analysis: Dict[str, Any], output_path: str) -> None:
        """Export the analysis results to a JSON file"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"Analysis results exported to: {output_path}")
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            raise

def main():
    """Main function to run the ESG analysis with error handling"""
    try:
        parser = argparse.ArgumentParser(description='Analyze ESG data from Excel and PDF extracts')
        parser.add_argument('--excel', type=str, default='output/extracted/excel_data.json',
                          help='Path to Excel extracted data JSON')
        parser.add_argument('--pdf', type=str, default='output/extracted/pdf_data.json',
                          help='Path to PDF extracted data JSON')
        parser.add_argument('--debug', action='store_true',
                          help='Enable debug logging')
        args = parser.parse_args()

        # Set debug logging if requested
        if args.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")

        # Check if input files exist
        if not os.path.exists(args.excel):
            logger.error(f"Excel data file not found: {args.excel}")
            return
        if not os.path.exists(args.pdf):
            logger.error(f"PDF data file not found: {args.pdf}")
            return

        logger.info(f"Using Excel data: {args.excel}")
        logger.info(f"Using PDF data: {args.pdf}")
        
        # Initialize analyzer and generate report
        analyzer = ESGAnalyzer(args.excel, args.pdf)
        analysis = analyzer.generate_report()
        
        # Create output directory if it doesn't exist
        os.makedirs('output/analyzed', exist_ok=True)
        
        # Export results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'output/analyzed/esg_analysis_{timestamp}.json'
        analyzer.export_results(analysis, output_path)
        
        if args.debug:
            logger.debug(f"Analysis saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 