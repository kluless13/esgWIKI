import pandas as pd
import json
import argparse
from typing import Optional, Dict, Any
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Set OpenAI API key
OPENAI_API_KEY = "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"
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

def create_prompt(metric_type: str, data: str) -> str:
    """Create a specific prompt based on the metric type."""
    base_prompt = """You are an expert at analyzing sustainability reports. 
    Given the following data, extract the {metric_type} metrics for years 2022-2024.
    
    DATA:
    {data}
    
    Return ONLY a single valid JSON object with these fields:
    {{
        "2024": {{
            "value": number or null,
            "unit": string or null,
            "source_text": string or null
        }},
        "2023": {{
            "value": number or null,
            "unit": string or null,
            "source_text": string or null
        }},
        "2022": {{
            "value": number or null,
            "unit": string or null,
            "source_text": string or null
        }}
    }}
    
    IMPORTANT: 
    1. For emissions metrics, look for:
       - Scope 1: Direct emissions from owned or controlled sources (look for "Total scope 1" or similar)
       - Scope 2: Indirect emissions from purchased electricity (look for "Total scope 2" or similar)
       - Scope 3: All other indirect emissions in the value chain (look for "Total scope 3" or similar)
       - Units are typically in tCO2e (tonnes of CO2 equivalent)
       - For market-based values, use those if available instead of location-based
       
    2. Return EXACT values from the data, do not calculate or infer
    3. Include the FULL line of text where you found the data in source_text
    4. Extract values for ALL three years (2022-2024) if available
    5. Use null for any year/metric where data is not available
    
    Example correct response:
    {{
        "2024": {{
            "value": 7313,
            "unit": "tCO2e",
            "source_text": "Total scope 1: 7,313 tCO2e"
        }},
        "2023": {{
            "value": 7590,
            "unit": "tCO2e",
            "source_text": "Total scope 1: 7,590 tCO2e"
        }},
        "2022": {{
            "value": 6619,
            "unit": "tCO2e",
            "source_text": "Total scope 1: 6,619 tCO2e"
        }}
    }}"""

    metric_descriptions = {
        "scope1": "Scope 1 (direct) emissions",
        "scope2": "Scope 2 (indirect) emissions from purchased electricity",
        "scope3": "Scope 3 (value chain) emissions",
        "energy": "Energy consumption metrics (direct, indirect, gross, net)",
        "renewable": "Renewable electricity consumption and sources",
        "emissions": "All emissions data (Scope 1, 2, and 3 totals)"
    }

    return base_prompt.format(
        metric_type=metric_descriptions.get(metric_type, metric_type),
        data=data
    )

def get_completion(prompt):
    """Get completion from OpenAI."""
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4",
        temperature=0
    )
    response = llm.invoke(prompt)
    return response.content

def extract_metric(df, metric_type):
    """Extract metrics from the DataFrame."""
    # Clean up column names
    df.columns = [str(col).strip() for col in df.columns]
    
    # Identify the description column (first column)
    description_col = df.columns[0]
    df[description_col] = df[description_col].fillna('').astype(str).str.strip()
    
    # Create a mapping of year columns
    year_mapping = {}
    for col in df.columns:
        for year in ['2024', '2023', '2022']:
            if year in str(col):
                year_mapping[year] = col
                break
    
    # If we haven't found the year columns in the headers, look for them in the data
    if not year_mapping:
        for col in df.columns[1:]:  # Skip the description column
            col_data = df[col].astype(str).str.strip()
            for year in ['2024', '2023', '2022']:
                if col_data.str.contains(year).any():
                    year_mapping[year] = col
    
    # Filter rows based on metric type
    if metric_type == 'scope1':
        relevant_rows = df[df[description_col].str.contains('Total scope 1', case=False, na=False)]
    elif metric_type == 'scope2':
        relevant_rows = df[df[description_col].str.contains('Total scope 2|Purchased energy', case=False, na=False)]
    elif metric_type == 'scope3':
        relevant_rows = df[df[description_col].str.contains('Total scope 3', case=False, na=False)]
    elif metric_type == 'emissions':
        relevant_rows = df[df[description_col].str.contains('Gross GHG emissions .Scope 1, 2 and 3. prior', case=False, na=False)]
    elif metric_type == 'renewable':
        # Get both renewable electricity purchases and RE project percentages
        renewable_rows = df[df[description_col].str.contains('Renewable electricity purchased', case=False, na=False)]
        re_solar_rows = df[df[description_col].str.contains('RE - Solar', case=False, na=False)]
        re_wind_rows = df[df[description_col].str.contains('RE - Wind', case=False, na=False)]
        carbon_offset_rows = df[df[description_col].str.contains('Carbon offsets retired', case=False, na=False)]
        
        # Combine all renewable data
        all_renewable_data = {}
        
        # Process renewable electricity purchases
        if not renewable_rows.empty:
            for year in ['2024', '2023', '2022']:
                if year in year_mapping:
                    try:
                        col = year_mapping[year]
                        value_str = str(renewable_rows[col].iloc[0]).strip()
                        value_str = ''.join(c for c in value_str if c.isdigit() or c in '.-')
                        if value_str:
                            value = abs(float(value_str))
                            if year not in all_renewable_data:
                                all_renewable_data[year] = {}
                            all_renewable_data[year]["renewable_electricity"] = {
                                "value": value,
                                "unit": "MWh",
                                "source_text": f"Renewable electricity purchased: {value} MWh"
                            }
                    except Exception as e:
                        print(f"Error extracting renewable electricity for {year}: {str(e)}")
        
        # Process carbon offsets
        if not carbon_offset_rows.empty:
            for year in ['2024', '2023', '2022']:
                if year in year_mapping:
                    try:
                        col = year_mapping[year]
                        value_str = str(carbon_offset_rows[col].iloc[0]).strip()
                        value_str = ''.join(c for c in value_str if c.isdigit() or c in '.-')
                        if value_str:
                            value = abs(float(value_str))
                            if year not in all_renewable_data:
                                all_renewable_data[year] = {}
                            all_renewable_data[year]["carbon_offsets"] = {
                                "value": value,
                                "unit": "tCO2-e",
                                "source_text": f"Carbon offsets retired: {value} tCO2-e"
                            }
                    except Exception as e:
                        print(f"Error extracting carbon offsets for {year}: {str(e)}")
        
        # Process RE project percentages
        for year in ['2024', '2023', '2022']:
            if year in year_mapping:
                try:
                    col = year_mapping[year]
                    
                    # Solar
                    if not re_solar_rows.empty:
                        solar_str = str(re_solar_rows[col].iloc[0]).strip()
                        if solar_str and solar_str != 'nan':
                            solar_value = float(solar_str)
                            if year not in all_renewable_data:
                                all_renewable_data[year] = {}
                            all_renewable_data[year]["solar_percentage"] = {
                                "value": solar_value,
                                "unit": "percentage",
                                "source_text": f"RE - Solar: {solar_value}%"
                            }
                            
                            # Calculate solar offset contribution if we have carbon offset data
                            if "carbon_offsets" in all_renewable_data.get(year, {}):
                                total_offsets = all_renewable_data[year]["carbon_offsets"]["value"]
                                solar_contribution = total_offsets * (solar_value / 100)
                                all_renewable_data[year]["solar_offset_contribution"] = {
                                    "value": solar_contribution,
                                    "unit": "tCO2-e",
                                    "source_text": f"Solar RE offset contribution: {solar_contribution} tCO2-e"
                                }
                    
                    # Wind
                    if not re_wind_rows.empty:
                        wind_str = str(re_wind_rows[col].iloc[0]).strip()
                        if wind_str and wind_str != 'nan':
                            wind_value = float(wind_str)
                            if year not in all_renewable_data:
                                all_renewable_data[year] = {}
                            all_renewable_data[year]["wind_percentage"] = {
                                "value": wind_value,
                                "unit": "percentage",
                                "source_text": f"RE - Wind: {wind_value}%"
                            }
                            
                            # Calculate wind offset contribution if we have carbon offset data
                            if "carbon_offsets" in all_renewable_data.get(year, {}):
                                total_offsets = all_renewable_data[year]["carbon_offsets"]["value"]
                                wind_contribution = total_offsets * (wind_value / 100)
                                all_renewable_data[year]["wind_offset_contribution"] = {
                                    "value": wind_contribution,
                                    "unit": "tCO2-e",
                                    "source_text": f"Wind RE offset contribution: {wind_contribution} tCO2-e"
                                }
                            
                except Exception as e:
                    print(f"Error extracting RE percentages for {year}: {str(e)}")
        
        return all_renewable_data
    elif metric_type == 'energy':
        # Find rows for different energy metrics
        purchased_energy_rows = df[df[description_col].str.contains('Purchased energy', case=False, na=False)]
        fuel_energy_rows = df[df[description_col].str.contains('Fuel and energy-related activities', case=False, na=False)]
        
        result = {}
        for year in ['2024', '2023', '2022']:
            if year in year_mapping:
                result[year] = {
                    "purchased_energy": {
                        "value": None,
                        "unit": "tCO2-e",
                        "source_text": None
                    },
                    "fuel_energy": {
                        "value": None,
                        "unit": "tCO2-e",
                        "source_text": None
                    }
                }
                
                try:
                    col = year_mapping[year]
                    
                    # Extract purchased energy
                    if not purchased_energy_rows.empty:
                        value_str = str(purchased_energy_rows[col].iloc[0]).strip()
                        if value_str and value_str != 'nan':
                            value = float(''.join(c for c in value_str if c.isdigit() or c in '.-'))
                            result[year]["purchased_energy"] = {
                                "value": value,
                                "unit": "tCO2-e",
                                "source_text": f"{purchased_energy_rows[description_col].iloc[0].strip()}: {value} tCO2-e"
                            }
                    
                    # Extract fuel and energy-related activities
                    if not fuel_energy_rows.empty:
                        value_str = str(fuel_energy_rows[col].iloc[0]).strip()
                        if value_str and value_str != 'nan':
                            value = float(''.join(c for c in value_str if c.isdigit() or c in '.-'))
                            result[year]["fuel_energy"] = {
                                "value": value,
                                "unit": "tCO2-e",
                                "source_text": f"{fuel_energy_rows[description_col].iloc[0].strip()}: {value} tCO2-e"
                            }
                            
                except Exception as e:
                    print(f"Error extracting energy metrics for {year}: {str(e)}")
                    
        return result
    else:
        relevant_rows = df[df[description_col].str.contains(metric_type, case=False, na=False)]
    
    if relevant_rows.empty and metric_type != 'renewable':
        print(f"No rows found for {metric_type}")
        return None
    
    # Print debug information
    print(f"\nRelevant data for {metric_type}:")
    if metric_type == 'renewable':
        if not renewable_rows.empty:
            print("\nRenewable electricity purchases:")
            print(renewable_rows.to_string())
        if not re_solar_rows.empty:
            print("\nSolar RE projects:")
            print(re_solar_rows.to_string())
        if not re_wind_rows.empty:
            print("\nWind RE projects:")
            print(re_wind_rows.to_string())
    else:
        print(relevant_rows.to_string())
    print("\nYear mapping found:", year_mapping)
    
    # Extract values for each year (for non-renewable metrics)
    if metric_type != 'renewable':
        result = {}
        for year in ['2024', '2023', '2022']:
            if year in year_mapping and not relevant_rows.empty:
                try:
                    col = year_mapping[year]
                    value_str = str(relevant_rows[col].iloc[0]).strip()
                    value_str = ''.join(c for c in value_str if c.isdigit() or c in '.-')
                    if value_str:
                        value = float(value_str)
                        result[year] = {
                            "value": value,
                            "unit": "tCO2-e",
                            "source_text": f"{relevant_rows[description_col].iloc[0].strip()}: {value} tCO2-e"
                        }
                    else:
                        result[year] = {
                            "value": None,
                            "unit": None,
                            "source_text": None
                        }
                except Exception as e:
                    print(f"Error extracting value for {year}: {str(e)}")
                    result[year] = {
                        "value": None,
                        "unit": None,
                        "source_text": None
                    }
            else:
                result[year] = {
                    "value": None,
                    "unit": None,
                    "source_text": None
                }
        return result

def find_emissions_sheet(file_path: str) -> str:
    """Find the most likely sheet containing emissions data."""
    try:
        xl = pd.ExcelFile(file_path)
        sheets = xl.sheet_names
        
        # Common variations of emissions sheet names
        emissions_keywords = [
            'ghg', 'emission', 'emissions', 'greenhouse', 'carbon', 'co2', 
            'climate', 'environmental', 'environment', 'scope'
        ]
        
        # First, try exact matches
        exact_matches = [
            'GHG Emissions', 'Emissions', 'GHG', 'Greenhouse Gas',
            'Carbon Emissions', 'Environmental Data'
        ]
        for sheet in sheets:
            if sheet in exact_matches:
                return sheet
        
        # Then try keyword matching
        for sheet in sheets:
            sheet_lower = sheet.lower()
            for keyword in emissions_keywords:
                if keyword in sheet_lower:
                    return sheet
        
        # If no matches, look for sheets with emissions data
        for sheet in sheets:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet)
                text = df.to_string().lower()
                if any(keyword in text for keyword in ['scope 1', 'scope1', 'scope 2', 'scope2', 'scope 3', 'scope3']):
                    return sheet
            except:
                continue
        
        # If still no match, return the first sheet as fallback
        return sheets[0]
    except Exception as e:
        print(f"Error finding emissions sheet: {str(e)}")
        return 'Sheet1'  # Default fallback

def main():
    parser = argparse.ArgumentParser(description='Extract specific metrics from sustainability report')
    parser.add_argument('--energy', action='store_true', help='Extract all energy metrics')
    parser.add_argument('--scope1', action='store_true', help='Extract Scope 1 emissions')
    parser.add_argument('--scope2', action='store_true', help='Extract Scope 2 emissions')
    parser.add_argument('--scope3', action='store_true', help='Extract Scope 3 emissions')
    parser.add_argument('--renewable', action='store_true', help='Extract renewable energy metrics')
    parser.add_argument('--emissions', action='store_true', help='Extract all emissions metrics')
    parser.add_argument('--file', type=str, default='2024-sustainability-data-pack.xlsx', help='Excel file to analyze')
    parser.add_argument('--sheet', type=str, help='Sheet name to analyze (optional, will auto-detect if not provided)')
    parser.add_argument('--debug', action='store_true', help='Show debug information')
    
    args = parser.parse_args()
    
    try:
        # Auto-detect the emissions sheet if not specified
        if not args.sheet:
            args.sheet = find_emissions_sheet(args.file)
            if args.debug:
                print(f"\nAuto-detected emissions sheet: {args.sheet}")
        
        # Read the Excel file
        df = pd.read_excel(args.file, sheet_name=args.sheet)
        
        # Print the data we're working with if in debug mode
        if args.debug:
            print("\nAll rows of data:")
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            print(df)
            pd.reset_option('display.max_rows')
            pd.reset_option('display.max_columns')
            pd.reset_option('display.width')
        
        # Clean the dataframe
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Fix column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Convert numeric columns to float, handling any non-numeric values
        for col in df.columns:
            try:
                if col not in ['Gross greenhouse gas emissions by activity1']:
                    df[col] = df[col].apply(lambda x: float(str(x).replace(',', '')) if pd.notna(x) and str(x).replace(',', '').replace('.', '').replace('-', '').isdigit() else x)
            except:
                continue
        
        # Clean text columns
        text_cols = ['Gross greenhouse gas emissions by activity1']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str).apply(lambda x: x.strip())
        
        results = {}
        
        if args.energy:
            results['energy'] = extract_metric(df, 'energy')
        if args.scope1:
            results['scope1'] = extract_metric(df, 'scope1')
        if args.scope2:
            results['scope2'] = extract_metric(df, 'scope2')
        if args.scope3:
            results['scope3'] = extract_metric(df, 'scope3')
        if args.renewable:
            results['renewable'] = extract_metric(df, 'renewable')
        if args.emissions:
            results['emissions'] = extract_metric(df, 'emissions')
            
        if not any(vars(args).values()):
            # If no arguments provided, show usage
            parser.print_help()
        else:
            # Create output directory if it doesn't exist
            os.makedirs('output/extracted', exist_ok=True)
            
            # Save results to JSON file
            output_file = 'output/extracted/excel_data.json'
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            if args.debug:
                print("\nFinal Results:")
                print(json.dumps(results, indent=2))
                print(f"\nResults saved to: {output_file}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()