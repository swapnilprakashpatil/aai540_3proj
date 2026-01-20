import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class CMSDataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dict = None
        self.df = None
        self.metadata = {}
        
    def load_data_dictionary(self, filename: str = "datadictionary.json") -> Dict:
        dict_path = self.data_dir / filename
        
        with open(dict_path, 'r') as f:
            self.data_dict = json.load(f)
        
        print(f"Loaded data dictionary with {len(self.data_dict['data']['fields'])} fields")
        return self.data_dict
    
    def load_dataset(self, filename: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        file_path = self.data_dir / filename
        
        print(f"Loading dataset: {filename}")
        
        # Load data
        if sample_size:
            self.df = pd.read_csv(file_path, nrows=sample_size, low_memory=False)
            print(f"Loaded sample of {sample_size:,} rows")
        else:
            self.df = pd.read_csv(file_path, low_memory=False)
            print(f"Loaded full dataset with {len(self.df):,} rows")
        
        # Store metadata
        self.metadata['shape'] = self.df.shape
        self.metadata['columns'] = self.df.columns.tolist()
        self.metadata['filename'] = filename
        
        return self.df
    
    def preprocess_data(self, date_columns: List[str], numeric_columns: List[str]) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("No data loaded. Call load_dataset() first.")
        
        print("\nPreprocessing data...")
        
        # Convert date columns
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                print(f"  Converted {col} to datetime")
        
        # Convert numeric columns
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                print(f"  Converted {col} to numeric")
        
        # Extract temporal features
        if 'Date_of_Payment' in self.df.columns:
            self.df['Payment_Year'] = self.df['Date_of_Payment'].dt.year
            self.df['Payment_Month'] = self.df['Date_of_Payment'].dt.month
            self.df['Payment_Quarter'] = self.df['Date_of_Payment'].dt.quarter
            self.df['Payment_DayOfWeek'] = self.df['Date_of_Payment'].dt.dayofweek
            print("  Extracted temporal features")
        
        return self.df
    
    def get_basic_stats(self) -> Dict:
        if self.df is None:
            raise ValueError("No data loaded. Call load_dataset() first.")
        
        stats = {
            'n_rows': len(self.df),
            'n_columns': len(self.df.columns),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': self.df.isnull().sum().sum(),
            'missing_percentage': (self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100,
            'duplicate_rows': self.df.duplicated().sum(),
            'numeric_columns': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.df.select_dtypes(include=['object']).columns.tolist(),
            'datetime_columns': self.df.select_dtypes(include=['datetime64']).columns.tolist()
        }
        
        return stats
    
    def get_missing_value_summary(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("No data loaded. Call load_dataset() first.")
        
        missing_summary = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum().values,
            'Missing_Percentage': (self.df.isnull().sum().values / len(self.df) * 100).round(2),
            'Data_Type': self.df.dtypes.values
        })
        
        missing_summary = missing_summary[missing_summary['Missing_Count'] > 0].sort_values(
            'Missing_Percentage', ascending=False
        ).reset_index(drop=True)
        
        return missing_summary
    
    def get_value_counts_summary(self, column: str, top_n: int = 10) -> pd.DataFrame:
        if self.df is None or column not in self.df.columns:
            raise ValueError(f"Column {column} not found in dataset")
        
        counts = self.df[column].value_counts().head(top_n)
        percentages = (counts / len(self.df) * 100).round(2)
        
        summary = pd.DataFrame({
            'Value': counts.index,
            'Count': counts.values,
            'Percentage': percentages.values
        })
        
        return summary
    
    def aggregate_by_recipient(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("No data loaded. Call load_dataset() first.")
        
        print("\nAggregating by recipient...")
        
        # Define aggregation based on recipient type
        agg_dict = {
            'Total_Amount_of_Payment_USDollars': ['sum', 'mean', 'median', 'std', 'min', 'max', 'count'],
            'Number_of_Payments_Included_in_Total_Amount': 'sum',
            'Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_ID': 'nunique',
            'Form_of_Payment_or_Transfer_of_Value': lambda x: x.mode()[0] if len(x.mode()) > 0 else None,
            'Nature_of_Payment_or_Transfer_of_Value': lambda x: x.mode()[0] if len(x.mode()) > 0 else None
        }
        
        # Group by appropriate identifier
        if 'Covered_Recipient_Profile_ID' in self.df.columns:
            group_cols = ['Covered_Recipient_Profile_ID', 'Covered_Recipient_Type']
            
            # Add specialty and location if available
            if 'Covered_Recipient_Primary_Type_1' in self.df.columns:
                group_cols.append('Covered_Recipient_Primary_Type_1')
            if 'Recipient_State' in self.df.columns:
                group_cols.append('Recipient_State')
            
            aggregated = self.df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
            
            # Flatten column names
            aggregated.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                                  for col in aggregated.columns.values]
            
            print(f"Aggregated to {len(aggregated):,} unique recipients")
            
            return aggregated
        else:
            print("WARNING: No recipient identifier found for aggregation")
            return None
    
    def get_data_quality_report(self) -> Dict:
        if self.df is None:
            raise ValueError("No data loaded. Call load_dataset() first.")
        
        report = {
            'completeness': {
                'total_cells': self.df.shape[0] * self.df.shape[1],
                'missing_cells': self.df.isnull().sum().sum(),
                'completeness_rate': (1 - self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100
            },
            'uniqueness': {
                'duplicate_rows': self.df.duplicated().sum(),
                'duplicate_percentage': (self.df.duplicated().sum() / len(self.df)) * 100
            },
            'validity': {},
            'consistency': {}
        }
        
        # Check amount validity
        if 'Total_Amount_of_Payment_USDollars' in self.df.columns:
            report['validity']['negative_payments'] = (self.df['Total_Amount_of_Payment_USDollars'] < 0).sum()
            report['validity']['zero_payments'] = (self.df['Total_Amount_of_Payment_USDollars'] == 0).sum()
        
        # Check date validity
        if 'Date_of_Payment' in self.df.columns:
            report['validity']['future_dates'] = (self.df['Date_of_Payment'] > pd.Timestamp.now()).sum()
        
        return report


class DataDictionaryHelper:
    def __init__(self, data_dict: Dict):
        self.data_dict = data_dict
        self.fields = data_dict['data']['fields']
    
    def get_field_description(self, field_name: str) -> str:
        for field in self.fields:
            if field['name'] == field_name:
                return field.get('description', 'No description available')
        return f"Field {field_name} not found in dictionary"
    
    def get_field_type(self, field_name: str) -> str:
        for field in self.fields:
            if field['name'] == field_name:
                return field.get('type', 'Unknown')
        return 'Unknown'
    
    def get_field_constraints(self, field_name: str) -> Dict:
        for field in self.fields:
            if field['name'] == field_name:
                return field.get('constraints', {})
        return {}
    
    def list_all_fields(self) -> List[str]:
        return [field['name'] for field in self.fields]
    
    def search_fields(self, keyword: str) -> List[Dict]:
        keyword = keyword.lower()
        results = []
        
        for field in self.fields:
            if (keyword in field['name'].lower() or 
                keyword in field.get('description', '').lower()):
                results.append({
                    'name': field['name'],
                    'description': field.get('description', '')[:100] + '...',
                    'type': field.get('type', 'Unknown')
                })
        
        return results
