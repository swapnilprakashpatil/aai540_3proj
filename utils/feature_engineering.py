import pandas as pd
import numpy as np
from typing import List


class FeatureEngineer:
    @staticmethod
    def create_aggregated_features(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        print(f"Creating aggregated features grouped by: {', '.join(group_cols)}")
        
        agg_dict = {
            'Total_Amount_of_Payment_USDollars': ['sum', 'mean', 'median', 'std', 'min', 'max', 'count'],
            'Number_of_Payments_Included_in_Total_Amount': 'sum',
            'Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_ID': 'nunique',
            'Form_of_Payment_or_Transfer_of_Value': lambda x: x.nunique(),
            'Nature_of_Payment_or_Transfer_of_Value': lambda x: x.nunique()
        }
        
        aggregated = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
        
        aggregated.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                              for col in aggregated.columns.values]
        
        print(f"Created {len(aggregated):,} aggregated records with {len(aggregated.columns)} features")
        
        return aggregated
    
    @staticmethod
    def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if 'Total_Amount_of_Payment_USDollars_std' in df.columns and 'Total_Amount_of_Payment_USDollars_mean' in df.columns:
            df['amount_cv'] = df['Total_Amount_of_Payment_USDollars_std'] / (df['Total_Amount_of_Payment_USDollars_mean'] + 1)
        
        if 'Total_Amount_of_Payment_USDollars_sum' in df.columns:
            df['log_total_amount'] = np.log1p(df['Total_Amount_of_Payment_USDollars_sum'])
        
        if 'Total_Amount_of_Payment_USDollars_mean' in df.columns:
            df['log_mean_amount'] = np.log1p(df['Total_Amount_of_Payment_USDollars_mean'])
        
        if 'Form_of_Payment_or_Transfer_of_Value_<lambda>' in df.columns:
            df['payment_form_diversity'] = df['Form_of_Payment_or_Transfer_of_Value_<lambda>']
        
        if 'Nature_of_Payment_or_Transfer_of_Value_<lambda>' in df.columns:
            df['payment_nature_diversity'] = df['Nature_of_Payment_or_Transfer_of_Value_<lambda>']
        
        print(f"Created {sum(1 for col in df.columns if col.startswith(('amount_cv', 'log_', 'payment_')))} derived features")
        
        return df
