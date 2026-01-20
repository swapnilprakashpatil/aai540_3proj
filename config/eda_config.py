"""
Configuration module for CMS Open Payments EDA
Contains all settings, constants, and configurations for analysis
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os


@dataclass
class DataConfig:
    """Data loading and processing configuration"""
    
    # File paths
    data_dir: str = "../data"
    light_dataset: str = "lightdataset.csv"
    full_dataset: str = "OP_DTL_GNRL_PGYR2024_P06302025_06162025.csv"
    data_dictionary: str = "datadictionary.json"
    
    # Data types
    date_columns: List[str] = field(default_factory=lambda: [
        'Date_of_Payment',
        'Payment_Publication_Date'
    ])
    
    numeric_columns: List[str] = field(default_factory=lambda: [
        'Total_Amount_of_Payment_USDollars',
        'Number_of_Payments_Included_in_Total_Amount',
        'Covered_Recipient_NPI'
    ])
    
    categorical_columns: List[str] = field(default_factory=lambda: [
        'Change_Type',
        'Covered_Recipient_Type',
        'Form_of_Payment_or_Transfer_of_Value',
        'Nature_of_Payment_or_Transfer_of_Value',
        'Recipient_State',
        'Covered_Recipient_Primary_Type_1',
        'Related_Product_Indicator',
        'Physician_Ownership_Indicator'
    ])
    
    # Missing value threshold
    missing_threshold: float = 0.5
    
    def get_full_path(self, filename: str) -> str:
        """Get full path for a data file"""
        return os.path.join(self.data_dir, filename)


@dataclass
class VisualizationConfig:
    """Visualization settings"""
    
    # Plot style
    style: str = "darkgrid"
    context: str = "notebook"
    palette: str = "viridis"
    
    # Figure sizes
    default_figsize: tuple = (12, 6)
    large_figsize: tuple = (16, 8)
    small_figsize: tuple = (10, 5)
    square_figsize: tuple = (10, 10)
    
    # Color schemes
    color_scheme_1: List[str] = field(default_factory=lambda: [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ])
    
    color_scheme_2: List[str] = field(default_factory=lambda: [
        '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
        '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd'
    ])
    
    # Plot limits
    max_categories: int = 20
    outlier_percentile: float = 99.5
    
    # 3D plot settings
    plot_3d_elevation: int = 20
    plot_3d_azimuth: int = 45


@dataclass
class AnalysisConfig:
    """Analysis parameters"""
    
    # Anomaly detection
    contamination_rate: float = 0.02  # Expected proportion of anomalies
    random_state: int = 42
    
    # Statistical thresholds
    correlation_threshold: float = 0.7
    significance_level: float = 0.05
    
    # Aggregation settings
    top_n_entities: int = 20
    min_payment_count: int = 5
    
    # Feature engineering
    log_transform_columns: List[str] = field(default_factory=lambda: [
        'Total_Amount_of_Payment_USDollars'
    ])


@dataclass
class EDAConfig:
    """Master configuration for EDA"""
    
    data: DataConfig = field(default_factory=DataConfig)
    viz: VisualizationConfig = field(default_factory=VisualizationConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    
    # Project metadata
    project_name: str = "CMS Open Payments Exploratory Data Analysis"
    version: str = "1.0.0"
    program_year: int = 2024


# Global configuration instance
CONFIG = EDAConfig()
