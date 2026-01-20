import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class PaymentVisualizer:
    def __init__(self, style: str = "darkgrid", palette: str = "viridis", context: str = "notebook"):
        sns.set_style(style)
        sns.set_context(context)
        sns.set_palette(palette)
        self.palette = palette
        
    def plot_missing_values(self, df: pd.DataFrame, top_n: int = 20, 
                           figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        # Calculate missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        
        # Filter and sort
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing_Count': missing.values,
            'Missing_Percentage': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
            'Missing_Percentage', ascending=False
        ).head(top_n)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.barh(missing_df['Column'], missing_df['Missing_Percentage'], 
                      color=sns.color_palette("rocket", len(missing_df)))
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                   f'{width:.1f}%', ha='left', va='center', fontsize=9)
        
        ax.set_xlabel('Missing Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Column Name', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Columns with Missing Values', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_payment_distribution(self, df: pd.DataFrame, amount_col: str = 'Total_Amount_of_Payment_USDollars',
                                 figsize: Tuple[int, int] = (16, 6)) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Remove NaN and zero values for better visualization
        data = df[amount_col].dropna()
        data_positive = data[data > 0]
        
        # Histogram
        axes[0].hist(data_positive, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Payment Amount ($)', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0].set_title('Payment Distribution (Linear Scale)', fontsize=12, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        # Log scale histogram
        axes[1].hist(data_positive, bins=50, color='coral', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Payment Amount ($)', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Frequency (log scale)', fontsize=11, fontweight='bold')
        axes[1].set_title('Payment Distribution (Log Scale)', fontsize=12, fontweight='bold')
        axes[1].set_yscale('log')
        axes[1].grid(alpha=0.3)
        
        # Box plot
        box_parts = axes[2].boxplot(data_positive, vert=True, patch_artist=True,
                                     boxprops=dict(facecolor='lightgreen', alpha=0.7),
                                     medianprops=dict(color='darkred', linewidth=2),
                                     whiskerprops=dict(color='black', linewidth=1.5),
                                     capprops=dict(color='black', linewidth=1.5))
        axes[2].set_ylabel('Payment Amount ($)', fontsize=11, fontweight='bold')
        axes[2].set_title('Payment Distribution (Box Plot)', fontsize=12, fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)
        axes[2].set_yscale('log')
        
        plt.suptitle('Payment Amount Distribution Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def plot_category_distribution(self, df: pd.DataFrame, column: str, 
                                   top_n: int = 15, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        # Get value counts
        value_counts = df[column].value_counts().head(top_n)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Bar plot
        colors = sns.color_palette("husl", len(value_counts))
        bars = axes[0].barh(range(len(value_counts)), value_counts.values, color=colors)
        axes[0].set_yticks(range(len(value_counts)))
        axes[0].set_yticklabels([str(x)[:30] + '...' if len(str(x)) > 30 else str(x) 
                                 for x in value_counts.index], fontsize=9)
        axes[0].set_xlabel('Count', fontsize=11, fontweight='bold')
        axes[0].set_title(f'Top {top_n} {column}', fontsize=12, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            axes[0].text(width, bar.get_y() + bar.get_height()/2, 
                        f'{int(width):,}', ha='left', va='center', fontsize=8)
        
        # Pie chart
        axes[1].pie(value_counts.values, labels=[str(x)[:20] + '...' if len(str(x)) > 20 else str(x) 
                                                 for x in value_counts.index],
                   autopct='%1.1f%%', startangle=90, colors=colors)
        axes[1].set_title(f'Proportion of {column}', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_temporal_trends(self, df: pd.DataFrame, date_col: str = 'Date_of_Payment',
                           amount_col: str = 'Total_Amount_of_Payment_USDollars',
                           figsize: Tuple[int, int] = (16, 10)) -> plt.Figure:
        # Prepare data
        df_temp = df[[date_col, amount_col]].dropna().copy()
        df_temp['Year_Month'] = df_temp[date_col].dt.to_period('M')
        
        # Aggregate by month
        monthly_stats = df_temp.groupby('Year_Month')[amount_col].agg(['sum', 'mean', 'count']).reset_index()
        monthly_stats['Year_Month'] = monthly_stats['Year_Month'].dt.to_timestamp()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Total payments over time
        axes[0, 0].plot(monthly_stats['Year_Month'], monthly_stats['sum']/1e6, 
                       marker='o', linewidth=2, markersize=6, color='steelblue')
        axes[0, 0].set_xlabel('Month', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Total Amount ($ Millions)', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Total Payments Over Time', fontsize=12, fontweight='bold')
        axes[0, 0].grid(alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Average payment over time
        axes[0, 1].plot(monthly_stats['Year_Month'], monthly_stats['mean'], 
                       marker='s', linewidth=2, markersize=6, color='coral')
        axes[0, 1].set_xlabel('Month', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Average Payment ($)', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Average Payment Over Time', fontsize=12, fontweight='bold')
        axes[0, 1].grid(alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Payment count over time
        axes[1, 0].bar(monthly_stats['Year_Month'], monthly_stats['count'], 
                      color='lightgreen', edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Month', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Number of Payments', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Payment Count Over Time', fontsize=12, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Day of week distribution
        df_temp['DayOfWeek'] = df_temp[date_col].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = df_temp['DayOfWeek'].value_counts().reindex(day_order)
        
        axes[1, 1].bar(day_counts.index, day_counts.values, 
                      color=sns.color_palette("viridis", 7), edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Day of Week', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('Number of Payments', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Payments by Day of Week', fontsize=12, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Temporal Trends Analysis', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, numeric_cols: List[str],
                                figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        # Select numeric columns
        df_numeric = df[numeric_cols].select_dtypes(include=[np.number])
        
        # Calculate correlation
        corr = df_numeric.corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   ax=ax, vmin=-1, vmax=1)
        
        ax.set_title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_scatter_3d(self, df: pd.DataFrame, x_col: str, y_col: str, z_col: str,
                                     color_col: Optional[str] = None, title: str = "3D Scatter Plot",
                                     size_col: Optional[str] = None) -> go.Figure:
        fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col,
                           color=color_col if color_col else None,
                           size=size_col if size_col else None,
                           title=title,
                           labels={x_col: x_col.replace('_', ' '),
                                  y_col: y_col.replace('_', ' '),
                                  z_col: z_col.replace('_', ' ')},
                           opacity=0.7)
        
        fig.update_layout(
            scene=dict(
                xaxis_title=x_col.replace('_', ' '),
                yaxis_title=y_col.replace('_', ' '),
                zaxis_title=z_col.replace('_', ' ')
            ),
            font=dict(size=12),
            title_font_size=16
        )
        
        return fig
    
    def create_sunburst_chart(self, df: pd.DataFrame, path_cols: List[str], 
                             value_col: str, title: str = "Hierarchical Breakdown") -> go.Figure:
        fig = px.sunburst(df, path=path_cols, values=value_col, title=title)
        
        fig.update_layout(
            font=dict(size=12),
            title_font_size=16
        )
        
        return fig
    
    def create_parallel_coordinates(self, df: pd.DataFrame, dimensions: List[str],
                                   color_col: str, title: str = "Parallel Coordinates") -> go.Figure:
        fig = px.parallel_coordinates(df, dimensions=dimensions, color=color_col,
                                     title=title)
        
        fig.update_layout(
            font=dict(size=11),
            title_font_size=16
        )
        
        return fig
    
    def plot_geographic_distribution(self, df: pd.DataFrame, state_col: str = 'Recipient_State',
                                    value_col: str = 'Total_Amount_of_Payment_USDollars',
                                    figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
        # Aggregate by state
        state_agg = df.groupby(state_col)[value_col].agg(['sum', 'mean', 'count']).reset_index()
        state_agg = state_agg.sort_values('sum', ascending=False).head(20)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Total amount by state
        colors1 = sns.color_palette("rocket", len(state_agg))
        bars1 = axes[0].barh(state_agg[state_col], state_agg['sum']/1e6, color=colors1)
        axes[0].set_xlabel('Total Amount ($ Millions)', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('State', fontsize=11, fontweight='bold')
        axes[0].set_title('Top 20 States by Total Payment Amount', fontsize=12, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Payment count by state
        colors2 = sns.color_palette("mako", len(state_agg))
        bars2 = axes[1].barh(state_agg[state_col], state_agg['count'], color=colors2)
        axes[1].set_xlabel('Number of Payments', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('State', fontsize=11, fontweight='bold')
        axes[1].set_title('Top 20 States by Payment Count', fontsize=12, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_outliers_analysis(self, df_aggregated: pd.DataFrame, amounts: pd.Series, 
                              lower_bound: float, upper_bound: float, z_threshold: float = 3,
                              figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        from scipy import stats
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        axes[0, 0].boxplot(amounts, vert=True, patch_artist=True,
                           boxprops=dict(facecolor='lightblue', alpha=0.7),
                           medianprops=dict(color='darkred', linewidth=2))
        axes[0, 0].set_ylabel('Payment Amount ($)', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Box Plot - IQR Method', fontsize=12, fontweight='bold')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        axes[0, 1].hist(amounts, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(upper_bound, color='red', linestyle='--', linewidth=2, 
                          label=f'Upper Bound: ${upper_bound:,.0f}')
        axes[0, 1].set_xlabel('Payment Amount ($)', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Distribution with IQR Threshold', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(alpha=0.3)
        
        z_scores_all = stats.zscore(df_aggregated['Total_Amount_of_Payment_USDollars_sum'].fillna(0))
        axes[1, 0].hist(z_scores_all, bins=50, color='coral', edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(z_threshold, color='red', linestyle='--', linewidth=2, 
                          label=f'Threshold: {z_threshold}')
        axes[1, 0].axvline(-z_threshold, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Z-Score', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Z-Score Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        outlier_counts = pd.Series({
            'IQR': df_aggregated['is_outlier_iqr'].sum(),
            'Z-Score': df_aggregated['is_outlier_zscore'].sum(),
            'Percentile (99th)': df_aggregated['is_outlier_percentile'].sum()
        })
        axes[1, 1].bar(outlier_counts.index, outlier_counts.values, 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black', alpha=0.7)
        axes[1, 1].set_ylabel('Number of Outliers', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Outlier Detection Methods Comparison', fontsize=12, fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=15)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_choropleth_map(self, df: pd.DataFrame, state_col: str = 'Recipient_State',
                             amount_col: str = 'Total_Amount_of_Payment_USDollars',
                             title: str = 'Total Payment Amount by State') -> go.Figure:
        state_agg = df.groupby(state_col).agg({
            amount_col: ['sum', 'mean', 'count']
        }).reset_index()
        
        state_agg.columns = ['State', 'Total_Amount', 'Avg_Amount', 'Payment_Count']
        
        fig = px.choropleth(state_agg,
                            locations='State',
                            locationmode='USA-states',
                            color='Total_Amount',
                            scope='usa',
                            color_continuous_scale='Viridis',
                            title=title,
                            labels={'Total_Amount': 'Total Amount ($)'},
                            hover_data={
                                'State': True,
                                'Total_Amount': ':$,.2f',
                                'Avg_Amount': ':$,.2f',
                                'Payment_Count': ':,'
                            })
        
        fig.update_layout(
            geo=dict(
                bgcolor='rgba(240,240,240,1)',
                lakecolor='rgb(200,200,255)',
                showlakes=True
            ),
            font=dict(size=12),
            title_font_size=16,
            height=600
        )
        
        return fig
    
    def create_scatter_geo_map(self, df: pd.DataFrame, state_col: str = 'Recipient_State',
                              amount_col: str = 'Total_Amount_of_Payment_USDollars',
                              title: str = 'Payment Distribution Across US States') -> go.Figure:
        state_metrics = df.groupby(state_col).agg({
            amount_col: ['sum', 'mean', 'median', 'count']
        }).reset_index()
        
        state_metrics.columns = ['State', 'Total_Payment', 'Avg_Payment', 'Median_Payment', 'Payment_Count']
        
        fig = px.scatter_geo(state_metrics,
                             locations='State',
                             locationmode='USA-states',
                             size='Total_Payment',
                             color='Avg_Payment',
                             hover_name='State',
                             hover_data={
                                 'State': False,
                                 'Total_Payment': ':$,.0f',
                                 'Avg_Payment': ':$,.2f',
                                 'Median_Payment': ':$,.2f',
                                 'Payment_Count': ':,'
                             },
                             size_max=50,
                             color_continuous_scale='Plasma',
                             scope='usa',
                             title=title + '<br><sub>Bubble size = Total payments, Color = Average payment amount</sub>')
        
        fig.update_geos(
            showland=True,
            landcolor='rgb(243, 243, 243)',
            coastlinecolor='rgb(204, 204, 204)',
            showlakes=True,
            lakecolor='rgb(200, 200, 255)',
            showcountries=True,
            countrycolor='rgb(204, 204, 204)',
            projection_type='albers usa'
        )
        
        fig.update_layout(
            height=600,
            font=dict(size=12),
            title_font_size=16,
            coloraxis_colorbar=dict(
                title="Avg Payment ($)",
                thickness=15,
                len=0.7
            )
        )
        
        return fig
    
    def plot_risk_heatmap(self, df_aggregated: pd.DataFrame, 
                         specialty_col: str = 'Covered_Recipient_Primary_Type_1',
                         state_col: str = 'Recipient_State',
                         risk_col: str = 'risk_score',
                         top_specialties: int = 15,
                         top_states: int = 20,
                         figsize: Tuple[int, int] = (16, 10)) -> plt.Figure:
        top_spec = df_aggregated[specialty_col].value_counts().head(top_specialties).index
        top_st = df_aggregated[state_col].value_counts().head(top_states).index
        
        df_heatmap = df_aggregated[
            df_aggregated[specialty_col].isin(top_spec) &
            df_aggregated[state_col].isin(top_st)
        ]
        
        heatmap_data = df_heatmap.pivot_table(
            values=risk_col,
            index=specialty_col,
            columns=state_col,
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(heatmap_data, annot=False, fmt='.1f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Average Risk Score'}, ax=ax)
        ax.set_xlabel('State', fontsize=12, fontweight='bold')
        ax.set_ylabel('Specialty', fontsize=12, fontweight='bold')
        ax.set_title('Average Risk Score by Specialty and State', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        return fig
    
    def plot_quarterly_trends(self, df: pd.DataFrame, quarter_col: str = 'Payment_Quarter',
                             amount_col: str = 'Total_Amount_of_Payment_USDollars',
                             figsize: Tuple[int, int] = (16, 6)) -> plt.Figure:
        quarterly_stats = df.groupby(quarter_col)[amount_col].agg([
            'count', 'sum', 'mean', 'median'
        ]).round(2)
        
        quarterly_stats.columns = ['Count', 'Total ($)', 'Mean ($)', 'Median ($)']
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        quarterly_stats['Count'].plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
        axes[0].set_xlabel('Quarter', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Payment Count', fontsize=11, fontweight='bold')
        axes[0].set_title('Payments by Quarter', fontsize=12, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].tick_params(axis='x', rotation=0)
        
        (quarterly_stats['Total ($)'] / 1e6).plot(kind='bar', ax=axes[1], color='coral', edgecolor='black')
        axes[1].set_xlabel('Quarter', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Total Amount ($ Millions)', fontsize=11, fontweight='bold')
        axes[1].set_title('Total Payment Amount by Quarter', fontsize=12, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        return fig
    
    def plot_bivariate_comparison(self, df: pd.DataFrame, group_col: str, amount_col: str,
                                 figsize: Tuple[int, int] = (16, 6)) -> plt.Figure:
        df_plot = df[df[amount_col] > 0].copy()
        df_plot[amount_col] = np.log1p(df_plot[amount_col])
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        sns.boxplot(data=df_plot, y=group_col, 
                   x=amount_col, ax=axes[0], palette='Set2')
        axes[0].set_xlabel(f'Log({amount_col})', fontsize=11, fontweight='bold')
        axes[0].set_ylabel(group_col, fontsize=11, fontweight='bold')
        axes[0].set_title(f'Box Plot: {group_col} vs Amount', fontsize=12, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        sns.violinplot(data=df_plot, y=group_col,
                      x=amount_col, ax=axes[1], palette='Set2', inner='box')
        axes[1].set_xlabel(f'Log({amount_col})', fontsize=11, fontweight='bold')
        axes[1].set_ylabel(group_col, fontsize=11, fontweight='bold')
        axes[1].set_title(f'Violin Plot: {group_col} vs Amount', fontsize=12, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_payment_nature_by_total(self, df: pd.DataFrame, 
                                    nature_col: str = 'Nature_of_Payment_or_Transfer_of_Value',
                                    amount_col: str = 'Total_Amount_of_Payment_USDollars',
                                    top_n: int = 15,
                                    figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot top payment natures by total amount as a horizontal bar chart.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing payment data
        nature_col : str
            Column name containing nature of payment categories
        amount_col : str
            Column name containing payment amounts
        top_n : int
            Number of top categories to display
        figsize : Tuple[int, int]
            Figure size (width, height)
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        nature_stats = df.groupby(nature_col)[amount_col].agg([
            'count', 'sum', 'mean', 'median'
        ]).round(2)
        
        nature_stats.columns = ['Count', 'Total ($)', 'Mean ($)', 'Median ($)']
        nature_stats = nature_stats.sort_values('Total ($)', ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        nature_stats['Total ($)'].plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
        ax.set_xlabel('Total Amount ($)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Nature of Payment', fontsize=12, fontweight='bold')
        ax.set_title(f'TOP {top_n} Payment Natures by Total Amount', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(nature_stats['Total ($)']):
            ax.text(v, i, f' ${v:,.0f}', va='center', fontsize=9)
        
        plt.tight_layout()
        return fig
