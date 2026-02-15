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
                           figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:

        # Calculate missing values
        missing_stats = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': df.isnull().sum().values,
            'Missing_Percent': (df.isnull().sum().values / len(df) * 100)
        })
        
        missing_stats = missing_stats[missing_stats['Missing_Count'] > 0].sort_values(
            'Missing_Percent', ascending=False
        )
        
        if len(missing_stats) == 0:
            print("No missing values detected")
            return None
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        top_missing = missing_stats.head(top_n)
        
        # Color code based on severity
        colors = ['#e74c3c' if x > 50 else '#f39c12' if x > 20 else '#3498db' 
                  for x in top_missing['Missing_Percent']]
        
        ax.barh(range(len(top_missing)), top_missing['Missing_Percent'], 
                color=colors, edgecolor='black', alpha=0.7)
        ax.set_yticks(range(len(top_missing)))
        ax.set_yticklabels(top_missing['Column'])
        ax.set_xlabel('Missing Values (%)', fontsize=11, fontweight='bold')
        ax.set_title('Top 20 Columns by Missing Values', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add percentage labels
        for i, (idx, row) in enumerate(top_missing.iterrows()):
            ax.text(row['Missing_Percent'] + 1, i, f"{row['Missing_Percent']:.1f}%", 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_payment_distribution(self, df: pd.DataFrame, amount_col: str = 'total_amount_of_payment_usdollars',
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
    
    def plot_temporal_trends(self, df: pd.DataFrame, date_col: str = 'date_of_payment',
                           amount_col: str = 'total_amount_of_payment_usdollars',
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

    def plot_geographic_distribution(self, df: pd.DataFrame, 
                                state_col: str = 'recipient_state',
                                payment_col: str = None,
                                figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:

        fig = plt.figure(figsize=figsize)
        
        if payment_col and payment_col in df.columns:
            # Create choropleth based on payment amounts
            state_totals = df.groupby(state_col)[payment_col].sum()
        else:
            # Create choropleth based on count
            state_totals = df.groupby(state_col).size()
        
        # Rest of the visualization code...
        
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
        
        z_scores_all = stats.zscore(df_aggregated['total_amount'].fillna(0))
        axes[1, 0].hist(z_scores_all, bins=50, color='coral', edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(z_threshold, color='red', linestyle='--', linewidth=2, 
                        label=f'Threshold: {z_threshold}')
        axes[1, 0].axvline(-z_threshold, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Z-Score', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Z-Score Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Calculate outliers using different methods
        outlier_counts = {}
        if 'is_outlier_iqr' in df_aggregated.columns:
            outlier_counts['IQR'] = df_aggregated['is_outlier_iqr'].sum()
        if 'is_outlier_zscore' in df_aggregated.columns:
            outlier_counts['Z-Score'] = df_aggregated['is_outlier_zscore'].sum()
        if 'is_outlier_percentile' in df_aggregated.columns:
            outlier_counts['Percentile (99th)'] = df_aggregated['is_outlier_percentile'].sum()
        
        if outlier_counts:
            outlier_counts = pd.Series(outlier_counts)
        else:
            # If no outlier columns exist, calculate them
            z_scores = np.abs(stats.zscore(df_aggregated['total_amount'].fillna(0)))
            outlier_counts = pd.Series({
                'IQR': ((df_aggregated['total_amount'] < lower_bound) | 
                        (df_aggregated['total_amount'] > upper_bound)).sum(),
                'Z-Score': (z_scores > z_threshold).sum()
            })
        
        axes[1, 1].bar(outlier_counts.index, outlier_counts.values, 
                    color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black', alpha=0.7)
        axes[1, 1].set_ylabel('Number of Outliers', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Outlier Detection Methods Comparison', fontsize=12, fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=15)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig

    def create_choropleth_map(self, df, state_col, amount_col, title):

        # Create state-level aggregations
        state_agg = df.groupby(state_col).agg({
            amount_col: ['sum', 'mean', 'count']
        }).reset_index()
        
        # Rename columns
        state_agg.columns = ['state', 'total_amount', 'avg_amount', 'payment_count']
        
        # Create choropleth map
        fig = px.choropleth(
            state_agg,
            locations='state',  # Changed from 'State' to 'state'
            locationmode='USA-states',
            color='total_amount',
            scope='usa',
            color_continuous_scale='Viridis',
            title=title,
            labels={
                'total_amount': 'Total Amount ($)',
                'state': 'State',
                'avg_amount': 'Average Amount ($)',
                'payment_count': 'Number of Payments'
            },
            hover_data={
                'state': True,
                'total_amount': ':$,.2f',
                'avg_amount': ':$,.2f',
                'payment_count': ':,'
            }
        )
        
        # Update layout
        fig.update_layout(
            geo=dict(
                bgcolor='rgba(240,240,240,1)',
                lakecolor='rgb(255, 255, 255)',
                showlakes=True,
                showland=True,
                landcolor='rgb(242, 242, 242)',
                showcoastlines=True,
                coastlinecolor='rgb(180, 180, 180)',
                showstates=True,
                statecolor='rgb(180, 180, 180)'
            ),
            width=1000,
            height=600
        )
        
        return fig
        
    def create_scatter_geo_map(self, df: pd.DataFrame, state_col: str = 'recipient_state',
                            amount_col: str = 'total_amount_of_payment_usdollars',
                            title: str = 'Payment Distribution Across US States') -> go.Figure:
        state_metrics = df.groupby(state_col).agg({
            amount_col: ['sum', 'mean', 'median', 'count']
        }).reset_index()
        
        state_metrics.columns = ['state', 'total_payment', 'avg_payment', 'median_payment', 'payment_count']
        
        fig = px.scatter_geo(state_metrics,
                            locations='State',
                            locationmode='USA-states',
                            size='total_payment',
                            color='avg_payment',
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
                        specialty_col: str = 'covered_recipient_primary_type_1',
                        state_col: str = 'recipient_state',
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
    
    def plot_quarterly_trends(self, df: pd.DataFrame, quarter_col: str = 'payment_quarter',
                            amount_col: str = 'total_amount_of_payment_usdollars',
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
                                    nature_col: str = 'nature_of_payment_or_transfer_of_value',
                                    amount_col: str = 'total_amount_of_payment_usdollars',
                                    top_n: int = 15,
                                    figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:

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
    
    def plot_payment_distribution_detailed(self, df: pd.DataFrame, 
                                        payment_col: str = 'total_amount_of_payment_usdollars',
                                        figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Histogram
        axes[0, 0].hist(df[payment_col].dropna(), bins=50, color='steelblue', 
                        edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Payment Amount ($)', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Payment Amount Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].grid(alpha=0.3)
        
        # Log-scale histogram
        log_payments = np.log10(df[payment_col][df[payment_col] > 0])
        axes[0, 1].hist(log_payments, bins=50, color='coral', 
                        edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Log10(Payment Amount)', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Payment Amount Distribution (Log Scale)', fontsize=12, fontweight='bold')
        axes[0, 1].grid(alpha=0.3)
        
        # Box plot
        axes[1, 0].boxplot(df[payment_col].dropna(), vert=True, patch_artist=True,
                        boxprops=dict(facecolor='lightgreen', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
        axes[1, 0].set_ylabel('Payment Amount ($)', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Payment Amount Box Plot', fontsize=12, fontweight='bold')
        axes[1, 0].grid(alpha=0.3)
        
        # Violin plot
        parts = axes[1, 1].violinplot([df[payment_col].dropna()], vert=True, 
                                    showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('plum')
            pc.set_alpha(0.7)
        axes[1, 1].set_ylabel('Payment Amount ($)', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Payment Amount Violin Plot', fontsize=12, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_categorical_distributions(self, df: pd.DataFrame, categorical_cols: List[str],
                                    figsize: Tuple[int, int] = (16, None)) -> plt.Figure:

        available_cols = [col for col in categorical_cols if col in df.columns]
        
        if len(available_cols) == 0:
            return None
            
        n_cols = min(len(available_cols), 2)
        n_rows = (len(available_cols) + 1) // 2
        
        if figsize[1] is None:
            figsize = (figsize[0], 6 * n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows * n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, col in enumerate(available_cols):
            value_counts = df[col].value_counts().head(15)
            
            axes[idx].barh(range(len(value_counts)), value_counts.values,
                        color=sns.color_palette('viridis', len(value_counts)),
                        edgecolor='black', alpha=0.7)
            axes[idx].set_yticks(range(len(value_counts)))
            axes[idx].set_yticklabels(value_counts.index, fontsize=9)
            axes[idx].set_xlabel('Count', fontsize=11, fontweight='bold')
            axes[idx].set_title(f'{col}\n(Top 15)', fontsize=12, fontweight='bold')
            axes[idx].grid(axis='x', alpha=0.3)
        
        # Hide extra subplots
        for idx in range(len(available_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_payment_by_recipient_type(self, df: pd.DataFrame, 
                                    recipient_type_col: str = 'covered_recipient_type',
                                    payment_col: str = 'total_amount_of_payment_usdollars',
                                    figsize: Tuple[int, int] = (16, 6)) -> plt.Figure:

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Box plot
        df.boxplot(column=payment_col, by=recipient_type_col, ax=axes[0],
                patch_artist=True, grid=True)
        axes[0].set_xlabel('Recipient Type', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Payment Amount ($)', fontsize=11, fontweight='bold')
        axes[0].set_title('Payment Distribution by Recipient Type', fontsize=12, fontweight='bold')
        plt.sca(axes[0])
        plt.xticks(rotation=45, ha='right')
        
        # Bar plot - total amounts
        total_by_type = df.groupby(recipient_type_col)[payment_col].sum().sort_values(ascending=False)
        colors = sns.color_palette('rocket', len(total_by_type))
        axes[1].bar(range(len(total_by_type)), total_by_type.values,
                    color=colors, edgecolor='black', alpha=0.7)
        axes[1].set_xticks(range(len(total_by_type)))
        axes[1].set_xticklabels(total_by_type.index, rotation=45, ha='right')
        axes[1].set_xlabel('Recipient Type', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Total Payment Amount ($)', fontsize=11, fontweight='bold')
        axes[1].set_title('Total Payments by Recipient Type', fontsize=12, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_monthly_trends(self, df: pd.DataFrame, 
                        payment_col: str = 'total_amount_of_payment_usdollars',
                        month_col: str = 'payment_month',
                        figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:

        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Monthly payment count
        monthly_counts = df.groupby(month_col).size()
        axes[0].plot(monthly_counts.index, monthly_counts.values, 
                    marker='o', linewidth=2, markersize=8, color='steelblue')
        axes[0].set_xlabel('Month', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Number of Payments', fontsize=11, fontweight='bold')
        axes[0].set_title('Monthly Payment Count', fontsize=12, fontweight='bold')
        axes[0].grid(alpha=0.3)
        axes[0].set_xticks(range(1, 13))
        
        # Monthly payment total
        monthly_totals = df.groupby(month_col)[payment_col].sum()
        axes[1].bar(monthly_totals.index, monthly_totals.values,
                    color=sns.color_palette('viridis', 12), edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Month', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Total Payment Amount ($)', fontsize=11, fontweight='bold')
        axes[1].set_title('Monthly Total Payment Amount', fontsize=12, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].set_xticks(range(1, 13))
        
        plt.tight_layout()
        return fig
    
    def plot_quarterly_comparison(self, df: pd.DataFrame,
                                payment_col: str = 'total_amount_of_payment_usdollars',
                                quarter_col: str = 'payment_quarter',
                                figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:

        quarterly_stats = df.groupby(quarter_col)[payment_col].agg(['sum']).round(2)
        quarterly_stats.columns = ['Total ($)']
        
        fig, ax = plt.subplots(figsize=figsize)
        quarterly_stats['Total ($)'].plot(kind='bar', ax=ax, 
                                        color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'],
                                        edgecolor='black', alpha=0.7)
        ax.set_xlabel('Quarter', fontsize=11, fontweight='bold')
        ax.set_ylabel('Total Payment Amount ($)', fontsize=11, fontweight='bold')
        ax.set_title('Quarterly Total Payment Amount', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=0)
        plt.tight_layout()
        return fig
    
    def plot_state_comparison(self, df: pd.DataFrame,
                            state_col: str = 'recipient_state',
                            payment_col: str = 'total_amount_of_payment_usdollars',
                            top_n: int = 20,
                            figsize: Tuple[int, int] = (14, 12)) -> plt.Figure:

        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Top states by count
        top_states_count = df[state_col].value_counts().head(top_n)
        axes[0].barh(range(len(top_states_count)), top_states_count.values,
                    color=sns.color_palette('rocket', len(top_states_count)),
                    edgecolor='black', alpha=0.7)
        axes[0].set_yticks(range(len(top_states_count)))
        axes[0].set_yticklabels(top_states_count.index)
        axes[0].set_xlabel('Number of Payments', fontsize=11, fontweight='bold')
        axes[0].set_title(f'Top {top_n} States by Payment Count', fontsize=12, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Top states by total amount
        top_states_amount = df.groupby(state_col)[payment_col].sum().sort_values(ascending=False).head(top_n)
        axes[1].barh(range(len(top_states_amount)), top_states_amount.values,
                    color=sns.color_palette('mako', len(top_states_amount)),
                    edgecolor='black', alpha=0.7)
        axes[1].set_yticks(range(len(top_states_amount)))
        axes[1].set_yticklabels(top_states_amount.index)
        axes[1].set_xlabel('Total Payment Amount ($)', fontsize=11, fontweight='bold')
        axes[1].set_title(f'Top {top_n} States by Total Payment Amount', fontsize=12, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_scatter(self, df: pd.DataFrame, 
                                    x_col: str, y_col: str,
                                    color_col: Optional[str] = None,
                                    size_col: Optional[str] = None,
                                    hover_data: Optional[List[str]] = None,
                                    title: str = 'Interactive Scatter Plot',
                                    figsize: Tuple[int, int] = (12, 8)) -> go.Figure:

            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_col,
                size=size_col,
                hover_data=hover_data,
                title=title,
                labels={x_col: x_col.replace('_', ' '), y_col: y_col.replace('_', ' ')},
                height=figsize[1] * 100
            )
            
            fig.update_layout(
                font=dict(size=12),
                title_font_size=16
            )
            
            return fig


class ModelVisualizer:
    """Visualization utilities for anomaly detection models"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_training_history(self, history: Dict, figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
        """Plot training history for neural network models (e.g., Autoencoder)"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Loss plot
        axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Loss (MSE)', fontsize=11, fontweight='bold')
        axes[0].set_title('Model Training History', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # MAE plot (if available)
        if 'mae' in history:
            axes[1].plot(history['mae'], label='Training MAE', linewidth=2)
            if 'val_mae' in history:
                axes[1].plot(history['val_mae'], label='Validation MAE', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
            axes[1].set_ylabel('MAE', fontsize=11, fontweight='bold')
            axes[1].set_title('Model Training MAE', fontsize=12, fontweight='bold')
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_anomaly_scores(self, train_scores: np.ndarray, test_scores: np.ndarray,
                           threshold: float, model_name: str = 'Model',
                           figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Plot anomaly score distributions and comparisons"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Combined histogram
        ax1 = axes[0, 0]
        ax1.hist(train_scores, bins=50, alpha=0.7, label='Train', color='blue', edgecolor='black')
        ax1.hist(test_scores, bins=50, alpha=0.7, label='Test', color='red', edgecolor='black')
        ax1.axvline(threshold, color='green', linestyle='--', linewidth=2.5, label='Threshold')
        ax1.set_xlabel('Anomaly Score', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title(f'{model_name} Score Distribution', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # All scores histogram
        all_scores = np.concatenate([train_scores, test_scores])
        ax2 = axes[0, 1]
        ax2.hist(all_scores, bins=100, alpha=0.8, color='purple', edgecolor='black')
        ax2.axvline(threshold, color='red', linestyle='--', linewidth=2.5, label='Threshold')
        ax2.axvline(all_scores.mean(), color='orange', linestyle=':', linewidth=2.5, label='Mean')
        ax2.set_xlabel('Anomaly Score', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title('Overall Score Distribution', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Percentile distribution
        ax3 = axes[1, 0]
        percentiles = np.arange(1, 101)
        ax3.plot(percentiles, np.percentile(all_scores, percentiles), 
                linewidth=2.5, color='darkblue', marker='o', markersize=3)
        ax3.axhline(y=threshold, color='red', linestyle='--', linewidth=2.5, 
                   label=f'Threshold: {threshold:.6f}')
        ax3.fill_between(percentiles, 0, np.percentile(all_scores, percentiles), 
                        alpha=0.2, color='blue')
        ax3.set_xlabel('Percentile', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Anomaly Score', fontsize=11, fontweight='bold')
        ax3.set_title('Score Percentile Distribution', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Box plot comparison
        ax4 = axes[1, 1]
        data_to_plot = [train_scores, test_scores]
        bp = ax4.boxplot(data_to_plot, labels=['Train', 'Test'], 
                        patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax4.axhline(y=threshold, color='green', linestyle='--', linewidth=2, label='Threshold')
        ax4.set_ylabel('Anomaly Score', fontsize=11, fontweight='bold')
        ax4.set_title('Score Distribution Comparison', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_anomaly_comparison(self, normal_df: pd.DataFrame, anomaly_df: pd.DataFrame,
                               amount_col: str = 'total_amount_of_payment_usdollars',
                               score_col: str = 'anomaly_score',
                               figsize: Tuple[int, int] = (16, 6)) -> plt.Figure:
        """Compare normal vs anomalous payments"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        normal_amounts = normal_df[amount_col].dropna()
        anomaly_amounts = anomaly_df[amount_col].dropna()
        
        # Histogram comparison
        ax1 = axes[0]
        ax1.hist(normal_amounts, bins=50, alpha=0.6, label='Normal', color='blue', edgecolor='black')
        ax1.hist(anomaly_amounts, bins=50, alpha=0.6, label='Anomalies', color='red', edgecolor='black')
        ax1.set_xlabel('Payment Amount (USD)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Payment Amount Distribution: Normal vs Anomalies', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Box plot comparison
        ax2 = axes[1]
        data_to_plot = [normal_amounts, anomaly_amounts]
        bp = ax2.boxplot(data_to_plot, labels=['Normal', 'Anomalies'], 
                        patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax2.set_ylabel('Payment Amount (USD)', fontsize=12, fontweight='bold')
        ax2.set_title('Payment Amount Box Plot Comparison', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_isolation_forest_analysis(self, estimator_range: List[int],
                                      mean_scores_train: List[float],
                                      mean_scores_test: List[float],
                                      std_scores_train: List[float],
                                      std_scores_test: List[float],
                                      training_times: List[float],
                                      anomaly_counts: List[int],
                                      final_anomaly_count: int,
                                      figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """Plot Isolation Forest training curve analysis"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Mean Anomaly Score Convergence
        ax1 = axes[0, 0]
        ax1.plot(estimator_range, mean_scores_train, marker='o', linewidth=2.5, 
                markersize=8, label='Train', color='blue')
        ax1.plot(estimator_range, mean_scores_test, marker='s', linewidth=2.5, 
                markersize=8, label='Test', color='orange')
        ax1.set_xlabel('Number of Estimators', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Mean Anomaly Score', fontsize=12, fontweight='bold')
        ax1.set_title('Anomaly Score Convergence', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Score Stability
        ax2 = axes[0, 1]
        ax2.plot(estimator_range, std_scores_train, marker='o', linewidth=2.5, 
                markersize=8, label='Train', color='blue')
        ax2.plot(estimator_range, std_scores_test, marker='s', linewidth=2.5, 
                markersize=8, label='Test', color='orange')
        ax2.set_xlabel('Number of Estimators', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Std Dev of Anomaly Score', fontsize=12, fontweight='bold')
        ax2.set_title('Score Stability Over Estimators', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Training Time
        ax3 = axes[1, 0]
        ax3.plot(estimator_range, training_times, marker='D', linewidth=2.5, 
                markersize=8, color='green')
        ax3.set_xlabel('Number of Estimators', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
        ax3.set_title('Training Time vs Model Complexity', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Anomaly Count Stability
        ax4 = axes[1, 1]
        ax4.plot(estimator_range, anomaly_counts, marker='v', linewidth=2.5, 
                markersize=8, color='red')
        ax4.axhline(y=final_anomaly_count, color='darkred', linestyle='--', linewidth=2, 
                   label=f'Final: {final_anomaly_count}')
        ax4.set_xlabel('Number of Estimators', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Detected Anomalies', fontsize=12, fontweight='bold')
        ax4.set_title('Anomaly Detection Stability', fontsize=13, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_reconstruction_error_analysis(self, train_mse: np.ndarray, test_mse: np.ndarray,
                                          threshold: float, all_reconstruction_errors: np.ndarray,
                                          figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
        """Plot reconstruction error analysis for Autoencoder"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Reconstruction Error Distribution
        ax1 = axes[0]
        ax1.hist(train_mse, bins=50, alpha=0.7, label='Training', color='blue', edgecolor='black')
        ax1.hist(test_mse, bins=50, alpha=0.7, label='Test', color='red', edgecolor='black')
        ax1.axvline(threshold, color='green', linestyle='--', linewidth=2.5, label='Threshold')
        ax1.set_xlabel('Reconstruction Error (MSE)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title('Reconstruction Error Distribution', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Percentile Distribution
        ax2 = axes[1]
        percentiles = np.arange(1, 101)
        ax2.plot(percentiles, np.percentile(all_reconstruction_errors, percentiles), 
                linewidth=2.5, color='darkblue', marker='o', markersize=3)
        ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=2.5, 
                   label=f'95th Percentile: {threshold:.6f}')
        ax2.fill_between(percentiles, 0, np.percentile(all_reconstruction_errors, percentiles), 
                        alpha=0.2, color='blue')
        ax2.set_xlabel('Percentile', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Anomaly Score', fontsize=12, fontweight='bold')
        ax2.set_title('Anomaly Score Percentile Distribution', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def print_anomaly_stats(self, normal_df: pd.DataFrame, anomaly_df: pd.DataFrame,
                           score_col: str = 'anomaly_score',
                           comparison_features: Optional[List[str]] = None) -> pd.DataFrame:
        """Print statistical comparison between normal and anomalous payments"""
        # Default comparison features
        if comparison_features is None:
            comparison_features = ['total_amount_of_payment_usdollars']
        
        # Filter existing columns
        available_features = [f for f in comparison_features if f in normal_df.columns and f in anomaly_df.columns]
        
        if not available_features:
            print("No comparison features available")
            return None
        
        # Create comparison statistics
        comparison_stats = pd.DataFrame({
            'Normal_Mean': normal_df[available_features].mean(),
            'Normal_Median': normal_df[available_features].median(),
            'Anomaly_Mean': anomaly_df[available_features].mean(),
            'Anomaly_Median': anomaly_df[available_features].median(),
            'Difference_%': ((anomaly_df[available_features].mean() - normal_df[available_features].mean()) / 
                            normal_df[available_features].mean() * 100)
        })
        
        # Score statistics
        score_stats = pd.DataFrame({
            'Metric': ['Mean', 'Median', 'Min', 'Max'],
            'Normal': [
                normal_df[score_col].mean(),
                normal_df[score_col].median(),
                normal_df[score_col].min(),
                normal_df[score_col].max()
            ],
            'Anomaly': [
                anomaly_df[score_col].mean(),
                anomaly_df[score_col].median(),
                anomaly_df[score_col].min(),
                anomaly_df[score_col].max()
            ]
        })
        
        print("\n=== Statistical Comparison ===")
        print(comparison_stats)
        print(f"\n=== {score_col.replace('_', ' ').title()} Statistics ===")
        print(score_stats)
        
        return comparison_stats
    
    def display_top_anomalies(self, anomalies_df: pd.DataFrame, 
                             score_col: str = 'anomaly_score',
                             top_n: int = 10,
                             key_columns: Optional[List[str]] = None) -> None:
        """Display top anomalies with key features"""
        if key_columns is None:
            # Default columns to display
            key_columns = [
                score_col,
                'anomaly_score_percentile',
                'total_amount_of_payment_usdollars',
                'covered_recipient_type',
                'nature_of_payment_or_transfer_of_value'
            ]
        
        # Filter available columns
        available_cols = [col for col in key_columns if col in anomalies_df.columns]
        
        # Add optional columns if they exist
        optional_cols = ['amt_to_avg_ratio', 'hist_pay_avg', 'is_new_recipient', 
                        'is_weekend', 'is_high_risk_nature']
        for col in optional_cols:
            if col in anomalies_df.columns and col not in available_cols:
                available_cols.append(col)
        
        print(f"\n=== Top {top_n} Anomalous Payments (n={len(anomalies_df):,}) ===")
        return anomalies_df[available_cols].head(top_n)
    
    def plot_xgboost_training_curves(self, results: Dict, best_iteration: int = None,
                                     figsize: Tuple[int, int] = (16, 5)) -> plt.Figure:
        """Plot XGBoost training curves and overfitting analysis"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        train_auc = results['validation_0']['auc']
        test_auc = results['validation_1']['auc']
        epochs = range(len(train_auc))
        
        # AUC curves
        ax1 = axes[0]
        ax1.plot(epochs, train_auc, linewidth=2.5, label='Train AUC', color='blue', marker='o', markersize=4)
        ax1.plot(epochs, test_auc, linewidth=2.5, label='Test AUC', color='orange', marker='s', markersize=4)
        if best_iteration is not None:
            ax1.axvline(x=best_iteration, color='red', linestyle='--', 
                       linewidth=2, label=f'Best Iteration: {best_iteration}')
        ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax1.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
        ax1.set_title('XGBoost Training Curve (AUC)', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Overfitting analysis
        ax2 = axes[1]
        gap = np.array(train_auc) - np.array(test_auc)
        ax2.plot(epochs, gap, linewidth=2.5, color='purple', marker='D', markersize=4)
        ax2.axhline(y=0, color='green', linestyle='--', linewidth=2, label='No Gap')
        if best_iteration is not None:
            ax2.axvline(x=best_iteration, color='red', linestyle='--', 
                       linewidth=2, label=f'Best Iteration: {best_iteration}')
        ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax2.set_ylabel('AUC Gap (Train - Test)', fontsize=12, fontweight='bold')
        ax2.set_title('Overfitting Analysis', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrices(self, y_train, train_pred, y_test, test_pred,
                               figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:
        """Plot confusion matrices for train and test sets"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Train confusion matrix
        cm_train = confusion_matrix(y_train, train_pred)
        ax1 = axes[0]
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Anomaly'], 
                   yticklabels=['Normal', 'Anomaly'],
                   ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax1.set_title('Train Confusion Matrix', fontsize=13, fontweight='bold')
        
        # Test confusion matrix
        cm_test = confusion_matrix(y_test, test_pred)
        ax2 = axes[1]
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Oranges', 
                   xticklabels=['Normal', 'Anomaly'], 
                   yticklabels=['Normal', 'Anomaly'],
                   ax=ax2, cbar_kws={'label': 'Count'})
        ax2.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax2.set_title('Test Confusion Matrix', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curves(self, y_train, train_proba, y_test, test_proba,
                       train_auc: float, test_auc: float,
                       figsize: Tuple[int, int] = (16, 6)) -> plt.Figure:
        """Plot ROC curves for train and test sets"""
        from sklearn.metrics import roc_curve
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Train ROC
        fpr_train, tpr_train, _ = roc_curve(y_train, train_proba)
        ax1 = axes[0]
        ax1.plot(fpr_train, tpr_train, linewidth=2.5, color='blue', 
                label=f'Train AUC = {train_auc:.4f}')
        ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
        ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax1.set_title('Train ROC Curve', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Test ROC
        fpr_test, tpr_test, _ = roc_curve(y_test, test_proba)
        ax2 = axes[1]
        ax2.plot(fpr_test, tpr_test, linewidth=2.5, color='orange', 
                label=f'Test AUC = {test_auc:.4f}')
        ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
        ax2.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax2.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax2.set_title('Test ROC Curve', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, feature_names, feature_importances,
                               top_n: int = 15, figsize: Tuple[int, int] = (16, 6)) -> Tuple[plt.Figure, pd.DataFrame]:
        """Plot feature importance analysis"""
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Feature importance bar plot
        ax1 = axes[0]
        top_features = feature_importance.head(top_n)
        ax1.barh(range(len(top_features)), top_features['importance'], color='steelblue', edgecolor='black')
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'])
        ax1.invert_yaxis()
        ax1.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax1.set_title(f'Top {top_n} Feature Importance', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Cumulative importance
        ax2 = axes[1]
        cumulative_importance = feature_importance['importance'].cumsum() / feature_importance['importance'].sum()
        ax2.plot(range(len(cumulative_importance)), cumulative_importance, 
                linewidth=2.5, color='darkgreen', marker='o', markersize=4)
        ax2.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='80% Threshold')
        ax2.axhline(y=0.9, color='orange', linestyle='--', linewidth=2, label='90% Threshold')
        ax2.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Importance', fontsize=12, fontweight='bold')
        ax2.set_title('Cumulative Feature Importance', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, feature_importance
    
    def plot_grid_search_results(self, grid_results_df: pd.DataFrame, 
                                figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """Plot grid search results analysis"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Score vs n_estimators
        pivot_estimators = grid_results_df.pivot_table(
            values='mean_test_score',
            index='param_n_estimators',
            aggfunc='mean'
        )
        axes[0, 0].plot(pivot_estimators.index, pivot_estimators.values, marker='o', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Number of Estimators', fontsize=12)
        axes[0, 0].set_ylabel('Mean Test Score', fontsize=12)
        axes[0, 0].set_title('Performance vs. Number of Estimators', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Score vs contamination
        pivot_contamination = grid_results_df.pivot_table(
            values='mean_test_score',
            index='param_contamination',
            aggfunc='mean'
        )
        axes[0, 1].plot(pivot_contamination.index, pivot_contamination.values, marker='s', linewidth=2, markersize=8, color='green')
        axes[0, 1].set_xlabel('Contamination', fontsize=12)
        axes[0, 1].set_ylabel('Mean Test Score', fontsize=12)
        axes[0, 1].set_title('Performance vs. Contamination', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Training time comparison
        top_10_params = grid_results_df.head(10)
        param_labels = [f"Config {i+1}" for i in range(len(top_10_params))]
        axes[1, 0].bar(param_labels, top_10_params['mean_fit_time'], color='coral', alpha=0.7)
        axes[1, 0].set_xlabel('Configuration', fontsize=12)
        axes[1, 0].set_ylabel('Mean Fit Time (s)', fontsize=12)
        axes[1, 0].set_title('Training Time - Top 10 Configurations', fontsize=14, fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Score vs fit time tradeoff
        scatter = axes[1, 1].scatter(
            grid_results_df['mean_fit_time'],
            grid_results_df['mean_test_score'],
            c=grid_results_df['param_n_estimators'].astype(float),
            s=100,
            alpha=0.6,
            cmap='viridis'
        )
        axes[1, 1].set_xlabel('Mean Fit Time (s)', fontsize=12)
        axes[1, 1].set_ylabel('Mean Test Score', fontsize=12)
        axes[1, 1].set_title('Score vs. Training Time Trade-off', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label('N Estimators', fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Mark best point
        best_idx = grid_results_df.index[0]
        axes[1, 1].scatter(
            grid_results_df.loc[best_idx, 'mean_fit_time'],
            grid_results_df.loc[best_idx, 'mean_test_score'],
            color='red',
            s=200,
            marker='*',
            edgecolors='black',
            linewidths=2,
            label='Best Model'
        )
        axes[1, 1].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_random_search_results(self, random_results_df: pd.DataFrame, 
                                   best_score: float,
                                   figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """Plot randomized search results analysis"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Score distribution
        axes[0, 0].hist(random_results_df['mean_test_score'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(best_score, color='red', linestyle='--', linewidth=2, label=f'Best: {best_score:.6f}')
        axes[0, 0].set_xlabel('Mean Test Score', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title('Score Distribution - Randomized Search', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Parameter importance (correlation with score)
        param_cols = ['param_n_estimators', 'param_contamination', 'param_max_features']
        correlations = []
        param_names = []
        
        for col in param_cols:
            if col in random_results_df.columns:
                numeric_vals = pd.to_numeric(random_results_df[col], errors='coerce')
                if numeric_vals.notna().sum() > 0:
                    corr = numeric_vals.corr(random_results_df['mean_test_score'])
                    if not np.isnan(corr):
                        correlations.append(corr)
                        param_names.append(col.replace('param_', ''))
        
        axes[0, 1].barh(param_names, correlations, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Correlation with Test Score', fontsize=12)
        axes[0, 1].set_title('Parameter Importance', fontsize=14, fontweight='bold')
        axes[0, 1].axvline(0, color='black', linewidth=0.5)
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # Plot 3: Convergence plot (score vs iteration)
        iterations = range(1, len(random_results_df) + 1)
        cumulative_best = random_results_df['mean_test_score'].expanding().max()
        axes[1, 0].plot(iterations, random_results_df['mean_test_score'], 'o', alpha=0.3, label='Individual Scores')
        axes[1, 0].plot(iterations, cumulative_best, 'r-', linewidth=2, label='Best Score Found')
        axes[1, 0].set_xlabel('Iteration', fontsize=12)
        axes[1, 0].set_ylabel('Test Score', fontsize=12)
        axes[1, 0].set_title('Convergence Plot', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Score vs parameters scatter (n_estimators vs contamination)
        scatter = axes[1, 1].scatter(
            random_results_df['param_n_estimators'],
            random_results_df['param_contamination'],
            c=random_results_df['mean_test_score'],
            s=100,
            alpha=0.6,
            cmap='RdYlGn'
        )
        axes[1, 1].set_xlabel('N Estimators', fontsize=12)
        axes[1, 1].set_ylabel('Contamination', fontsize=12)
        axes[1, 1].set_title('Parameter Space Exploration', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label('Test Score', fontsize=10)
        
        # Mark best point
        best_idx = random_results_df.index[0]
        axes[1, 1].scatter(
            random_results_df.loc[best_idx, 'param_n_estimators'],
            random_results_df.loc[best_idx, 'param_contamination'],
            color='red',
            s=300,
            marker='*',
            edgecolors='black',
            linewidths=2,
            label='Best Model',
            zorder=5
        )
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_search_comparison(self, baseline_train_time: float, 
                              grid_search_time: float, 
                              random_search_time: float,
                              grid_results_df: pd.DataFrame,
                              random_results_df: pd.DataFrame,
                              figsize: Tuple[int, int] = (16, 6)) -> plt.Figure:
        """Plot comparison between search methods"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Search time comparison
        methods = ['Baseline', 'Grid Search', 'Random Search']
        times = [baseline_train_time, grid_search_time, random_search_time]
        colors_bar = ['#3498db', '#e74c3c', '#2ecc71']
        
        bars = axes[0].bar(methods, times, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
        axes[0].set_ylabel('Time (seconds)', fontsize=12)
        axes[0].set_title('Hyperparameter Search Time Comparison', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{time_val:.2f}s\n({time_val/60:.2f}m)',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 2: Configurations tested vs time efficiency
        configs = [1, len(grid_results_df), len(random_results_df)]
        time_per_config = [t/c if c > 0 else 0 for t, c in zip(times, configs)]
        
        ax2 = axes[1]
        ax2_twin = ax2.twinx()
        
        x_pos = np.arange(len(methods))
        width = 0.35
        
        bar1 = ax2.bar(x_pos - width/2, configs, width, label='Configs Tested', color='#3498db', alpha=0.7, edgecolor='black')
        bar2 = ax2_twin.bar(x_pos + width/2, time_per_config, width, label='Time per Config', color='#e67e22', alpha=0.7, edgecolor='black')
        
        ax2.set_xlabel('Method', fontsize=12)
        ax2.set_ylabel('Configurations Tested', fontsize=12, color='#3498db')
        ax2_twin.set_ylabel('Time per Configuration (s)', fontsize=12, color='#e67e22')
        ax2.set_title('Search Efficiency Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(methods)
        ax2.tick_params(axis='y', labelcolor='#3498db')
        ax2_twin.tick_params(axis='y', labelcolor='#e67e22')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add legends
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(self, baseline_test_scores: np.ndarray,
                             optimized_test_scores: np.ndarray,
                             baseline_train_anomalies: int,
                             baseline_test_anomalies: int,
                             optimized_train_anomalies: int,
                             optimized_test_anomalies: int,
                             X_train_len: int,
                             X_test_len: int,
                             baseline_score_mean: float,
                             optimized_score_mean: float,
                             baseline_score_std: float,
                             optimized_score_std: float,
                             figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """Plot baseline vs optimized model comparison"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Test score distribution comparison
        axes[0, 0].hist(baseline_test_scores, bins=50, alpha=0.5, label='Baseline', color='blue', edgecolor='black')
        axes[0, 0].hist(optimized_test_scores, bins=50, alpha=0.5, label='Optimized', color='green', edgecolor='black')
        axes[0, 0].axvline(baseline_test_scores.mean(), color='blue', linestyle='--', linewidth=2, label='Baseline Mean')
        axes[0, 0].axvline(optimized_test_scores.mean(), color='green', linestyle='--', linewidth=2, label='Optimized Mean')
        axes[0, 0].set_xlabel('Anomaly Score', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title('Test Score Distribution Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Box plot comparison
        box_data = [baseline_test_scores, optimized_test_scores]
        box_labels = ['Baseline', 'Optimized']
        bp = axes[0, 1].boxplot(box_data, labels=box_labels, patch_artist=True,
                                boxprops=dict(facecolor='lightblue', alpha=0.7),
                                medianprops=dict(color='red', linewidth=2))
        axes[0, 1].set_ylabel('Anomaly Score', fontsize=12)
        axes[0, 1].set_title('Score Distribution - Box Plot', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Anomaly detection rate comparison
        categories = ['Train', 'Test']
        baseline_rates = [
            baseline_train_anomalies/X_train_len*100,
            baseline_test_anomalies/X_test_len*100
        ]
        optimized_rates = [
            optimized_train_anomalies/X_train_len*100,
            optimized_test_anomalies/X_test_len*100
        ]
        
        x_pos = np.arange(len(categories))
        width = 0.35
        
        bars1 = axes[1, 0].bar(x_pos - width/2, baseline_rates, width, label='Baseline', color='blue', alpha=0.7, edgecolor='black')
        bars2 = axes[1, 0].bar(x_pos + width/2, optimized_rates, width, label='Optimized', color='green', alpha=0.7, edgecolor='black')
        
        axes[1, 0].set_ylabel('Anomaly Detection Rate (%)', fontsize=12)
        axes[1, 0].set_title('Anomaly Detection Rate Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(categories)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
        
        # Plot 4: Performance metrics comparison
        metrics = ['Score\nSeparation', 'Consistency\n(1/Std)', 'Detection\nRate']
        baseline_metrics_normalized = [
            abs(baseline_score_mean) / max(abs(baseline_score_mean), abs(optimized_score_mean)),
            (1/baseline_score_std) / max(1/baseline_score_std, 1/optimized_score_std),
            (baseline_test_anomalies/X_test_len) / max(baseline_test_anomalies/X_test_len, optimized_test_anomalies/X_test_len)
        ]
        optimized_metrics_normalized = [
            abs(optimized_score_mean) / max(abs(baseline_score_mean), abs(optimized_score_mean)),
            (1/optimized_score_std) / max(1/baseline_score_std, 1/optimized_score_std),
            (optimized_test_anomalies/X_test_len) / max(baseline_test_anomalies/X_test_len, optimized_test_anomalies/X_test_len)
        ]
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        axes[1, 1].bar(x_pos - width/2, baseline_metrics_normalized, width, label='Baseline', color='blue', alpha=0.7, edgecolor='black')
        axes[1, 1].bar(x_pos + width/2, optimized_metrics_normalized, width, label='Optimized', color='green', alpha=0.7, edgecolor='black')
        axes[1, 1].set_ylabel('Normalized Performance', fontsize=12)
        axes[1, 1].set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].set_ylim([0, 1.1])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_latency_distribution(self, latencies: List[float],
                                  latency_stats: Dict[str, float],
                                  figsize: Tuple[int, int] = (16, 5)) -> plt.Figure:
        """Plot inference latency distribution with histogram and box plot"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        axes[0].hist(latencies, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(latency_stats['Mean'], color='red', linestyle='--', linewidth=2, 
                       label=f"Mean: {latency_stats['Mean']:.2f} ms")
        axes[0].axvline(latency_stats['P95'], color='orange', linestyle='--', linewidth=2, 
                       label=f"P95: {latency_stats['P95']:.2f} ms")
        axes[0].set_xlabel('Latency (ms)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Inference Latency Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        bp = axes[1].boxplot(latencies, vert=True, patch_artist=True,
                            boxprops=dict(facecolor='lightblue', alpha=0.7),
                            medianprops=dict(color='red', linewidth=2))
        axes[1].set_ylabel('Latency (ms)', fontsize=12)
        axes[1].set_title('Latency Box Plot', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
