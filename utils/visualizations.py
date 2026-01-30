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
