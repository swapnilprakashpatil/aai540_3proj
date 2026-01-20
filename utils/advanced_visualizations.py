import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class AdvancedPaymentVisualizer:
    def __init__(self, color_palette: str = "viridis"):
        self.color_palette = color_palette
        
    def create_3d_scatter(self, df: pd.DataFrame, 
                         x_col: str, y_col: str, z_col: str,
                         color_col: Optional[str] = None,
                         size_col: Optional[str] = None,
                         title: str = "3D Scatter Plot",
                         sample_size: int = 1000,
                         log_y: bool = True) -> go.Figure:
        # Sample data if too large
        df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        fig = px.scatter_3d(
            df_sample,
            x=x_col,
            y=y_col,
            z=z_col,
            color=color_col,
            size=size_col if size_col else y_col,
            hover_data={
                x_col: ':,',
                y_col: ':$,.2f',
                z_col: True
            },
            labels={
                x_col: x_col.replace('_', ' ').title(),
                y_col: y_col.replace('_', ' ').title(),
                z_col: z_col.replace('_', ' ').title()
            },
            title=title,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(title=x_col.replace('_', ' ').title(), gridcolor='lightgray'),
                yaxis=dict(title=y_col.replace('_', ' ').title(), 
                          type='log' if log_y else 'linear', 
                          gridcolor='lightgray'),
                zaxis=dict(title=z_col.replace('_', ' ').title(), gridcolor='lightgray'),
                bgcolor='rgba(240,240,240,0.9)'
            ),
            height=700,
            font=dict(size=11)
        )
        
        return fig
    
    def create_parallel_coordinates(self, df: pd.DataFrame, 
                                    feature_cols: List[str],
                                    color_col: Optional[str] = None,
                                    title: str = "Parallel Coordinates",
                                    sample_size: int = 500) -> go.Figure:
        df_parallel = df[feature_cols].copy()
        df_parallel = df_parallel.sample(n=min(sample_size, len(df_parallel)), random_state=42)
        
        # Add color column if specified
        if color_col and color_col in df.columns:
            sample_idx = df_parallel.index
            df_parallel['Color_Column'] = df.loc[sample_idx, color_col]
            
            # Create numeric mapping for coloring
            if df_parallel['Color_Column'].dtype == 'object':
                type_map = {t: i for i, t in enumerate(df_parallel['Color_Column'].unique())}
                color_values = df_parallel['Color_Column'].map(type_map)
            else:
                color_values = df_parallel['Color_Column']
        else:
            color_values = df_parallel[feature_cols[0]]
        
        # Create dimensions
        dimensions = []
        for col in feature_cols:
            dimensions.append(
                dict(
                    label=col.replace('Total_Amount_of_Payment_USDollars_', '').replace('_', ' ').title()[:30],
                    values=df_parallel[col],
                    range=[df_parallel[col].quantile(0.01), df_parallel[col].quantile(0.99)]
                )
            )
        
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(
                    color=color_values,
                    colorscale='Viridis',
                    showscale=True,
                    cmin=color_values.min(),
                    cmax=color_values.max()
                ),
                dimensions=dimensions
            )
        )
        
        fig.update_layout(
            title=title,
            font=dict(size=11),
            height=600,
            margin=dict(l=150, r=150, t=80, b=50)
        )
        
        return fig
    
    def create_sunburst_chart(self, df: pd.DataFrame, 
                             path_cols: List[str],
                             value_col: str,
                             title: str = "Sunburst Chart") -> go.Figure:
        fig = px.sunburst(
            df,
            path=path_cols,
            values=value_col,
            title=title,
            color=value_col,
            color_continuous_scale='Viridis',
            hover_data={value_col: ':$,.2f'}
        )
        
        fig.update_layout(
            height=700,
            font=dict(size=11)
        )
        
        return fig
    
    def create_treemap(self, df: pd.DataFrame,
                      path_cols: List[str],
                      value_col: str,
                      title: str = "Treemap",
                      threshold_percentile: float = 0.90) -> go.Figure:
        # Filter by threshold
        threshold = df[value_col].quantile(threshold_percentile)
        df_filtered = df[df[value_col] >= threshold]
        
        fig = px.treemap(
            df_filtered,
            path=path_cols,
            values=value_col,
            title=title,
            color=value_col,
            color_continuous_scale='RdYlGn_r',
            hover_data={value_col: ':$,.2f'}
        )
        
        fig.update_layout(
            height=700,
            font=dict(size=11),
            coloraxis_colorbar=dict(
                title="Amount ($)",
                thickness=15,
                len=0.7
            )
        )
        
        fig.update_traces(
            textposition='middle center',
            textfont_size=10,
            marker=dict(line=dict(width=2, color='white'))
        )
        
        return fig
    
    def create_violin_plot(self, df: pd.DataFrame,
                          x_col: str, y_col: str,
                          title: str = "Violin Plot",
                          sample_size: int = 5000,
                          log_y: bool = True) -> go.Figure:
        df_violin = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        fig = px.violin(
            df_violin,
            x=x_col,
            y=y_col,
            color=x_col,
            box=True,
            points='outliers',
            title=title,
            labels={
                y_col: y_col.replace('_', ' ').title(),
                x_col: x_col.replace('_', ' ').title()
            },
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        fig.update_layout(
            showlegend=False,
            height=600,
            yaxis_type='log' if log_y else 'linear',
            font=dict(size=11),
            hovermode='closest'
        )
        
        fig.update_traces(
            meanline_visible=True,
            jitter=0.05,
            scalemode='width',
            width=0.8
        )
        
        return fig
    
    def create_radar_chart(self, df: pd.DataFrame,
                          group_col: str,
                          metric_cols: List[str],
                          title: str = "Radar Chart",
                          normalize: bool = True) -> go.Figure:
        radar_data = df.groupby(group_col)[metric_cols].mean()
        
        # Normalize if requested
        if normalize:
            scaler = MinMaxScaler()
            radar_data_normalized = pd.DataFrame(
                scaler.fit_transform(radar_data),
                index=radar_data.index,
                columns=radar_data.columns
            )
        else:
            radar_data_normalized = radar_data
        
        # Create figure
        fig = go.Figure()
        
        metric_labels = [m.replace('Total_Amount_of_Payment_USDollars_', '').replace('_', ' ').title() 
                        for m in metric_cols]
        
        for group in radar_data_normalized.index:
            values = radar_data_normalized.loc[group].tolist()
            values.append(values[0])  # Close the radar chart
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metric_labels + [metric_labels[0]],
                fill='toself',
                name=group,
                opacity=0.6
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1] if normalize else None,
                    showticklabels=True,
                    ticks='outside'
                )
            ),
            showlegend=True,
            title=title,
            height=600,
            font=dict(size=11)
        )
        
        return fig
    
    def create_animated_bubble(self, df: pd.DataFrame,
                              x_col: str, y_col: str,
                              size_col: str, color_col: str,
                              animation_col: str,
                              title: str = "Animated Bubble Chart") -> go.Figure:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            size=size_col,
            color=color_col,
            animation_frame=animation_col,
            hover_name=color_col,
            hover_data={
                x_col: ':,',
                y_col: ':$,.0f',
                size_col: ':$,.2f',
                animation_col: True
            },
            title=title,
            labels={
                x_col: x_col.replace('_', ' ').title(),
                y_col: y_col.replace('_', ' ').title(),
                color_col: color_col.replace('_', ' ').title()
            },
            size_max=60,
            range_x=[0, df[x_col].max() * 1.1],
            range_y=[0, df[y_col].max() * 1.1]
        )
        
        fig.update_layout(
            height=650,
            font=dict(size=11),
            xaxis=dict(title=x_col.replace('_', ' ').title(), gridcolor='lightgray'),
            yaxis=dict(title=y_col.replace('_', ' ').title(), gridcolor='lightgray'),
            plot_bgcolor='rgba(240,240,240,0.5)'
        )
        
        # Customize animation
        if hasattr(fig, 'layout') and hasattr(fig.layout, 'updatemenus'):
            fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 800
            fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 500
        
        return fig
    
    def create_correlation_heatmap(self, df: pd.DataFrame,
                                   feature_cols: List[str],
                                   title: str = "Correlation Matrix",
                                   annotate: bool = True) -> go.Figure:
        corr_matrix = df[feature_cols].corr()
        
        # Shorten labels
        labels = [col.replace('Total_Amount_of_Payment_USDollars_', '').replace('_', ' ').title()[:30] 
                 for col in corr_matrix.columns]
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=labels,
            y=labels,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}' if annotate else None,
            textfont={"size": 10},
            colorbar=dict(title="Correlation", thickness=15, len=0.7)
        ))
        
        fig.update_layout(
            title=title,
            xaxis={'side': 'bottom'},
            height=600,
            width=700,
            font=dict(size=10),
            margin=dict(l=150, r=50, t=80, b=150)
        )
        
        fig.update_xaxes(tickangle=45, tickfont=dict(size=9))
        fig.update_yaxes(tickfont=dict(size=9))
        
        return fig
    
    def get_strong_correlations(self, df: pd.DataFrame,
                               feature_cols: List[str],
                               threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        corr_matrix = df[feature_cols].corr()
        
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    strong_corr.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        return sorted(strong_corr, key=lambda x: abs(x[2]), reverse=True)
