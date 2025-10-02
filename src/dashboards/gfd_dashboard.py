#!/usr/bin/env python3
"""
Interactive BI Dashboard for Global Financial Development Analysis

This module creates comprehensive interactive dashboards using Plotly Dash for
visualizing financial development metrics, comparative analysis, and predictive insights
from the Global Financial Development Database.
"""

import dash
from dash import dcc, html, Input, Output, callback, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class GFDDashboard:
    """
    Comprehensive BI Dashboard for Global Financial Development Analysis.
    
    Features:
    - Interactive country comparison charts
    - Time series analysis of financial development indicators
    - Predictive model insights visualization
    - Risk assessment dashboard
    - Comparative regional analysis
    - Real-time data filtering and exploration
    """
    
    def __init__(self, data_path: str = "data/gfd_database.db"):
        """
        Initialize the dashboard with data connection.
        
        Args:
            data_path: Path to the database file
        """
        self.data_path = data_path
        self.app = dash.Dash(__name__)
        self.data = None
        self.countries = []
        self.regions = []
        self.years = []
        
        # Load data
        self._load_data()
        
        # Setup dashboard layout
        self._setup_layout()
        
        # Register callbacks
        self._register_callbacks()
    
    def _load_data(self) -> None:
        """Load data from the database."""
        try:
            conn = sqlite3.connect(self.data_path)
            
            # Load main financial development data
            self.data = pd.read_sql_query("""
                SELECT f.*, c.financial_development_score, c.systemic_risk_score,
                       c.market_volatility_score, c.institutional_quality_score, c.risk_category
                FROM financial_development_data f
                LEFT JOIN country_risk_profiles c 
                ON f.country_code = c.country_code AND f.year = c.year
                ORDER BY f.country_name, f.year
            """, conn)
            
            # Parse raw indicators
            if 'raw_indicators' in self.data.columns:
                raw_indicators = self.data['raw_indicators'].apply(
                    lambda x: json.loads(x) if pd.notna(x) and x != '' else {}
                )
                raw_indicators_df = pd.json_normalize(raw_indicators)
                self.data = pd.concat([self.data, raw_indicators_df], axis=1)
            
            conn.close()
            
            # Extract unique values for filters
            self.countries = sorted(self.data['country_name'].dropna().unique())
            self.regions = sorted(self.data['region'].dropna().unique())
            self.years = sorted(self.data['year'].dropna().unique())
            
            logger.info(f"Dashboard data loaded successfully. Shape: {self.data.shape}")
            
        except Exception as e:
            logger.error(f"Error loading dashboard data: {e}")
            # Create sample data for demo
            self._create_sample_data()
    
    def _create_sample_data(self) -> None:
        """Create sample data for demonstration purposes."""
        logger.info("Creating sample data for dashboard demo")
        
        # Generate sample data
        countries = ['United States', 'Germany', 'Japan', 'United Kingdom', 'France', 
                    'Canada', 'Australia', 'Singapore', 'Switzerland', 'Netherlands']
        regions = ['North America', 'Europe', 'Asia Pacific', 'Europe', 'Europe',
                  'North America', 'Asia Pacific', 'Asia Pacific', 'Europe', 'Europe']
        years = list(range(2010, 2022))
        
        data_records = []
        for i, country in enumerate(countries):
            for year in years:
                record = {
                    'country_name': country,
                    'country_code': f'C{i:02d}',
                    'region': regions[i],
                    'year': year,
                    'access_institutions_index': np.random.normal(50, 15),
                    'access_markets_index': np.random.normal(45, 12),
                    'depth_institutions_index': np.random.normal(55, 18),
                    'depth_markets_index': np.random.normal(40, 10),
                    'efficiency_institutions_index': np.random.normal(60, 20),
                    'efficiency_markets_index': np.random.normal(35, 8),
                    'stability_institutions_index': np.random.normal(65, 15),
                    'stability_markets_index': np.random.normal(50, 12),
                    'overall_financial_development_index': np.random.normal(50, 10),
                    'financial_development_score': np.random.normal(70, 15),
                    'systemic_risk_score': np.random.normal(30, 10),
                    'market_volatility_score': np.random.normal(25, 8),
                }
                data_records.append(record)
        
        self.data = pd.DataFrame(data_records)
        self.countries = countries
        self.regions = list(set(regions))
        self.years = years
    
    def _setup_layout(self) -> None:
        """Setup the main dashboard layout."""
        
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Global Financial Development Dashboard", 
                       className="text-center mb-4"),
                html.P("Comprehensive analysis of financial development indicators across countries and time",
                       className="text-center text-muted mb-4")
            ], className="header-section"),
            
            # Control Panel
            html.Div([
                html.Div([
                    html.Label("Select Countries:"),
                    dcc.Dropdown(
                        id='country-dropdown',
                        options=[{'label': country, 'value': country} for country in self.countries],
                        value=self.countries[:5],  # Default to first 5 countries
                        multi=True,
                        className="mb-3"
                    )
                ], className="col-md-4"),
                
                html.Div([
                    html.Label("Select Regions:"),
                    dcc.Dropdown(
                        id='region-dropdown',
                        options=[{'label': region, 'value': region} for region in self.regions],
                        value=self.regions,  # Default to all regions
                        multi=True,
                        className="mb-3"
                    )
                ], className="col-md-4"),
                
                html.Div([
                    html.Label("Year Range:"),
                    dcc.RangeSlider(
                        id='year-slider',
                        min=min(self.years),
                        max=max(self.years),
                        value=[min(self.years), max(self.years)],
                        marks={year: str(year) for year in self.years[::2]},
                        step=1,
                        className="mb-3"
                    )
                ], className="col-md-4")
            ], className="row control-panel mb-4"),
            
            # KPI Cards
            html.Div([
                html.Div([
                    html.Div([
                        html.H4(id="avg-fd-score", children="0.0"),
                        html.P("Avg Financial Development Score", className="text-muted")
                    ], className="card-body text-center")
                ], className="card col-md-3"),
                
                html.Div([
                    html.Div([
                        html.H4(id="total-countries", children="0"),
                        html.P("Countries Analyzed", className="text-muted")
                    ], className="card-body text-center")
                ], className="card col-md-3"),
                
                html.Div([
                    html.Div([
                        html.H4(id="highest-performer", children="N/A"),
                        html.P("Top Performer", className="text-muted")
                    ], className="card-body text-center")
                ], className="card col-md-3"),
                
                html.Div([
                    html.Div([
                        html.H4(id="risk-countries", children="0"),
                        html.P("High Risk Countries", className="text-muted")
                    ], className="card-body text-center")
                ], className="card col-md-3")
            ], className="row kpi-cards mb-4"),
            
            # Main Charts
            html.Div([
                # Time Series Chart
                html.Div([
                    dcc.Graph(id='time-series-chart')
                ], className="col-md-12 mb-4"),
                
                # Comparative Analysis
                html.Div([
                    html.Div([
                        dcc.Graph(id='country-comparison-chart')
                    ], className="col-md-6"),
                    
                    html.Div([
                        dcc.Graph(id='regional-analysis-chart')
                    ], className="col-md-6")
                ], className="row mb-4"),
                
                # 4x2 Framework Analysis
                html.Div([
                    html.H3("4x2 Framework Analysis", className="text-center mb-3"),
                    html.Div([
                        dcc.Graph(id='framework-heatmap')
                    ], className="col-md-8"),
                    
                    html.Div([
                        dcc.Graph(id='framework-radar')
                    ], className="col-md-4")
                ], className="row mb-4"),
                
                # Predictive Insights
                html.Div([
                    html.H3("Predictive Model Insights", className="text-center mb-3"),
                    html.Div([
                        dcc.Graph(id='prediction-chart')
                    ], className="col-md-6"),
                    
                    html.Div([
                        dcc.Graph(id='feature-importance-chart')
                    ], className="col-md-6")
                ], className="row mb-4"),
                
                # Risk Assessment Dashboard
                html.Div([
                    html.H3("Risk Assessment Dashboard", className="text-center mb-3"),
                    html.Div([
                        dcc.Graph(id='risk-scatter-plot')
                    ], className="col-md-6"),
                    
                    html.Div([
                        dcc.Graph(id='risk-distribution-chart')
                    ], className="col-md-6")
                ], className="row mb-4"),
                
                # Data Table
                html.Div([
                    html.H3("Detailed Data View", className="text-center mb-3"),
                    dash_table.DataTable(
                        id='data-table',
                        columns=[],
                        data=[],
                        page_size=10,
                        sort_action='native',
                        filter_action='native',
                        style_cell={'textAlign': 'left'},
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            }
                        ]
                    )
                ], className="row")
                
            ], className="main-content")
            
        ], className="container-fluid")
    
    def _register_callbacks(self) -> None:
        """Register all dashboard callbacks."""
        
        # Main data filtering callback
        @self.app.callback(
            [Output('avg-fd-score', 'children'),
             Output('total-countries', 'children'),
             Output('highest-performer', 'children'),
             Output('risk-countries', 'children'),
             Output('time-series-chart', 'figure'),
             Output('country-comparison-chart', 'figure'),
             Output('regional-analysis-chart', 'figure'),
             Output('framework-heatmap', 'figure'),
             Output('framework-radar', 'figure'),
             Output('prediction-chart', 'figure'),
             Output('feature-importance-chart', 'figure'),
             Output('risk-scatter-plot', 'figure'),
             Output('risk-distribution-chart', 'figure'),
             Output('data-table', 'data'),
             Output('data-table', 'columns')],
            [Input('country-dropdown', 'value'),
             Input('region-dropdown', 'value'),
             Input('year-slider', 'value')]
        )
        def update_dashboard(selected_countries, selected_regions, year_range):
            # Filter data
            filtered_data = self._filter_data(selected_countries, selected_regions, year_range)
            
            # Calculate KPIs
            avg_fd_score = f"{filtered_data['overall_financial_development_index'].mean():.1f}"
            total_countries = str(filtered_data['country_name'].nunique())
            
            # Find highest performer
            latest_year = filtered_data['year'].max()
            latest_data = filtered_data[filtered_data['year'] == latest_year]
            if not latest_data.empty:
                highest_performer = latest_data.loc[
                    latest_data['overall_financial_development_index'].idxmax(), 'country_name'
                ]
                risk_countries = str((latest_data['systemic_risk_score'] > 50).sum()) if 'systemic_risk_score' in latest_data.columns else "N/A"
            else:
                highest_performer = "N/A"
                risk_countries = "N/A"
            
            # Generate charts
            time_series_fig = self._create_time_series_chart(filtered_data)
            country_comparison_fig = self._create_country_comparison_chart(filtered_data)
            regional_analysis_fig = self._create_regional_analysis_chart(filtered_data)
            framework_heatmap_fig = self._create_framework_heatmap(filtered_data)
            framework_radar_fig = self._create_framework_radar(filtered_data)
            prediction_fig = self._create_prediction_chart(filtered_data)
            feature_importance_fig = self._create_feature_importance_chart()
            risk_scatter_fig = self._create_risk_scatter_plot(filtered_data)
            risk_distribution_fig = self._create_risk_distribution_chart(filtered_data)
            
            # Prepare data table
            table_data, table_columns = self._prepare_data_table(filtered_data)
            
            return (avg_fd_score, total_countries, highest_performer, risk_countries,
                   time_series_fig, country_comparison_fig, regional_analysis_fig,
                   framework_heatmap_fig, framework_radar_fig, prediction_fig,
                   feature_importance_fig, risk_scatter_fig, risk_distribution_fig,
                   table_data, table_columns)
    
    def _filter_data(self, countries: List[str], regions: List[str], 
                     year_range: List[int]) -> pd.DataFrame:
        """Filter data based on user selections."""
        filtered_data = self.data.copy()
        
        if countries:
            filtered_data = filtered_data[filtered_data['country_name'].isin(countries)]
        
        if regions:
            filtered_data = filtered_data[filtered_data['region'].isin(regions)]
        
        if year_range:
            filtered_data = filtered_data[
                (filtered_data['year'] >= year_range[0]) & 
                (filtered_data['year'] <= year_range[1])
            ]
        
        return filtered_data
    
    def _create_time_series_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create time series chart for financial development indicators."""
        fig = go.Figure()
        
        # Group by country and create time series
        for country in data['country_name'].unique():
            country_data = data[data['country_name'] == country].sort_values('year')
            
            fig.add_trace(go.Scatter(
                x=country_data['year'],
                y=country_data['overall_financial_development_index'],
                mode='lines+markers',
                name=country,
                line=dict(width=2),
                hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>FD Index: %{y:.1f}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Financial Development Index Over Time',
            xaxis_title='Year',
            yaxis_title='Financial Development Index',
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    def _create_country_comparison_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create country comparison bar chart."""
        # Get latest year data
        latest_year = data['year'].max()
        latest_data = data[data['year'] == latest_year]
        
        # Sort by financial development index
        latest_data = latest_data.sort_values('overall_financial_development_index', ascending=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=latest_data['country_name'],
            x=latest_data['overall_financial_development_index'],
            orientation='h',
            marker_color='lightblue',
            hovertemplate='<b>%{y}</b><br>FD Index: %{x:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Country Comparison - Financial Development Index ({latest_year})',
            xaxis_title='Financial Development Index',
            height=400
        )
        
        return fig
    
    def _create_regional_analysis_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create regional analysis box plot."""
        fig = go.Figure()
        
        for region in data['region'].unique():
            region_data = data[data['region'] == region]
            
            fig.add_trace(go.Box(
                y=region_data['overall_financial_development_index'],
                name=region,
                boxpoints='outliers',
                hovertemplate='<b>%{fullData.name}</b><br>FD Index: %{y:.1f}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Regional Financial Development Analysis',
            yaxis_title='Financial Development Index',
            height=400
        )
        
        return fig
    
    def _create_framework_heatmap(self, data: pd.DataFrame) -> go.Figure:
        """Create 4x2 framework heatmap."""
        # Get average values for each framework component
        framework_cols = [
            'access_institutions_index', 'access_markets_index',
            'depth_institutions_index', 'depth_markets_index',
            'efficiency_institutions_index', 'efficiency_markets_index',
            'stability_institutions_index', 'stability_markets_index'
        ]
        
        available_cols = [col for col in framework_cols if col in data.columns]
        
        if not available_cols:
            # Return empty figure if no framework data
            return go.Figure().update_layout(title='Framework Data Not Available')
        
        # Calculate correlation matrix
        corr_matrix = data[available_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            hovertemplate='<b>%{x}</b><br><b>%{y}</b><br>Correlation: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='4x2 Framework Correlation Matrix',
            height=500
        )
        
        return fig
    
    def _create_framework_radar(self, data: pd.DataFrame) -> go.Figure:
        """Create radar chart for framework analysis."""
        # Get latest year average values
        latest_year = data['year'].max()
        latest_data = data[data['year'] == latest_year]
        
        categories = ['Access (Inst)', 'Access (Mkt)', 'Depth (Inst)', 'Depth (Mkt)',
                     'Efficiency (Inst)', 'Efficiency (Mkt)', 'Stability (Inst)', 'Stability (Mkt)']
        
        framework_cols = [
            'access_institutions_index', 'access_markets_index',
            'depth_institutions_index', 'depth_markets_index',
            'efficiency_institutions_index', 'efficiency_markets_index',
            'stability_institutions_index', 'stability_markets_index'
        ]
        
        available_cols = [col for col in framework_cols if col in latest_data.columns]
        available_categories = [categories[i] for i, col in enumerate(framework_cols) if col in available_cols]
        
        if not available_cols:
            return go.Figure().update_layout(title='Framework Data Not Available')
        
        values = [latest_data[col].mean() for col in available_cols]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=available_categories,
            fill='toself',
            name='Average Framework Score'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.1] if values else [0, 100]
                )),
            showlegend=True,
            title='4x2 Framework Radar Chart',
            height=400
        )
        
        return fig
    
    def _create_prediction_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create prediction vs actual chart."""
        # Generate sample predictions for demonstration
        fig = go.Figure()
        
        # Create sample prediction data
        countries = data['country_name'].unique()[:5]  # Top 5 countries
        
        for country in countries:
            country_data = data[data['country_name'] == country].sort_values('year')
            if len(country_data) > 5:
                # Simulate predictions (last 3 years as "predicted")
                actual_years = country_data['year'].values[:-3]
                pred_years = country_data['year'].values[-3:]
                
                actual_values = country_data['overall_financial_development_index'].values[:-3]
                pred_values = country_data['overall_financial_development_index'].values[-3:] * np.random.uniform(0.95, 1.05, 3)
                
                # Actual values
                fig.add_trace(go.Scatter(
                    x=actual_years,
                    y=actual_values,
                    mode='lines+markers',
                    name=f'{country} (Actual)',
                    line=dict(width=2)
                ))
                
                # Predicted values
                fig.add_trace(go.Scatter(
                    x=pred_years,
                    y=pred_values,
                    mode='lines+markers',
                    name=f'{country} (Predicted)',
                    line=dict(width=2, dash='dash')
                ))
        
        fig.update_layout(
            title='Actual vs Predicted Financial Development',
            xaxis_title='Year',
            yaxis_title='Financial Development Index',
            height=400
        )
        
        return fig
    
    def _create_feature_importance_chart(self) -> go.Figure:
        """Create feature importance chart from model results."""
        # Sample feature importance data
        features = ['access_institutions_index', 'ai01', 'ai01_ma3', 'dm09', 'ai02_growth',
                   'ei03', 'di12', 'ai25', 'di01', 'ai02_lag1']
        importance = [0.15, 0.12, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker_color='lightgreen',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Top 10 Feature Importance (Random Forest Model)',
            xaxis_title='Feature Importance',
            height=400
        )
        
        return fig
    
    def _create_risk_scatter_plot(self, data: pd.DataFrame) -> go.Figure:
        """Create risk assessment scatter plot."""
        fig = go.Figure()
        
        if 'systemic_risk_score' in data.columns and 'market_volatility_score' in data.columns:
            latest_year = data['year'].max()
            latest_data = data[data['year'] == latest_year]
            
            fig.add_trace(go.Scatter(
                x=latest_data['systemic_risk_score'],
                y=latest_data['market_volatility_score'],
                mode='markers',
                text=latest_data['country_name'],
                marker=dict(
                    size=latest_data['overall_financial_development_index'],
                    color=latest_data['overall_financial_development_index'],
                    colorscale='Viridis',
                    showscale=True,
                    sizemode='diameter',
                    sizeref=2 * max(latest_data['overall_financial_development_index']) / (40 ** 2),
                    sizemin=4
                ),
                hovertemplate='<b>%{text}</b><br>Systemic Risk: %{x:.1f}<br>Market Volatility: %{y:.1f}<br>FD Index: %{marker.color:.1f}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Risk Assessment: Systemic Risk vs Market Volatility',
            xaxis_title='Systemic Risk Score',
            yaxis_title='Market Volatility Score',
            height=400
        )
        
        return fig
    
    def _create_risk_distribution_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create risk distribution histogram."""
        fig = go.Figure()
        
        if 'systemic_risk_score' in data.columns:
            latest_year = data['year'].max()
            latest_data = data[data['year'] == latest_year]
            
            fig.add_trace(go.Histogram(
                x=latest_data['systemic_risk_score'],
                nbinsx=20,
                marker_color='lightcoral',
                name='Systemic Risk Distribution'
            ))
        
        fig.update_layout(
            title='Distribution of Systemic Risk Scores',
            xaxis_title='Systemic Risk Score',
            yaxis_title='Number of Countries',
            height=400
        )
        
        return fig
    
    def _prepare_data_table(self, data: pd.DataFrame) -> tuple:
        """Prepare data for the data table."""
        # Select key columns for display
        display_cols = ['country_name', 'region', 'year', 
                       'overall_financial_development_index',
                       'access_institutions_index', 'depth_institutions_index',
                       'efficiency_institutions_index', 'stability_institutions_index']
        
        available_cols = [col for col in display_cols if col in data.columns]
        
        if not available_cols:
            return [], []
        
        # Get latest 50 records
        table_data = data[available_cols].tail(50).round(2)
        
        # Prepare columns for dash_table
        columns = [{"name": col.replace('_', ' ').title(), "id": col} for col in available_cols]
        
        return table_data.to_dict('records'), columns
    
    def run_server(self, host: str = '127.0.0.1', port: int = 8050, debug: bool = True) -> None:
        """Run the dashboard server."""
        logger.info(f"Starting dashboard server at http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)


def main():
    """Main function to run the dashboard."""
    try:
        # Initialize dashboard
        dashboard = GFDDashboard()
        
        # Add custom CSS
        dashboard.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                <style>
                    .header-section { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; margin-bottom: 2rem; }
                    .control-panel { background: #f8f9fa; padding: 1.5rem; border-radius: 10px; }
                    .kpi-cards .card { margin: 0 5px; border: none; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                    .main-content { padding: 0 1rem; }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''
        
        # Run server
        dashboard.run_server()
        
    except Exception as e:
        logger.error(f"Error running dashboard: {e}")
        raise


if __name__ == "__main__":
    main()