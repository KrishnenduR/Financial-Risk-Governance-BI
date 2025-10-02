# 🏦 Financial Risk Governance BI Dashboard

A comprehensive interactive business intelligence dashboard for analyzing global financial development metrics, model performance, and risk governance insights.

## 📊 Dashboard Features

### 1. 📊 Overview Tab
- **Key Performance Indicators**: Global averages, country counts, data coverage, model accuracy
- **Global Trends**: Financial development progression over time
- **Distribution Analysis**: Current financial development distribution across countries
- **Top/Bottom Performers**: Country ranking tables with conditional formatting

### 2. 🌍 Geographic Analysis Tab
- **Interactive World Map**: Choropleth visualization of financial development by country
- **Year Slider**: Dynamic filtering to view historical progression
- **Country Comparison**: Horizontal bar charts showing top performers by year
- **Regional Insights**: Geographic patterns in financial development

### 3. 📈 Time Series Analysis Tab
- **Multi-Country Comparison**: Interactive line charts for selected countries
- **Component Analysis**: Detailed breakdown of financial development components
- **Trend Analysis**: Long-term patterns and growth trajectories
- **Dynamic Filtering**: Country selection dropdown with multi-select capability

### 4. 🤖 Model Performance Tab
- **Model Comparison**: R² scores, RMSE, MAE, MAPE metrics
- **Validation Results**: Cross-validation and backtesting performance
- **Stress Testing**: Model resilience under various economic scenarios
- **Performance Tables**: Detailed model statistics with status indicators

### 5. ⚠️ Risk Assessment Tab
- **Risk Factor Heatmap**: Correlation analysis between risks and financial components
- **Country Risk Profiles**: Risk level classification and recommendations
- **Stress Testing Results**: Model sensitivity to economic shocks
- **Risk Monitoring**: Automated alerts and thresholds

### 6. 🔍 Data Quality Tab
- **Data Completeness**: Missing data analysis by year and indicator
- **Quality Trends**: Data quality scores over time
- **Missing Data Patterns**: Visual analysis of data gaps
- **Quality Metrics**: Comprehensive data quality assessment tables

## 🚀 Quick Start

### Prerequisites
```bash
pip install dash plotly pandas numpy sqlite3
```

### Launch Dashboard
```bash
# Basic launch (default: localhost:8050)
python run_dashboard.py

# Advanced options
python run_dashboard.py --host 0.0.0.0 --port 8080 --debug

# With custom database
python run_dashboard.py --db-path /path/to/your/database.db
```

### Command Line Options
- `--host HOST`: Host address (default: 127.0.0.1)
- `--port PORT`: Port number (default: 8050)
- `--debug`: Enable debug mode with auto-reload
- `--db-path PATH`: Custom database path

## 🏗️ Architecture

### Dashboard Structure
```
src/dashboards/
├── financial_bi_dashboard.py    # Main dashboard application
└── components/                  # Reusable dashboard components
    ├── charts.py               # Chart creation utilities
    ├── tables.py               # Table formatting utilities
    └── layouts.py              # Layout components
```

### Data Sources
- **SQLite Database**: Primary data source (`data/processed/financial_data.db`)
- **Model Results**: JSON files from ML pipeline (`models/results/`)
- **Validation Results**: Model validation outputs (`models/validation/`)
- **Fallback Data**: Sample data generated when database unavailable

### Key Classes

#### `FinancialDashboard`
Main dashboard class that orchestrates the entire application:

```python
class FinancialDashboard:
    def __init__(self, db_path: str)
    def _load_data(self)                    # Load financial data
    def _load_model_results(self)           # Load ML model results
    def _load_validation_results(self)      # Load validation metrics
    def _setup_layout(self)                 # Create dashboard layout
    def _setup_callbacks(self)              # Setup interactivity
    def run(self, host, port, debug)        # Launch dashboard
```

## 📈 Visualization Types

### Charts and Graphs
- **Line Charts**: Time series trends and multi-country comparisons
- **Bar Charts**: Country rankings and model performance comparisons
- **Choropleth Maps**: Geographic distribution of financial development
- **Heatmaps**: Risk factor correlations and data quality matrices
- **Histograms**: Distribution analysis and data completeness
- **Scatter Plots**: Model validation performance (R² vs MAE)

### Interactive Components
- **Sliders**: Year selection for temporal analysis
- **Dropdowns**: Country selection with multi-select
- **Tabs**: Main navigation between dashboard sections
- **Tables**: Sortable and filterable data displays
- **Conditional Formatting**: Color-coded status indicators

## 🔧 Customization

### Adding New Tabs
```python
# In _setup_layout method
dcc.Tab(label='🆕 New Tab', value='newtab')

# In _setup_callbacks method
elif active_tab == 'newtab':
    return self._create_newtab()

# Create new tab method
def _create_newtab(self):
    return html.Div([
        html.H3("New Analysis Tab"),
        # Add your components here
    ])
```

### Custom Visualizations
```python
def _create_custom_chart(self):
    """Create a custom visualization."""
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(
        x=data['x'],
        y=data['y'],
        mode='markers',
        name='Custom Data'
    ))
    
    # Update layout
    fig.update_layout(
        title='Custom Chart',
        template='plotly_white'
    )
    
    return fig
```

### Styling and Themes
The dashboard uses:
- **Plotly White Theme**: Clean, professional appearance
- **CSS Grid Layout**: Responsive design with Dash CSS classes
- **Color Palette**: Consistent color scheme across all visualizations
- **Bootstrap CSS**: External stylesheet for professional styling

## 🎨 Color Scheme

### Status Colors
- 🟢 **Success/Good**: `#27ae60` (Green)
- 🟡 **Warning/Medium**: `#f39c12` (Orange)
- 🔴 **Error/High Risk**: `#e74c3c` (Red)
- 🔵 **Info/Neutral**: `#3498db` (Blue)
- ⚫ **Secondary**: `#7f8c8d` (Gray)

### Chart Colors
- **Primary**: `#3498db` (Blue)
- **Secondary**: `#e74c3c` (Red)
- **Accent**: `#27ae60` (Green)
- **Warning**: `#f39c12` (Orange)
- **Background**: `#ecf0f1` (Light Gray)

## 📱 Responsive Design

The dashboard is designed to work across different screen sizes:
- **Desktop**: Full feature set with multi-column layouts
- **Tablet**: Responsive grid that stacks on medium screens
- **Mobile**: Single-column layout with touch-friendly controls

## 🔒 Security Considerations

### Data Protection
- **Local Database**: Data remains on your machine
- **No External APIs**: Self-contained dashboard
- **Localhost Binding**: Default configuration restricts external access
- **Debug Mode**: Disable in production environments

### Network Security
```bash
# Secure production deployment
python run_dashboard.py --host 127.0.0.1 --port 8050
# Never use --host 0.0.0.0 in production without proper firewall
```

## 🐛 Troubleshooting

### Common Issues

#### Dashboard Won't Start
```bash
# Check dependencies
pip install -r requirements.txt

# Verify Python version (3.7+)
python --version

# Check port availability
netstat -an | grep 8050
```

#### Database Connection Issues
```bash
# Check database file exists
ls -la data/processed/financial_data.db

# Use sample data mode
python run_dashboard.py --db-path non_existent.db
```

#### Performance Issues
```bash
# Reduce data size for testing
python run_dashboard.py --debug

# Check memory usage
# Consider data sampling for large datasets
```

### Error Messages

#### Import Errors
```
ImportError: No module named 'dash'
Solution: pip install dash plotly pandas numpy
```

#### Port in Use
```
OSError: [Errno 98] Address already in use
Solution: Use different port or kill existing process
```

#### Memory Issues
```
MemoryError: Unable to allocate array
Solution: Implement data pagination or sampling
```

## 📊 Performance Optimization

### Data Loading
- **Lazy Loading**: Load data only when needed
- **Caching**: Cache expensive computations
- **Sampling**: Use data samples for large datasets
- **Indexing**: Ensure proper database indexing

### Visualization Performance
- **Data Aggregation**: Pre-aggregate large datasets
- **Chart Limits**: Limit data points in visualizations
- **Async Loading**: Use callbacks for dynamic content
- **Memory Management**: Clear unused data objects

## 🔄 Updates and Maintenance

### Updating Data
1. Replace the SQLite database file
2. Restart the dashboard application
3. Refresh browser to see new data

### Adding New Models
1. Update model results JSON files
2. Modify validation results structure
3. Add new model visualizations
4. Update performance comparison charts

### Version Control
- Dashboard code is version controlled
- Configuration files included
- Sample data for development
- Documentation updates with releases

## 📞 Support

### Getting Help
- Check the troubleshooting section above
- Review error logs in console output
- Use `--debug` flag for detailed error messages
- Check network connectivity and firewall settings

### Feature Requests
- Create detailed requirements documentation
- Consider impact on existing functionality
- Test with sample data first
- Maintain backward compatibility

---

*Created for Financial Risk Governance BI System*  
*Last Updated: September 29, 2024*