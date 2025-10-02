#!/usr/bin/env python
"""
Dashboard Launcher Script

This script provides a convenient way to launch the Financial Risk Governance BI Dashboard
with various configuration options.

Usage:
    python run_dashboard.py [options]

Options:
    --host HOST     Host to run the dashboard on (default: 127.0.0.1)
    --port PORT     Port to run the dashboard on (default: 8050)
    --debug         Run in debug mode (default: False)
    --help          Show this help message

Example:
    python run_dashboard.py --host 0.0.0.0 --port 8080 --debug
"""

import argparse
import sys
import os
import logging

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from dashboards.financial_bi_dashboard import FinancialDashboard
except ImportError as e:
    print(f"Error importing dashboard: {e}")
    print("Please ensure all required packages are installed:")
    print("pip install dash plotly pandas numpy sqlite3")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(
        description='Financial Risk Governance BI Dashboard Launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_dashboard.py                    # Run with default settings
  python run_dashboard.py --debug            # Run in debug mode
  python run_dashboard.py --port 8080        # Run on port 8080
  python run_dashboard.py --host 0.0.0.0     # Run on all interfaces
        """
    )
    
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host to run the dashboard on (default: 127.0.0.1)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8050,
        help='Port to run the dashboard on (default: 8050)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode (enables auto-reload and detailed error messages)'
    )
    
    parser.add_argument(
        '--db-path',
        default='data/gfd_database.db',
        help='Path to the SQLite database (default: data/gfd_database.db)'
    )
    
    args = parser.parse_args()
    
    # Print startup information
    print("=" * 70)
    print("🏦 FINANCIAL RISK GOVERNANCE BI DASHBOARD")
    print("=" * 70)
    print(f"🌐 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"🐛 Debug Mode: {'ON' if args.debug else 'OFF'}")
    print(f"💾 Database: {args.db_path}")
    print("=" * 70)
    print()
    print("📊 Dashboard Features:")
    print("   • 📊 Overview - Key metrics and performance indicators")
    print("   • 🌍 Geographic Analysis - World map and country comparisons")
    print("   • 📈 Time Series Analysis - Trends and component analysis")
    print("   • 🤖 Model Performance - ML model validation and results")
    print("   • ⚠️ Risk Assessment - Risk factors and country profiles")
    print("   • 🔍 Data Quality - Missing data and quality metrics")
    print()
    print("🚀 Starting dashboard...")
    print(f"💡 Access dashboard at: http://{args.host}:{args.port}")
    print("   Press Ctrl+C to stop the server")
    print("=" * 70)
    
    try:
        # Initialize dashboard
        dashboard = FinancialDashboard(db_path=args.db_path)
        
        # Run the dashboard
        dashboard.run(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
        
    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("🛑 Dashboard stopped by user")
        print("=" * 70)
    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}")
        print(f"\n❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Ensure all required packages are installed:")
        print("      pip install dash plotly pandas numpy sqlite3")
        print("   2. Check if the database file exists")
        print("   3. Verify port is not in use")
        print("   4. Try running with --debug for more details")
        sys.exit(1)

if __name__ == "__main__":
    main()