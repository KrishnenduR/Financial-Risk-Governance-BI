import sqlite3
import os

db_path = 'data/gfd_database.db'

if os.path.exists(db_path):
    print(f"Database found at: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables: {tables}")
    
    # Check financial_development_data table
    if ('financial_development_data',) in tables:
        cursor.execute('SELECT COUNT(*) FROM financial_development_data')
        count = cursor.fetchone()[0]
        print(f"Records in financial_development_data: {count}")
        
        # Get column info
        cursor.execute("PRAGMA table_info(financial_development_data)")
        columns = cursor.fetchall()
        print(f"Columns: {len(columns)}")
        for col in columns:  # All columns
            print(f"  - {col[1]} ({col[2]})")
        
        # Get sample data
        cursor.execute('SELECT * FROM financial_development_data LIMIT 1')
        sample = cursor.fetchone()
        print(f"Sample record keys: {len(sample) if sample else 0} values")
    else:
        print("financial_development_data table not found")
    
    conn.close()
else:
    print(f"Database not found at: {db_path}")