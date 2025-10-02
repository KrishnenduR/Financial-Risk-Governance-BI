import pandas as pd

file_path = r"C:\Users\Krishnendu Rarhi\Downloads\Datasets\Global FInancial Dataset\20220909-global-financial-development-database.xlsx"

# Get sheet names
xl = pd.ExcelFile(file_path)
print("Available sheets:", xl.sheet_names)

# Load the first sheet to see structure
first_sheet = pd.read_excel(file_path, sheet_name=xl.sheet_names[0], nrows=10)
print(f"\nFirst few rows of '{xl.sheet_names[0]}':")
print(first_sheet.head())
print(f"\nColumns: {list(first_sheet.columns)}")