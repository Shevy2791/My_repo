import pyodbc
from sqlalchemy import create_engine, text
import urllib
import pandas as pd

# Define parameters
params = urllib.parse.quote_plus(
    "DRIVER={ODBC Driver 18 for SQL Server};"
    "SERVER=PAUL-LAPTOP;"
    "DATABASE=AdventureWorks2022;"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)

# Create connection string
connection_string = f"mssql+pyodbc:///?odbc_connect={params}"

# Create engine
engine = create_engine(connection_string)

# Connect to database
with engine.connect() as connection:
    # Begin a transaction
    trans = connection.begin()

    # Define the SQL query using the text function
    query = text("SELECT JobTitle, Gender, VacationHours, SickleaveHours, LoginID FROM HumanResources.Employee ORDER BY VacationHours DESC;")

    df = pd.read_sql_query(query,connection)
    
    df.to_excel("query_results.xlsx", index=False)
    
    
    
