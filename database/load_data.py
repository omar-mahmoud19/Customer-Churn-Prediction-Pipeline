import pyodbc
import pandas as pd

# Path
CSV_PATH = r"D:\project\data\processed\clean_data.csv"

# Connection
conn = pyodbc.connect(
    "DRIVER={SQL Server};"
    "SERVER=OMAR-ZIZO;"
    "DATABASE=customer_churn;"
    "Trusted_Connection=yes;"
)
cursor = conn.cursor()

# Load CSV
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows from CSV")

# Insert data
cols = ", ".join(df.columns)
placeholders = ", ".join(["?" for _ in df.columns])
insert_query = f"INSERT INTO customers ({cols}) VALUES ({placeholders})"

for _, row in df.iterrows():
    cursor.execute(insert_query, tuple(row))

conn.commit()
print("Data inserted successfully!")

# Quick check
cursor.execute("SELECT COUNT(*) FROM customers")
result = cursor.fetchone()
print(f"Rows in DB: {result[0]}")

conn.close()