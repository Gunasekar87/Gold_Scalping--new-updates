import sqlite3
import pandas as pd

try:
    conn = sqlite3.connect('data/market_memory.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print("Tables:", tables)
    
    for table in tables:
        table_name = table[0]
        print(f"\n--- {table_name} ---")
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        for col in columns:
            print(col)
            
    conn.close()
except Exception as e:
    print(f"Error: {e}")
