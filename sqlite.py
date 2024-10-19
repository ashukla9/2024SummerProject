import os
import sqlite3

database_folder = r"C:\Users\anyas\Downloads\database\database"

def execute_schema_file(database_path, file_path):
    db_dir = os.path.dirname(database_path)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    conn = sqlite3.connect(database_path)
    print(f"Connected to SQLite database at {database_path}")
    
    with open(file_path, 'r', encoding='latin-1') as file:
        schema_sql = file.read()
        cursor = conn.cursor()
        cursor.executescript(schema_sql)
        conn.commit()
    
    conn.close()
    print(f"Executed schema file and closed database connection at {database_path}")

for subdir, _, files in os.walk(database_folder):
    for file in files:
        if file.endswith('.sqlite'):
            file_path = os.path.join(subdir, file)
            
            database_name = os.path.splitext(file)[0] + '.db'
            database_path = os.path.join(subdir, database_name)
            
            print(f"Executing schema file: {file_path}")
            try:
                execute_schema_file(database_path, file_path)
                print(f"Successfully executed: {file_path}")
            except Exception as e:
                print(f"Error executing {file_path}: {e}")