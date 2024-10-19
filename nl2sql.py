# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:00:20 2024

@author: anyas

"""
#%%
#initial imports
import pandas as pd
import openai
import sqlite3
import requests
#%%
openai.api_key = "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"

#database_path = r"C:\Users\anyas\Downloads\database\database\tracking_grants_for_research\schema.db"
database_path = r"C:\Users\anyas\Downloads\database\database\card_games\card_games.sqlite"

def connect_db():
    return sqlite3.connect(database_path)

def fetch_schema():
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema_details = {}

    for table in tables:
        table_name = table[0]
        print(f"Fetching details for table: {table_name}")

        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        columns = [(col[1], col[2]) for col in columns]  

        cursor.execute(f"PRAGMA table_info({table_name});")
        primary_keys = [col[1] for col in cursor.fetchall() if col[5] > 0]

        cursor.execute(f"PRAGMA foreign_key_list({table_name});")
        foreign_keys = [(fk[3], fk[2], fk[4]) for fk in cursor.fetchall()] 

        schema_details[table_name] = {
            "columns": columns,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys
        }

    conn.close()
    return schema_details
#%%
def execute_query(query):
    conn = connect_db()
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df
#%%
def translate_to_sql(natural_language_query, schema_details, model="gpt-4"):
    schema_str = "Here is the schema of the database:\n\n"
    for table, details in schema_details.items():
        schema_str += f"Table: {table}\n"
        schema_str += "Columns:\n"
        for column in details["columns"]:
            schema_str += f"  - {column[0]} ({column[1]})\n"
        schema_str += "Primary Keys:\n"
        for pk in details["primary_keys"]:
            schema_str += f"  - {pk}\n"
        schema_str += "Foreign Keys:\n"
        for fk in details["foreign_keys"]:
            schema_str += f"  - {fk[0]} references {fk[1]}({fk[2]})\n"
        schema_str += "\n"
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert SQL translator."},
            {"role": "user", "content": "Only complete the SQL query and provide no explanation, and do not select extra columns that are not explicitly requested by the query."},
            {"role": "user", "content": f"{schema_str}"},
            {"role": "user", "content": f"Translate the following natural language query to SQL: {natural_language_query}. SELECT"},
        ],
        temperature = 0.0,
        max_tokens=150
    )
    sql_query = "SELECT " + response['choices'][0]['message']['content'].strip()
    return sql_query

import json

def fetch_queries():
    
    json_file_path = r"C:\Users\anyas\Downloads\dev.json"

    with open(json_file_path, mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    schema_details = fetch_schema()
    
    results = []
    gold_results = []
    gold_queries = []
    
    for i in range(340, 450):
        query = data[i].get('question')
        evidence = data[i].get('evidence')
        gold_query = data[i].get('SQL')
        sql_query = translate_to_sql(query, schema_details, model="gpt-4")
        
        try:
            result = execute_query(sql_query)
            result_str = result.to_string(index=False)
        except Exception as e:
            result_str = f"An error occurred: {e}"
        
        results.append({
                "Natural Language Query": query,
                "Generated SQL Query": sql_query,
                "Result": result_str
            })
        
        try:
            result = execute_query(gold_query)
            result_str = result.to_string(index=False)
        except Exception as e:
            result_str = f"An error occurred: {e}"
        
        gold_results.append(result_str)
        gold_queries.append(gold_query)
    return results, gold_results, gold_queries

results, gold_results, gold_queries = fetch_queries()
#%%
llm_results_df = pd.DataFrame(results)

        
    # url = "https://datasets-server.huggingface.co/rows?dataset=xlangai%2Fspider&config=spider&split=train&offset=4320&length=78"
    # response = requests.get(url)
    # if response.status_code == 200:
    #     data = response.json()
    #     queries = [row['row']['question'] for row in data['rows']]
    #     gold_queries = [row['row']['query'] for row in data['rows']]
    #     return queries, gold_queries
    # else:
    #     raise Exception(f"Failed to fetch queries: {response.status_code}")

# for query in queries:
#     sql_query = translate_to_sql(query, schema_details, model="gpt-4")

#     try:
#         result = execute_query(sql_query)
#         result_str = result.to_string(index=False)
#     except Exception as e:
#         result_str = f"An error occurred: {e}"

#     results.append({
#         "Natural Language Query": query,
#         "Generated SQL Query": sql_query,
#         "Result": result_str
#     })

# llm_results_df = pd.DataFrame(results)

# gold_results = []

# for query in gold_queries:
    
#     try:
#         result = execute_query(query)
#         result_str = result.to_string(index=False)
#     except Exception as e:
#         result_str = f"An error occurred: {e}"

#     gold_results.append(result_str)
#%%
## Print out incorrect results
for i in range(70):
    if (llm_results_df['Result'][i]).lower() != (gold_results[i]).lower():
        print(llm_results_df['Result'][i])
        print(gold_results[i])
        print(llm_results_df['Natural Language Query'][i])
        print(llm_results_df['Generated SQL Query'][i])
        print(gold_queries[i])
        print('')
#%%
print(llm_results_df['Generated SQL Query'][65])
#%%
## Store queries so that they can be used again.
gold_queries_df = pd.DataFrame(gold_queries, columns=["Gold Queries"])

gold_queries_df.to_csv(r"C:\Users\anyas\Desktop\Summer Project\SQL CSVs\grants_gold.csv")
llm_results_df.to_csv(r"C:\Users\anyas\Desktop\Summer Project\SQL CSVs\grants_llm.csv")
#%%
print(execute_query("SELECT Projects.project_id, Projects.project_details\nFROM Projects\nLEFT JOIN Project_Outcomes ON Projects.project_id = Project_Outcomes.project_id\nWHERE Project_Outcomes.project_id IS NULL;"))
#%%
print(execute_query("SELECT transcript_id, transcript_date\nFROM Transcripts\nWHERE transcript_id = (\n  SELECT transcript_id\n  FROM Transcript_Contents\n  GROUP BY transcript_id\n  ORDER BY COUNT(*) ASC\n  LIMIT 1\n)"))
#%%
print(execute_query("SELECT T1.name ,  T1.id FROM station AS T1 JOIN status AS T2 ON T1.id  =  T2.station_id GROUP BY T2.station_id HAVING avg(T2.bikes_available)  >  14 UNION SELECT name ,  id FROM station WHERE installation_date LIKE \"12/%\""))