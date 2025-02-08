# -*- coding: utf-8 -*-
"""
Created on Mon Aug 5

@author: anyas

"""
#%%
#initial imports
import pandas as pd
import openai
import sqlite3
import requests
openai.api_key = "XYZ"
#%%
database_path = r"C:\Users\anyas\Downloads\database\database\episodes\schema.db"

def connect_db():
    return sqlite3.connect(database_path)

## GET SCHEMAS FOR EACH DATABASE ##
def fetch_schema():
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema_details = {}

    for table in tables:
        table_name = table[0]

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
    
    file_path = r"C:\Users\anyas\Desktop\Summer Project\SQL CSVs\episodes.txt"
    with open(file_path, "w") as file:
        file.write(schema_str)
    
    return schema_details

fetch_schema()
#%%
## CONVERT BENCHMARK CSV TO JSON ##
import csv
import json
import random

csv_file_path = r"C:\Users\anyas\Desktop\Summer Project\SQL CSVs\Benchmark_Official.csv"

json_file_path = r"C:\Users\anyas\Desktop\Summer Project\SQL CSVs\Benchmark_Official.json"

data = []
with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        data.append(row)
        
random.shuffle(data)

with open(json_file_path, mode='w', encoding='utf-8') as json_file:
    json.dump(data, json_file, indent = 4)

print(f"CSV data has been successfully converted to JSON and saved to {json_file_path}")
#%%
import json
from openai import OpenAI
client = OpenAI(api_key= "XYZ")

##EXECUTE QUERY ON DATASET ##
def execute_query(database, query):
    if database == 'california_schools' or database == 'card_games':
        database_path = "C:\\Users\\anyas\\Downloads\\database\\database\\" + database + "\\" + database + ".sqlite"
    else:
        database_path = "C:\\Users\\anyas\\Downloads\\database\\database\\" + database + "\\schema.db"
    
    conn = sqlite3.connect(database_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

## PASS THE NL QUERY INTO LLM AND ASK TO RETURN WHETHER QUERY IS CORRECT OR INCORRECT
## BASELINE VARIABLE REPRESENTS DIFFERENT PROMPT ENGINEERING APPROACHES
## JUST FOR EXPERIMENTING - BEST PROMPTS ARE DOWN BELOW ##
def translate_to_sql(database, natural_language_query, llm_sql, schema_str, baseline, model):
    
    if baseline=="baseline":
        messages=[
            {"role": "system", "content": "You are an expert SQL translator and verifier."},
            {"role": "system", "content": "First, determine whether the SQL query is correct or incorrect and provide no explanation. Second, if the query is incorrect, also provide an explanation as to why."},
            {"role": "user", "content": f"Here is the schema of the database: {schema_str}"},
            {"role": "user", "content": f"Here is a natural language query {natural_language_query}. It has been translated to SQL as follows: {llm_sql}. TRUE or FALSE: does the SQL response correctly answer the natural language query?"},
        ]
    elif baseline=="give_results":
        try:
            result = execute_query(database, llm_sql)
            result_str = result.to_string(index=False)
            truncated_result_str = result_str[:500]
        except Exception as e:
            truncated_result_str = f"An error occurred: {e}"
        messages=[
            {"role": "system", "content": "You are an expert SQL translator and verifier."},
            {"role": "system", "content": "First, determine whether the SQL query is correct or incorrect. Second, provide an explanation as to why."},
            {"role": "user", "content": f"Here is the schema of the database: {schema_str}"},
            {"role": "user", "content": f"Here is a natural language query {natural_language_query}. It has been translated to SQL as follows: {llm_sql}. Here is the first 500 characters of the SQL query result when executed against the database: {truncated_result_str}."},
            {"role": "user", "content": "TRUE or FALSE (and give an explanation): does the SQL response correctly answer the natural language query?"}
        ]
    elif baseline=="few_shot":
        database_path_1 = "C:\\Users\\anyas\\Desktop\\Summer Project\\SQL CSVs\\concert_singer.txt"
        database_path_2 = "C:\\Users\\anyas\\Desktop\\Summer Project\\SQL CSVs\\bike_1.txt"
        with open(database_path_1, 'r', encoding='utf-8') as file:
            schema_str_1 = file.read()
        with open(database_path_2, 'r', encoding='utf-8') as file:
            schema_str_2 = file.read()
        messages=[
            {"role": "system", "content": "You are an expert SQL translator and verifier."},
            {"role": "system", "content": "First, determine whether the SQL query is correct or incorrect. Second, provide an explanation as to why."},
            
            {"role": "system", "content": f"Here is the schema of the database: {schema_str_1}."},
            {"role": "system", "content": "Here is a natural language query: For each stadium, how many concerts play there? It has been translated to SQL as follows: SELECT T2.name ,  count(*) FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id GROUP BY T1.stadium_id."},
            {"role": "assistant", "content": "TRUE. This SQL query correctly returns the stadium names and concert counts by first joining the concert and stadium dataframes and then grouping by the stadium ID. As the stadium ID corresponds to the stadium name, this correctly provides the concert counts for each stadium."},
            
            {"role": "system", "content": f"Here is the schema of the database: {schema_str_1}."},
            {"role": "system", "content": "Here is a natural language query: What are the days that had the smallest temperature range, and what was that range? It has been translated to SQL as follows: SELECT date, (max_temperature_f - min_temperature_f) AS temperature_range FROM weather ORDER BY temperature_range ASC."},
            {"role": "assistant", "content": "FALSE. This SQL query correctly orders each day by the temperature range, but it does not limit the number of days in any way. This results in all the days being returned. The query should first find the smallest temperature range in the dataset, then return only the days with that temperature range. "},
            
            {"role": "user", "content": f"Here is the schema of the database: {schema_str}"},
            {"role": "user", "content": f"Here is a natural language query {natural_language_query}. It has been translated to SQL as follows: {llm_sql}."},
            {"role": "user", "content": "TRUE or FALSE (and give an explanation): does the SQL response correctly answer the natural language query?"}
        ]
    elif baseline=="hints":
        messages=[
            {"role": "system", "content": "You are an expert SQL translator and verifier."},
            {"role": "system", "content": "First, determine whether the SQL query is correct or incorrect. Second, provide an explanation as to why."},
            {"role": "system", "content": "Here are some tips: 1. If a subquery that also contains “LIMIT 1”, this means the subquery will only return one value and WHERE, =, IN will work with that subquery. 2. For the bike_1 dataset, each station can have multiple statuses, and this might affect any JOINs. 3. For the student_transcripts_tracking dataset, each degree program id corresponds to the name of the degree (i.e. Mathematics), not the degree program (Bachelors/Masters). Two different degree program ids can both be Bachelors programs."},
            {"role": "user", "content": f"Here is the schema of the database: {schema_str}"},
            {"role": "user", "content": f"Here is a natural language query {natural_language_query}. It has been translated to SQL as follows: {llm_sql}."},
            {"role": "user", "content": "TRUE or FALSE (and give an explanation): does the SQL response correctly answer the natural language query?"}
        ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature = 0.0,
        max_tokens=150
    )
    
    sql_query = response.choices[0].message.content
    return sql_query
#%%
## ASK LLM TO THINK ABOUT WHAT THIS QUERY IS QUERYING FOR BEFORE DETERMINING IF ANSWER IS TRUE OR FALSE
## JUST FOR EXPERIMENTING - BEST PROMPTS ARE DOWN BELOW ##
def conversation_to_sql(database, natural_language_query, llm_sql, schema_str, model):
    messages = [
        {"role": "system", "content": "You are an expert SQL translator and verifier."},
        {"role": "user", "content": f"Here is the schema of the database: {schema_str}"},
        {"role": "user", "content": f"Here is a SQL query: {llm_sql}. What is this SQL query querying for?"},
    ]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature = 0.0,
        max_tokens=150
    )
    
    query_result = response['choices'][0]['message']['content'].strip()
    messages.append({"role": "assistant", "content": query_result})
    messages.append({"role": "assistant", "content": f"Here is a natural language question {natural_language_query}. TRUE or FALSE: does the given SQL query match the natural language question? Explain your response."})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature = 0.0,
        max_tokens=150
    )
    
    sql_query = response.choices[0].message.content
    return query_result, sql_query
#%%
## EXTRACT SQL FROM LLM RESPONSE ##
def extract_sql_code(text):
    start_marker = '```sql'
    start_pos = text.find(start_marker)
    
    if start_pos == -1:
        return None
    
    start_pos += len(start_marker)
    
    end_marker = '```'
    end_pos = text.find(end_marker, start_pos)
    
    if end_pos == -1:
        return None
    
    return text[start_pos:end_pos].strip()
#%%
## BEST PROMPT FOR GPT-4O WHEN ASKING IT TO DETERMINE WHETHER A SQL QUERY IS CORRECT OR INCORRECT ##
def gpt_4o_optimization(database, natural_language_query, llm_sql, schema_str, model="gpt-4o"):
    try:
        result = execute_query(database, llm_sql)
        result_str = result.to_string(index=False)
        truncated_result_str = result_str[:500]
    except Exception as e:
        truncated_result_str = f"An error occurred: {e}" 
    messages=[
        {"role": "system", "content": "You are an expert SQL translator and verifier."},
        {"role": "system", "content": "First, determine whether the SQL query is correct or incorrect. If it is incorrect, provide only the corrected SQL query."},
        {"role": "system", "content": "The following is very important: my career depends on it. Assume all SQL queries are syntactically correct but may or may not be logically correct. All queries are written for SQLite."},
        {"role": "user", "content": f"{schema_str}"},
        {"role": "user", "content": f"Here is a natural language query: {natural_language_query}. It has been translated to SQL as follows: {llm_sql}. Here is the first 500 characters of the SQL query when executed against the database: {truncated_result_str}. "},
        {"role": "user", "content": "TRUE or FALSE (and give an explanation): does the SQL response correctly answer the natural language query?"}
    ]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature = 0.0,
        max_tokens=150
    )
    
    sql_query = response.choices[0].message.content
    sql_result = extract_sql_code(sql_query)
    if sql_result is not None and 'FALSE' in sql_query:
        try:
            second_result = execute_query(database, sql_result)
            second_result_str = second_result.to_string(index=False)
            second_truncated_result_str = second_result_str[:500]
        except Exception as e:
            second_truncated_result_str = f"An error occurred: {e}" 
        messages.append({"role": "assistant", "content": sql_query})
        messages.append({"role": "assistant", "content": f"Here is the first 500 characters of your SQL query when executed against the database: {second_truncated_result_str}. Now with this updated information, TRUE or FALSE (and if FALSE give an explanation): did the ORIGINAL SQL query correctly answer the natural language query?"})
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature = 0.0,
            max_tokens=150
        )
    
        sql_query = response['choices'][0]['message']['content'].strip()
    return sql_query
#%%
## BEST PROMPT FOR GPT-4 WHEN ASKING IT TO DETERMINE WHETHER A SQL QUERY IS CORRECT OR INCORRECT ##
def gpt_4_optimization(database, natural_language_query, llm_sql, schema_str, row_str, model="gpt-4"):
    try:
        result = execute_query(database, llm_sql)
        result_str = result.to_string(index=False)
        truncated_result_str = result_str[:500]
    except Exception as e:
        truncated_result_str = f"An error occurred: {e}" 
    messages=[
        {"role": "system", "content": "You are an expert SQL translator and verifier."},
        {"role": "system", "content": "First, determine whether the SQL query is correct or incorrect. Second, provide an explanation as to why."},
        {"role": "system", "content": "The following is very important: my career depends on it. Assume all SQL queries are syntactically correct but may or may not be logically correct."},
         {"role": "user", "content": f"{schema_str}"},
        {"role": "user", "content": f"Here is a natural language query {natural_language_query}. It has been translated to SQL as follows: {llm_sql}. Here is the first 500 characters of the SQL query when executed against the database: {truncated_result_str}. "},
        {"role": "user", "content": "TRUE or FALSE (and give an explanation): does the SQL response correctly answer the natural language query?"}
    ]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature = 0.0,
        max_tokens=150
    )
    
    sql_query = response.choices[0].message.content
    sql_result = extract_sql_code(sql_query)
    if sql_result is not None:
        try:
            second_result = execute_query(database, sql_result)
            second_result_str = second_result.to_string(index=False)
            second_truncated_result_str = second_result_str[:500]
        except Exception as e:
            second_truncated_result_str = f"An error occurred: {e}" 
        messages.append({"role": "assistant", "content": sql_query})
        messages.append({"role": "assistant", "content": f"Here is the first 500 characters of your SQL query when executed against the database: {second_truncated_result_str}. Keeping this in mind, TRUE or FALSE: did the original SQL query correctly answer the natural language query?"})
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature = 0.0,
            max_tokens=150
        )
    
        sql_query = response.choices[0].message.content
    return sql_query
#%%
## EXPERIMENTING WITH CODELLAMA ##
from openai import OpenAI

def code_llama(database, natural_language_query, llm_sql, schema_str):
    client = OpenAI(
    	base_url="https://wf488eypprk73r6t.us-east-1.aws.endpoints.huggingface.cloud/v1/", 
    	api_key="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" 
    )
    
    try:
        result = execute_query(database, llm_sql)
        result_str = result.to_string(index=False)
        truncated_result_str = result_str[:500]
    except Exception as e:
        truncated_result_str = f"An error occurred: {e}" 
        
    chat_completion = client.chat.completions.create(
    	model="tgi",
        
    	messages=[
                {"role": "system", "content": "You are an expert SQL translator and verifier."},
                {"role": "user", "content": f"{schema_str}. Here is a natural language query: {natural_language_query}. It has been translated to SQL as follows: {llm_sql}. TRUE or FALSE (and give an explanation): does the SQL response correctly answer the natural language query?"}
            ],
        temperature = 0.0,
        stream=True,
    	max_tokens=100
    )
    
    final_string = ""
    for message in chat_completion:
    	final_string += message.choices[0].delta.content
        
    return final_string
#%%
## RUNS LLM ON EVERY QUERY IN DATASET ##
results = []

json_file_path = r"C:\Users\anyas\Desktop\Summer Project\SQL CSVs\Benchmark_Official.json"

with open(json_file_path, mode='r', encoding='utf-8') as json_file:
    data = json.load(json_file)
    
for query in data:
    database = query.get('\ufeffDataset')
    natural_language = query.get('Natural Language Query')
    llm_query = query.get('LLM Query')
    database_path = "C:\\Users\\anyas\\Desktop\\Summer Project\\SQL CSVs\\" + database + ".txt"

    with open(database_path, 'r', encoding='utf-8') as file:
        schema_str = file.read()

    # sql_query = translate_to_sql(database, natural_language, llm_query, schema_str, baseline="baseline", model="gpt-4o")
    sql_query = gpt_4o_optimization(database, natural_language, llm_query, schema_str, model="gpt-4o")
    # sql_query = code_llama(database, natural_language, llm_query, schema_str)
    results.append({
        "Database": database,
        "Natural Language Query": query,
        "LLM SQL Query": llm_query,
        "LLM Explanation": sql_query
    })
#%%
## STORING DATASET TO PANDAS DATAFRAME ##
llm_results_df = pd.DataFrame(results)
#%%
## CLASSIFYING RESULTS AS CORRECT/INCORRECT, PRINTING RESULTS ##
llm_results_df['Correct'] = ""
for i in range(len(llm_results_df)):
    if 'false' in llm_results_df['LLM Explanation'][i].lower() or 'no' in llm_results_df['LLM Explanation'][i].lower() or 'does not answer' in llm_results_df['LLM Explanation'][i] or 'incorrect' in llm_results_df['LLM Explanation'][i]or 'does not accurately answer' in llm_results_df['LLM Explanation'][i]:
        llm_results_df['Correct'][i] = 'FALSE'
    elif 'true' in llm_results_df['LLM Explanation'][i].lower() or 'yes' in llm_results_df['LLM Explanation'][i].lower() or 'will correctly answer' in llm_results_df['LLM Explanation'][i] or 'correct' in llm_results_df['LLM Explanation'][i] or 'correctly answers' in llm_results_df['LLM Explanation'][i]:
        llm_results_df['Correct'][i] = 'TRUE'

correct = 0
false_positives = 0
false_negatives = 0

for i in range(len(llm_results_df)):
    if 'TRUE' in llm_results_df['Correct'][i]:
        llm_results_df['Correct'][i] = 'TRUE'
    elif 'FALSE' in llm_results_df['Correct'][i]:
        llm_results_df['Correct'][i] = 'FALSE'
        
    if llm_results_df['Correct'][i] == data[i]['Correct']:
        correct += 1
    elif llm_results_df['Correct'][i] == 'TRUE':
        false_positives += 1
    elif llm_results_df['Correct'][i] == 'FALSE':
        false_negatives += 1

print("Correct: ", correct, "/ 73")
print("False positives:", false_positives, "/ 73")
print("False negatives:", false_negatives, "/ 73")
print("Accuracy: ", correct/73)
#%%
## SEND RESULTS BACK TO FILE ##
json_file_path = r"C:\Users\anyas\Desktop\Summer Project\SQL CSVs\benchmark_gpt4o_6.json"

for i in range(len(llm_results_df)):
    data[i]['GPT Response'] = llm_results_df['Correct'][i]
    data[i]['LLM Explanation'] = llm_results_df['LLM Explanation'][i]
# Save the modified JSON data back to the file
with open(json_file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, indent=4)