# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:47:16 2024

@author: anyas
"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
#%%
df = pd.read_csv(r"C:\Users\anyas\Desktop\Summer Project\Correlation CSVs\job_descriptions.csv")
#%%
import re
def extract_salary_range(salary_range):
    match = re.findall(r'\d+', salary_range.replace(',', ''))
    if match:
        return int(match[0]), int(match[1])
    return None, None

df[['lowest_salary', 'highest_salary']] = df['Salary Range'].apply(lambda x: pd.Series(extract_salary_range(x)))

print(df)
## go through and get the highest and lowest salary
## then take the average of thsoe two and call that the actual salary
## then take the first 2000 rows only
#%%
new_df = df.head(3000)

new_df['Salary'] = (new_df['lowest_salary'] + new_df['highest_salary']) / 2
new_df.drop(columns={'lowest_salary', 'highest_salary', 'Salary Range'}, inplace=True)
#%%
new_df.to_csv(r"C:\Users\anyas\Desktop\Summer Project\Correlation CSVs\job_descriptions.csv")