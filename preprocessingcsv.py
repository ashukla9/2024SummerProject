# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
#%%
#initial imports
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
#%%
#data reading and preprocessing steps
file1 = r"C:\Users\anyas\Desktop\Summer Project\SYB66_325_202310_Expenditure on health.csv"
file2 = r"C:\Users\anyas\Desktop\Summer Project\SYB66_200_202310_Employment.csv" 
file3 = r"C:\Users\anyas\Desktop\Summer Project\SYB66_230_202310_GDP and GDP Per Capita.csv" 
file4 = r"C:\Users\anyas\Desktop\Summer Project\SYB66_245_202310_Public expenditure on education and access to computers.csv" 
file5 = r"C:\Users\anyas\Desktop\Summer Project\SYB66_246_202310_Population Growth, Fertility and Mortality Indicators.csv"
file6 = r"C:\Users\anyas\Desktop\Summer Project\SYB66_309_202310_Education.csv"
file7 = r"C:\Users\anyas\Desktop\Summer Project\SYB66_317_202310_Seats held by women in Parliament.csv" 
file8 = r"C:\Users\anyas\Desktop\Summer Project\SYB66_327_202310_International Migrants and Refugees.csv"  
file9 = r"C:\Users\anyas\Desktop\Summer Project\SYB66_329_202310_Labour Force and Unemployment.csv"  
file10 = r"C:\Users\anyas\Desktop\Summer Project\SYB66_328_202310_Intentional homicides and other crimes.csv"  

health_spending = pd.read_csv(file1, encoding = 'ISO-8859-1')
employment = pd.read_csv(file2, encoding = 'ISO-8859-1')
gdp = pd.read_csv(file3, encoding = 'ISO-8859-1')
education_spending = pd.read_csv(file4, encoding = 'ISO-8859-1')
population_growth = pd.read_csv(file5, encoding = 'ISO-8859-1')
education = pd.read_csv(file6, encoding = 'ISO-8859-1')
women_gov = pd.read_csv(file7, encoding = 'ISO-8859-1')
migrants = pd.read_csv(file8, encoding = 'ISO-8859-1')
unemployment = pd.read_csv(file9, encoding = 'ISO-8859-1')
homicides = pd.read_csv(file10, encoding = 'ISO-8859-1')

datasets = [health_spending, employment, gdp, education_spending, population_growth,
            education, women_gov, migrants, unemployment, homicides]

transformed_datasets = []

for dataset in datasets:
    new_header = dataset.iloc[0]
    dataset.columns = new_header
    dataset = dataset[1:]  # Drop the first row
    dataset = dataset.reset_index(drop=True)  # Reset index
    dataset = dataset.rename(columns={np.nan: 'Country'})
    dataset = dataset.drop(columns=['Region/Country/Area','Footnotes', 'Source'])
    dataset = dataset.pivot(index=['Country', 'Year'], columns='Series', values='Value').reset_index()
    transformed_datasets.append(dataset)
    
print(transformed_datasets[0].columns)

#%%
merged_df = transformed_datasets[0]
    
for df in transformed_datasets[1:]:
    merged_df = pd.merge(merged_df, df, on=['Country', 'Year'], how='inner')
#%%
print(merged_df.columns)
merged_df = merged_df.drop(columns={'Kidnapping at the national level, rate per 100,000',
'Percentage of male and female intentional homicide victims, Female',
'Percentage of male and female intentional homicide victims, Male',
'Theft at the national level, rate per 100,000 population',
'Total Sexual Violence at the national level, rate per 100,000', 'Basic access to computers by level of education: Lower secondary',
'Basic access to computers by level of education: Primary',
'Basic access to computers by level of education: Upper secondary'})
#%%
#data imputation
#If countries had data in one year but not the other, imputed by
#Filling in the value for both years
merged_df['Current health expenditure (% of GDP)'].fillna(6.2, inplace=True)

def conditional_fill(group, value):
    if group[value].notna().any():
        group[value] = group[value].ffill().bfill()
    return group

def clean_and_convert_to_float(value):
    if isinstance(value, str):
        # Remove commas and convert to float
        return float(value.replace(',', ''))
    else:
        return value 

for column in merged_df.columns:
    merged_df = merged_df.groupby('Country').apply(lambda group: conditional_fill(group, column))
    if column != 'Country':
        merged_df[column] = merged_df[column].apply(clean_and_convert_to_float)
#%%
#Then used KNN clustering imputation to impute average values of
#Similar countries. Attempted MICE imputation but results were skewed.
from sklearn.impute import KNNImputer

countries = merged_df['Country']
numeric_data = merged_df.drop(columns=['Country'])

imputer = KNNImputer(n_neighbors=2)

imputed_data = imputer.fit_transform(numeric_data)
imputed_df = pd.DataFrame(imputed_data, columns=numeric_data.columns)
imputed_df.insert(0, 'Country', countries)
#%%
import os

def split_dataframe_to_csv(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    variable_columns = [col for col in df.columns if col not in ['Country', 'Year']]
    
    for column in variable_columns:
        df_variable = df[['Country', 'Year', column]]
        
        file_name = f"{column.replace(' ', '_').replace('%', 'percent').replace('/', '_per_').replace(':', '_').replace('(', '').replace(')', '')}.csv"
        file_path = os.path.join(output_dir, file_name)
        
        df_variable.to_csv(file_path, index=False)
        print(f"Saved {file_path}")

split_dataframe_to_csv(imputed_df, r"C:\Users\anyas\Desktop\Summer Project\Modified CSV Files")