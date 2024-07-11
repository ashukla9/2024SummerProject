# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
#%%
#initial imports
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
#%%
#read in data
def read_csv_files_in_directory(directory_path):
    imputed_dfs = {}
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            
            df = pd.read_csv(file_path)
            
            df_name = os.path.splitext(filename)[0]
            imputed_dfs[df_name] = df
            print(f"Loaded {filename} as DataFrame: {df_name}")
    
    return imputed_dfs

directory = r"C:\Users\anyas\Desktop\Summer Project\Modified CSV Files"
imputed_dfs = read_csv_files_in_directory(directory)
#%%
# Values w/ VIF greater than 10 are collinear
# And will affect the model output

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

#merge them by country and year to create 2015 and 2010 dataframes
merged_df = list(imputed_dfs.values())[0]
for df in list(imputed_dfs.values())[1:]:
    merged_df = pd.merge(merged_df, df, on=['Country', 'Year'], how='inner')
#%%
merged_df = merged_df.drop(columns={'GDP in constant 2015 prices (millions of US dollars)'})
#%%
cmerged_df = merged_df.copy()
cmerged_df['Unemployed Difference'] = cmerged_df['Unemployment rate - Male'] - cmerged_df['Unemployment rate - Female']
cmerged_df['Total Employed Ratio'] = cmerged_df['Labour force participation - Total'] / cmerged_df['Population mid-year estimates (millions)']
cmerged_df['Economic Activity Index'] = cmerged_df['GDP real rates of growth (percent)'] + cmerged_df['Employment by industry: Services (%) Male and Female']
cmerged_df['Education and Workforce Index'] = (cmerged_df['Gross enrollment ratio - Primary (female)'] + cmerged_df['Gross enrollment ratio - Primary (male)'] + cmerged_df['Gross enrollment ratio - Lower secondary level (female)'] + cmerged_df['Gross enrollment ratio - Lower secondary level (male)'] + cmerged_df['Gross enrollment ratio - Upper secondary level (female)'] + cmerged_df['Gross enrollment ratio - Upper secondary level (male)']) / cmerged_df['Labour force participation - Total']
cmerged_df['Demographic Stability Index'] = cmerged_df['Life expectancy at birth for both sexes (years)'] / cmerged_df['Total fertility rate (children per women)'] + cmerged_df['Population annual rate of increase (percent)']
cmerged_df['Public Spending Efficiency Index'] = (cmerged_df['Public expenditure on education (% of GDP)'] + cmerged_df['Current health expenditure (% of GDP)']) / cmerged_df['GDP per capita (US dollars)']
cmerged_df['Migration Employment Impact'] = (cmerged_df['International migrant stock: Both sexes (% total population)'] + cmerged_df['Employment by industry: Agriculture (%) Male and Female'] + cmerged_df['Employment by industry: Industry (%) Male and Female'] + cmerged_df['Employment by industry: Services (%) Male and Female']) / 4
cmerged_df['Gender Equality Index'] = (cmerged_df['Seats held by women in national parliament, as of February (%)'] + cmerged_df['Labour force participation - Female']) / 2
#%%
# Split the merged dataframe into 2015 and 2010 dataframes
df_2015 = merged_df[merged_df['Year'] == 2015].copy()
df_2010 = merged_df[merged_df['Year'] == 2010].copy()
cdf_2015 = cmerged_df[cmerged_df['Year'] == 2015].copy()
cdf_2010 = cmerged_df[cmerged_df['Year'] == 2010].copy()

df_2015 = sm.add_constant(df_2015)
df_2010 = sm.add_constant(df_2010)
cdf_2015 = sm.add_constant(cdf_2015)
cdf_2010 = sm.add_constant(cdf_2010)
#%%
target_column = 'GDP in current prices (millions of US dollars)'
df_2015.set_index(['Country', 'Year'], inplace=True)
df_2010.set_index(['Country', 'Year'], inplace=True)
cdf_2015.set_index(['Country', 'Year'], inplace=True)
cdf_2010.set_index(['Country', 'Year'], inplace=True)

def find_correlations(target_df, pred_df, threshold=0.5):
    
    correlations = pred_df.corrwith(target_df[target_column])
    correlations = correlations[(correlations.abs() > threshold)]
    print(correlations)
#%%
def calculate_vif(df):
    vif = pd.DataFrame()
    vif['Feature'] = df.columns
    vif['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif

def vif_delete(df):
    while True:
        vif = calculate_vif(df)
        max_vif = vif["VIF"].max()
        if max_vif > 10:
            feature_to_drop = vif.loc[vif["VIF"].idxmax(), "Feature"]
            print(f"Dropping feature '{feature_to_drop}' with VIF {max_vif}")
            df.drop(columns=[feature_to_drop], inplace=True)
        else:
            break
#%%
from sklearn.preprocessing import MinMaxScaler

def scale_continuous(df):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = MinMaxScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df
#%%
## What variables are most significant in determining a country’s GDP? ##
## Suppose a country wanted to improve its GDP. What variables should it focus on? ##
# Multivariate Linear Regression example

import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def lin_reg(df, target_df):
    target_column = 'GDP in current prices (millions of US dollars)'
    X = df
    y = target_df[target_column]
    # X = sm.add_constant(X) 
    ## CHECK: add constant here or before vif
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = sm.OLS(y_train, X_train).fit()
    
    y_pred = model.predict(X_test)
    
    summary = model.summary()
    print(summary)
    
    print('')
    coefficients = model.params
    abs_coefficients = coefficients.abs()
    
    sorted_indices = abs_coefficients.sort_values(ascending=False).index
    
    top_10_features = coefficients.loc[sorted_indices].head(10)

    print("Top 10 features most associated with the target outcome based on absolute value of coefficient:")
    print(top_10_features)
#%%
def lin_reg_aic(df):
    target_column = 'GDP in current prices (millions of US dollars)'
    predictors = df.columns.drop(target_column)
    
    def calculate_aic(df, target_column):
        best_aic = float('inf')
        best_model = None
        best_features = None
        
        # Iterate over all combinations of predictors
        # best model uses 2 predictors
        # takes a long time to run over all the features
        # CHANGE VALUE
        for i in range(1, 5):
            for combo in itertools.combinations(predictors, i):
                X = df[list(combo)]
                y = df[target_column]
                
                model = sm.OLS(y, X).fit()
                aic = model.aic
                
                if aic < best_aic:
                    best_aic = aic
                    best_model = model
                    best_features = combo
    
        return best_model, best_aic, best_features
    
    best_model, best_aic, best_features = calculate_aic(df, target_column)
    
    print(f"Best model features: {best_features}")
    print(f"Best AIC: {best_aic}")
    if best_model:
        print(best_model.summary())
    else:
        print("No valid model found.")
#%%
## Direct variables ##
print('2015:')
find_correlations(df_2015, df_2015)
print('')
print('2010:')
find_correlations(df_2010, df_2010)
#%%
X_2015 = df_2015.drop(target_column, axis=1)
X_2010 = df_2010.drop(target_column, axis=1)
vif_delete(X_2015)
vif_delete(X_2010)
#%%
df_2015 = scale_continuous(df_2015)
df_2010 = scale_continuous(df_2010)
#%%
print('2015:')
lin_reg(df_2015[X_2015.columns], df_2015)
print('')
print('2010:')
lin_reg(df_2010[X_2010.columns], df_2010)
#%%
lin_reg_aic(df_2015)
#%%
## Indirect variables ##
print('2015:')
correlations = find_correlations(cdf_2015, cdf_2015)
print('')
print('2010:')
correlations = find_correlations(cdf_2010, cdf_2010)
#%%
cX_2015 = cdf_2015.drop(target_column, axis=1)
cX_2010 = cdf_2010.drop(target_column, axis=1)
vif_delete(cX_2015)
vif_delete(cX_2010)
#%%
cdf_2015 = scale_continuous(cdf_2015)
cdf_2010 = scale_continuous(cdf_2010)
#%%
print('2015:')
lin_reg(cdf_2015[cX_2015.columns], cdf_2015)
print('')
print('2010:')
lin_reg(cdf_2010[cX_2010.columns], cdf_2010)