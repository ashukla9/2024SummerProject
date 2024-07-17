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
def find_correlations(target_df, pred_df, threshold=0.5):
    
    correlations = pred_df.corrwith(target_df[target_column])
    correlations = correlations[(correlations.abs() > threshold)]
    print(correlations)
#%%
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

def calculate_vif(df):
    vif = pd.DataFrame()
    vif['Feature'] = df.columns
    vif['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    print(vif)
    return vif

def vif_delete(df):
    df_with_const = sm.add_constant(df)
    while True:
        vif = calculate_vif(df_with_const)
        if vif['VIF'].max() <= 10:
            break
        # Sort by VIF in descending order to check each variable
        vif = vif.sort_values(by='VIF', ascending=False)
        for feature, v in zip(vif['Feature'], vif['VIF']):
            if v > 10 and feature != 'const':
                print(f"Dropping feature '{feature}' with VIF {v}")
                df_with_const.drop(columns=[feature], inplace=True)
                break
        else:
            break
#%%
from sklearn.decomposition import PCA

def pca(df, target_column):
    pca = PCA()
    X = df.drop(columns=target_column)
    pca.fit(X)
    
    # Get explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    
    # Plotting the scree plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(np.arange(1, len(explained_variance) + 1))
    plt.grid(True)
    plt.show()
#%%
## What variables are most significant in determining a country’s GDP? ##
## Suppose a country wanted to improve its GDP. What variables should it focus on? ##
# Multivariate Linear Regression example

import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import ElasticNet
from scipy.stats import uniform
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV

def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals)
    plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), colors='r')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residuals Plot for {model_name}')
    plt.show()

def lin_reg(df, target_df, target_column, n_components = None):
    X = df
    y = target_df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    
    train_indices = X_train.index
    test_indices = X_test.index

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    if n_components:
       pca = PCA(n_components=n_components)
       X_train_pca = pca.fit_transform(X_train)
       X_test_pca = pca.transform(X_test)
       
       X_train = pd.DataFrame(X_train_pca, index=train_indices)
       X_test = pd.DataFrame(X_test_pca, index=test_indices)
    else:
       X_train = pd.DataFrame(X_train, columns=X.columns, index=train_indices)
       X_test = pd.DataFrame(X_test, columns=X.columns, index=test_indices)
   
    
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    model = sm.OLS(y_train, X_train).fit()
    
    y_pred = model.predict(X_test)
    
    summary = model.summary()
    print(summary)
    print('')
    test_mse = mean_squared_error(y_test, y_pred)
    print('Test MSE: ', test_mse)
    print('')
    variance = np.var(y_test, ddof=1)
    print(f"Variance of the target column (GDP): {variance}")
    print('')
    test_r2 = r2_score(y_test, y_pred)
    print(f"Test R^2: {test_r2}")
    print('')
    # Example usage after fitting your models
    plot_residuals(y_test, y_pred, 'Linear Regression')
    
    if n_components is None:
        coefficients = model.params
        abs_coefficients = coefficients.abs()
        
        sorted_indices = abs_coefficients.sort_values(ascending=False).index
        
        top_10_features = coefficients.loc[sorted_indices].head(10)
    
        print("Top 10 features most associated with the target outcome based on absolute value of coefficient:")
        print(top_10_features)
    
def elastic_net(df, target_df, target_column):
    
    X = df.drop(columns={target_column})
    y = target_df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # elastic_net = ElasticNet(random_state=42)
    
    # param_grid = {
    #     'alpha': uniform(0.01, 10),
    #     'l1_ratio': uniform(0, 1), 
    #     'max_iter': [1000, 5000, 10000], 
    #     'tol': [1e-4, 1e-3, 1e-2] 
    # }
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    #     random_search = RandomizedSearchCV(estimator=elastic_net, param_distributions=param_grid, n_iter=250, cv=5, scoring='neg_mean_squared_error', random_state=42)
    #     random_search.fit(X_train, y_train)
    
        best_model = ElasticNetCV(cv=10, random_state=42, l1_ratio=[.1, .5, .7, .9, .95, .99, 1])
        best_model.fit(X_train, y_train)
        
    # best_model = random_search.best_estimator_
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    plot_residuals(y_test, y_test_pred, 'Elastic Net')

    coefficients = best_model.coef_
    intercept = best_model.intercept_
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    variance = np.var(y_test, ddof=1)
    
    feature_names = X.columns
    coef_series = pd.Series(coefficients, index=feature_names)
    
    top_10_features = coef_series.abs().sort_values(ascending=False).head(10)
    
    print("Top 10 features most associated with the target outcome based on absolute value of coefficient:")
    print(top_10_features)

    # Print summary statistics
    print("Elastic Net Summary Statistics")
    print("================================")
    # print(f"Best Parameters: {random_search.best_params_}")
    print(f"Intercept: {intercept}")
    print("Coefficients:")
    best_features = []
    for feature, coef in zip(X.columns, coefficients):
        if coef != 0:
            print(f"  {feature}: {coef}")
            best_features.append(feature)
    print(f"Train MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")
    print(f"Test variance: {variance}")
    print(f"Train R-squared: {train_r2}")
    print(f"Test R-squared: {test_r2}")

# #%%
# def lin_reg_aic(df):
#     target_column = 'GDP in current prices (millions of US dollars)'
#     predictors = df.columns.drop(target_column)
    
#     def calculate_aic(df, target_column):
#         best_aic = float('inf')
#         best_model = None
#         best_features = None
        
#         # Iterate over all combinations of predictors
#         # best model uses 2 predictors
#         # takes a long time to run over all the features
#         # CHANGE VALUE
#         for i in range(1, 5):
#             for combo in itertools.combinations(predictors, i):
#                 X = df[list(combo)]
#                 y = df[target_column]
                
#                 model = sm.OLS(y, X).fit()
#                 aic = model.aic
                
#                 if aic < best_aic:
#                     best_aic = aic
#                     best_model = model
#                     best_features = combo
    
#         return best_model, best_aic, best_features
    
#     best_model, best_aic, best_features = calculate_aic(df, target_column)
    
#     print(f"Best model features: {best_features}")
#     print(f"Best AIC: {best_aic}")
#     if best_model:
#         print(best_model.summary())
#     else:
#         print("No valid model found.")
#%%
from catboost import CatBoostRegressor

def catboost_predict(df, target_column):
    X = df.drop(columns={target_column})
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #Change loss function depending on classification type
    catboost_model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, random_seed=42, verbose=100)

    catboost_model.fit(X_train, y_train)

    # Step 5: Make predictions
    y_train_pred = catboost_model.predict(X_train)
    y_test_pred = catboost_model.predict(X_test)
        
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Print summary statistics
    print("\nCatBoost Regression Summary Statistics")
    print("========================================")
    print(f"Train MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")
    print(f"Train R-squared: {train_r2}")
    print(f"Test R-squared: {test_r2}")
    
#%%

## GDP DATA PREPROCESSING ##

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

#merge them by country and year to create 2015 and 2010 dataframes
merged_df = list(imputed_dfs.values())[0]
for df in list(imputed_dfs.values())[1:]:
    merged_df = pd.merge(merged_df, df, on=['Country', 'Year'], how='inner')

merged_df = merged_df.drop(columns={'GDP in constant 2015 prices (millions of US dollars)'})

cmerged_df = merged_df.copy()
cmerged_df['Unemployed Difference'] = cmerged_df['Unemployment rate - Male'] - cmerged_df['Unemployment rate - Female']
cmerged_df['Total Employed Ratio'] = cmerged_df['Labour force participation - Total'] / cmerged_df['Population mid-year estimates (millions)']
cmerged_df['Economic Activity Index'] = cmerged_df['GDP real rates of growth (percent)'] + cmerged_df['Employment by industry: Services (%) Male and Female']
cmerged_df['Education and Workforce Index'] = (cmerged_df['Gross enrollment ratio - Primary (female)'] + cmerged_df['Gross enrollment ratio - Primary (male)'] + cmerged_df['Gross enrollment ratio - Lower secondary level (female)'] + cmerged_df['Gross enrollment ratio - Lower secondary level (male)'] + cmerged_df['Gross enrollment ratio - Upper secondary level (female)'] + cmerged_df['Gross enrollment ratio - Upper secondary level (male)']) / cmerged_df['Labour force participation - Total']
cmerged_df['Demographic Stability Index'] = cmerged_df['Life expectancy at birth for both sexes (years)'] / cmerged_df['Total fertility rate (children per women)'] + cmerged_df['Population annual rate of increase (percent)']
cmerged_df['Public Spending Efficiency Index'] = (cmerged_df['Public expenditure on education (% of GDP)'] + cmerged_df['Current health expenditure (% of GDP)']) / cmerged_df['GDP per capita (US dollars)']
cmerged_df['Migration Employment Impact'] = (cmerged_df['International migrant stock: Both sexes (% total population)'] + cmerged_df['Employment by industry: Agriculture (%) Male and Female'] + cmerged_df['Employment by industry: Industry (%) Male and Female'] + cmerged_df['Employment by industry: Services (%) Male and Female']) / 4
cmerged_df['Gender Equality Index'] = (cmerged_df['Seats held by women in national parliament, as of February (%)'] + cmerged_df['Labour force participation - Female']) / 2

# Split the merged dataframe into 2015 and 2010 dataframes
df_2015 = merged_df[merged_df['Year'] == 2015].copy()
df_2010 = merged_df[merged_df['Year'] == 2010].copy()
cdf_2015 = cmerged_df[cmerged_df['Year'] == 2015].copy()
cdf_2010 = cmerged_df[cmerged_df['Year'] == 2010].copy()

df_2015.set_index(['Country', 'Year'], inplace=True)
df_2010.set_index(['Country', 'Year'], inplace=True)
cdf_2015.set_index(['Country', 'Year'], inplace=True)
cdf_2010.set_index(['Country', 'Year'], inplace=True)
#%%
## Direct variables ##
target_column = 'GDP in current prices (millions of US dollars)'
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
print('2015:')
lin_reg(df_2015[X_2015.columns], df_2015, 'GDP in current prices (millions of US dollars)')
print('')
#%%
print('2015 elastic net')
elastic_net(df_2015, df_2015, 'GDP in current prices (millions of US dollars)')
#%%
pca(df_2015, 'GDP in current prices (millions of US dollars)')
lin_reg(df_2015, df_2015, 'GDP in current prices (millions of US dollars)', n_components=3)
print('')
#%%
print('2010:')
lin_reg(df_2010[X_2010.columns], df_2010, 'GDP in current prices (millions of US dollars)')
# #%%
# lin_reg_aic(df_2015)
#%%
# Indirect variables ##
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
print('2015:')
lin_reg(cdf_2015[cX_2015.columns], cdf_2015, 'GDP in current prices (millions of US dollars)')
print('')
#%%
print('2015 elastic net')
print(elastic_net(cdf_2015, cdf_2015, 'GDP in current prices (millions of US dollars)'))
print('')
#%%
print('2010:')
lin_reg(cdf_2010[cX_2010.columns], cdf_2010, 'GDP in current prices (millions of US dollars)')
#%%
## AIRBNB DATA PREPROCESSING ##

listings = pd.read_csv(r"C:\Users\anyas\Downloads\listings.csv.gz", compression='gzip')
price = pd.read_csv(r"C:\Users\anyas\Downloads\listings (2).csv")
#%%
listings['host_response_rate'] = listings['host_response_rate'].str.rstrip('%').astype('float') / 100
listings['host_acceptance_rate'] = listings['host_acceptance_rate'].str.rstrip('%').astype('float') / 100
#%%
merged_df = pd.merge(listings, price, on='id', suffixes=('', '_duplicate'))

# Drop duplicate columns
for col in merged_df.columns:
    if col.endswith('_duplicate') and col[:-10] in merged_df.columns:
        if merged_df[col[:-10]].equals(merged_df[col]):
            merged_df.drop(columns=col, inplace=True)
#%%
airbnb_df = merged_df.select_dtypes(include=[int, float])
airbnb_df = airbnb_df.dropna(subset=['price_duplicate'])
mean_A = airbnb_df['host_response_rate'].mean()
mean_B = airbnb_df['host_acceptance_rate'].mean()
# Replace NaN values in column 'A' with the mean of column 'A'
airbnb_df['host_response_rate'].fillna(mean_A, inplace=True)
airbnb_df['host_acceptance_rate'].fillna(mean_B, inplace=True)
airbnb_df['beds'].fillna(0, inplace=True)
#deleting calculated_host_listings_shared_rooms as all values are 0... doesn't really tell us much about the target variable
airbnb_df.drop(columns={'calendar_updated', 'license', 'neighbourhood_group', 'neighbourhood_group_cleansed', 'scrape_id', 'calculated_host_listings_count_shared_rooms'}, inplace=True)
airbnb_df = airbnb_df.dropna(subset='review_scores_rating')
airbnb_df.rename(columns={'price_duplicate': 'price'}, inplace=True)
airbnb_df.drop(columns='price').to_csv(r'C:\Users\anyas\Desktop\Summer Project\listings1.csv', index=False)
airbnb_df.to_csv(r'C:\Users\anyas\Desktop\Summer Project\listings2.csv', columns=['id', 'price'], index=False)
#%%
## Adding indirect variables ##

cairbnb_df = airbnb_df.copy()
cairbnb_df['Rating per Review'] = cairbnb_df['review_scores_rating'] / cairbnb_df['number_of_reviews']
cairbnb_df['Location and Value Combo'] = cairbnb_df['review_scores_location'] * cairbnb_df['review_scores_value']
# don't expect this to do well as it's a linear combo of existing variables
cairbnb_df['Nights available'] = cairbnb_df['maximum_nights'] - cairbnb_df['minimum_nights']
cairbnb_df['Host Rating'] = cairbnb_df.groupby('host_id')['review_scores_rating'].transform('mean')
cairbnb_df['Host Availability 30'] = cairbnb_df.groupby('host_id')['availability_30'].transform('mean')
cairbnb_df['beds_bathrooms_sum'] = cairbnb_df['beds'] + cairbnb_df['bathrooms']
cairbnb_df['Host Average House'] = cairbnb_df.groupby('host_id')['beds_bathrooms_sum'].transform('mean')
cairbnb_df['Availability Ratio'] = cairbnb_df['availability_30'] / cairbnb_df['availability_365']
cairbnb_df['Accomodates per Bath'] = cairbnb_df['accommodates'] / cairbnb_df['bathrooms']
cairbnb_df['Accomodates per Bed'] = cairbnb_df['accommodates'] / cairbnb_df['beds']
cairbnb_df.replace([np.inf, -np.inf], 0, inplace=True)
cairbnb_df.drop(columns='beds_bathrooms_sum')
#%%
target_column = 'price'
find_correlations(airbnb_df, airbnb_df)
find_correlations(cairbnb_df, cairbnb_df)
#%%
X_airbnb = airbnb_df.drop(target_column, axis=1)
cX_airbnb = cairbnb_df.drop(target_column, axis=1)
vif_delete(X_airbnb)
#%%
lin_reg(airbnb_df[X_airbnb.columns], airbnb_df, 'price')
#%%
elastic_net(airbnb_df, airbnb_df, 'price')
print('')
#%%
pca(airbnb_df, 'price')
lin_reg(airbnb_df, airbnb_df, 'price', 2)
#%%
vif_delete(cX_airbnb)
#%%
lin_reg(cairbnb_df[cX_airbnb.columns], cairbnb_df, 'price')
elastic_net(cairbnb_df, cairbnb_df, 'price')
# #%%
# print(catboost_predict(airbnb_df))
#%%

## CARS DATASET PREPROCESSING ##
country = pd.read_csv(r"C:\Users\anyas\Desktop\Summer Project\Cars_Country.csv")
cars = pd.read_csv(r"C:\Users\anyas\Desktop\Summer Project\Cars_Multi.csv")
price = pd.read_csv(r"C:\Users\anyas\Desktop\Summer Project\Cars_Price.csv")

merged_cars = pd.merge(cars, price, on=['ID'], how='inner')
merged_cars = merged_cars.select_dtypes(include=[int, float])
merged_cars.drop(columns={'origin'}, inplace=True)
#%%
## Adding indirect variables ##
ccars = merged_cars.copy()
ccars['Acceleration to Weight'] = ccars['acceleration'] / ccars['weight']
ccars['Engine Efficiency'] = ccars['displacement'] / ccars['acceleration']
ccars['Size to Weight'] = ccars['displacement'] / ccars['weight']
ccars['Efficiency Index'] = ccars['mpg'] * ccars['acceleration']
ccars['Cylinder / Weight'] = ccars['cylinders'] / ccars['acceleration']
ccars['cars_from_USA'] = ccars['USA'] * ccars['USA'].sum()
ccars['cars_from_Japan'] = ccars['Japan'] * ccars['Japan'].sum()
ccars['cars_from_Europe'] = ccars['Europe'] * ccars['Europe'].sum()
ccars['Country of Origin Cars'] = ccars['cars_from_Europe'] + ccars['cars_from_Japan'] + ccars['cars_from_USA']
ccars.drop(columns={'cars_from_Japan', 'cars_from_USA', 'cars_from_Europe'}, inplace=True)
ccars['Before 1980'] = (ccars['model'] >= 80).astype(int)
#%%
target_column='price'
find_correlations(merged_cars, merged_cars)
find_correlations(ccars, ccars)
X_cars = merged_cars.drop(target_column, axis=1)
cX_cars = ccars.drop(target_column, axis=1)
vif_delete(X_cars)
vif_delete(cX_cars)
#%%
lin_reg(merged_cars[X_cars.columns], merged_cars, 'price')
#%%
elastic_net(merged_cars, merged_cars, 'price')
print('')
#%%
lin_reg(ccars[cX_cars.columns], ccars, 'price')
elastic_net(ccars, ccars, 'price')
# #%%
# print(catboost_predict(merged_cars, 'price'))
# #%%

# ## GLOBE DATASET PREPROCESSING ##

# gdp = pd.read_csv(r"C:\Users\anyas\Desktop\Summer Project\gdp_data.csv")
# gdp = gdp[(gdp['Year'] == 2005) & (gdp['Series'] == 'GDP in constant 2015 prices (millions of US dollars)')]

# globe1 = pd.read_csv(r"C:\Users\anyas\Desktop\Summer Project\GLOBE-Phase-2-Aggregated-Leadership-Data.csv")
# globe2 = pd.read_csv(r"C:\Users\anyas\Desktop\Summer Project\GLOBE-Phase-2-Aggregated-Societal-Culture-Data.csv")

# merged_globe = pd.merge(globe1, globe2, on=['Country Name'], how='inner')
# gdp.rename(columns={'Country': 'Country Name', 'Value': 'GDP'}, inplace=True)
# gdp.drop(columns={'Source', 'Footnotes', 'Year', 'Series', 'Region/Country/Area'}, inplace=True)
# merged_globe = pd.merge(merged_globe, gdp, on=['Country Name'], how='inner')
# merged_globe = merged_globe.select_dtypes(include=[int, float])
# merged_globe.drop(columns={'Country_x', 'Country_y'}, inplace=True)
# #%%
# target_column='GDP'
# find_correlations(merged_globe, merged_globe)
# X_globe = merged_globe.drop(target_column, axis=1)
# vif_delete(X_globe)
# #%%
# lin_reg(merged_globe[X_globe.columns], merged_globe, 'GDP')
# #%%
# print(elastic_net(merged_globe, merged_globe, 'GDP'))
# print('')
# #%%
# print(catboost_predict(merged_globe, 'GDP'))