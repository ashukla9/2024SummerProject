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
## AIRBNB DATA PREPROCESSING ##

listings = pd.read_csv(r"C:\Users\anyas\Downloads\listings.csv.gz", compression='gzip')
price = pd.read_csv(r"C:\Users\anyas\Downloads\listings (2).csv")

listings['host_response_rate'] = listings['host_response_rate'].str.rstrip('%').astype('float') / 100
listings['host_acceptance_rate'] = listings['host_acceptance_rate'].str.rstrip('%').astype('float') / 100

merged_df = pd.merge(listings, price, on='id', suffixes=('', '_duplicate'))

# Drop duplicate columns
for col in merged_df.columns:
    if col.endswith('_duplicate') and col[:-10] in merged_df.columns:
        if merged_df[col[:-10]].equals(merged_df[col]):
            merged_df.drop(columns=col, inplace=True)

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
def find_correlations(target_df, threshold=0.5):
    
    correlations = target_df.corrwith(target_df[target_column])
    correlations = correlations[(correlations.abs() > threshold)]
    print(correlations)
#%%
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df):
    vif = pd.DataFrame()
    vif['Feature'] = df.columns
    vif['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    if (vif['VIF'] > 10).any():
        print("High VIF: multicollinearity detected. Regularized OLS recommended.")
    return vif

def vif_delete(X_train, X_test):
    while True:
        vif = calculate_vif(X_train)
        max_vif = vif["VIF"].max()
        if max_vif > 10:
            feature_to_drop = vif.loc[vif["VIF"].idxmax(), "Feature"]
            print(f"Dropping feature '{feature_to_drop}' with VIF {max_vif}")
            X_train.drop(columns=[feature_to_drop], inplace=True)
            X_test.drop(columns=[feature_to_drop], inplace=True)
        else:
            break
        
    return X_train, X_test
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, LinearRegression, LassoCV, RidgeCV, HuberRegressor
from sklearn.feature_selection import SelectFromModel
from statsmodels.stats.stattools import durbin_watson
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def detect_outliers_dep(y_train):
    Q1 = np.percentile(y_train, 25)
    Q3 = np.percentile(y_train, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = np.where((y_train < lower_bound) | (y_train > upper_bound))
    return outliers

def anomaly_detection(X_train, y_train):
    from sklearn.ensemble import IsolationForest
    y_outliers = detect_outliers_dep(y_train)
    
    X_train = X_train.drop(index=y_train.index[y_outliers[0]]).reset_index(drop=True)
    y_train = y_train.drop(index=y_train.index[y_outliers[0]]).reset_index(drop=True)

    iso_forest = IsolationForest(contamination='auto', random_state=42)
    X_train['Anomaly'] = iso_forest.fit_predict(X_train)

    non_anomalies_mask = X_train['Anomaly'] != -1

    X_train_clean = X_train[non_anomalies_mask].drop(columns='Anomaly').reset_index(drop=True)
    y_train_clean = y_train[non_anomalies_mask].reset_index(drop=True)

    return X_train_clean, y_train_clean

def remove_constant_columns(df):
    constant_columns = [col for col in df.columns if df[col].nunique() <= 1]
    df_cleaned = df.drop(columns=constant_columns)
    print(f'Removed constant columns: {constant_columns}')
    return df_cleaned

def preprocessing(df, target_column, outliers=False):
    X = df.drop(columns=target_column)
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = remove_constant_columns(X_train)
    
    if outliers == True:
        X_train, y_train = anomaly_detection(X_train, y_train)
    
    scaler = StandardScaler()
    
    train_indices = X_train.index
    test_indices = X_test.index
    test_columns = X_test.columns
    train_columns = X_train.columns
    
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train = pd.DataFrame(X_train, columns=train_columns, index=train_indices)
    X_test = pd.DataFrame(X_test, columns=test_columns, index=test_indices)

    calculate_vif(X_train)
    
    n, p = X_train.shape
    if n < p:
        print("Number of observations greater than number of predictors. Regularized OLS recommended.")
    else:
        print("Number of observations less than number of predictors.")
    
    return X_train, X_test, y_train, y_test

def plot_residuals(y_train, y_test, y_train_pred, y_test_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train, y_train_pred, label='Training Data', alpha=0.6)
    plt.scatter(y_test, y_test_pred, label='Test Data', alpha=0.6, color='orange')
    plt.plot([min(min(y_train), min(y_test)), max(max(y_train), max(y_test))], 
             [min(min(y_train), min(y_test)), max(max(y_train), max(y_test))], 
             color='red', linestyle='--', label='Perfect Fit')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name} Predicted vs Actual Values')
    plt.legend()
    plt.show()
    
    # Residuals Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train_pred, y_train - y_train_pred, label='Training Data', alpha=0.6)
    plt.scatter(y_test_pred, y_test - y_test_pred, label='Test Data', alpha=0.6, color='orange')
    plt.hlines(0, min(min(y_train_pred), min(y_test_pred)), max(max(y_train_pred), max(y_test_pred)), 
               colors='red', linestyles='dashed')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'{model_name} Residuals Plot')
    plt.legend()
    plt.show()

def model_results(model, X_train, y_train, y_test, y_train_pred, y_test_pred, model_name, extra_model = None):
    
    coefficients = model.coef_
    intercept = model.intercept_
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    variance = np.var(y_test, ddof=1)
    
    print(model_name + " Summary Statistics")
    print("================================")
    print(f"Intercept: {intercept}")
    print("Coefficients:")
    best_features = []
    for feature, coef in zip(X_test.columns, coefficients):
        if coef != 0:
            print(f"  {feature}: {coef}")
            best_features.append(feature)
    print(f"Train MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")
    print(f"Test variance: {variance}")
    print(f"Train R-squared: {train_r2}")
    print(f"Test R-squared: {test_r2}")
    
    dw_train = durbin_watson(y_train - y_train_pred)
    dw_test = durbin_watson(y_test - y_test_pred)
    
    print(f"Durbin-Watson statistic for training set: {dw_train}")
    print(f"Durbin-Watson statistic for test set: {dw_test}")
    
    if dw_train < 1.0 or dw_train > 3.0:
        print("Training set may be autocorrelated. Consider time series analysis.")
    if dw_test < 1.0 or dw_test > 3.0:
        print("Test set may be autocorrelated. Consider time series analysis.")
    
    if model_name == 'Randomized Lasso':
        feature_names = X_train.columns[extra_model.get_support()]
    else:
        feature_names = X_train.columns
        
    coef_series = pd.Series(coefficients, index=feature_names)
    sorted_coef_series = coef_series.abs().sort_values(ascending=False)
    
    if len(sorted_coef_series) < 10:
        top_10_features = coef_series.loc[sorted_coef_series.index]
    else:
        top_10_features = coef_series.loc[sorted_coef_series.head(10).index]
    
    # Print the top features with their original signs
    print("Top features most associated with the target outcome based on absolute value of coefficient:")
    print(top_10_features)
    
    plot_residuals(y_train, y_test, y_train_pred, y_test_pred, model_name)
    
def lin_reg(X_train, X_test, y_train, y_test, vif=False):
    
    if vif == True:
        X_train, X_test = vif_delete(X_train, X_test)
        
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    model_results(model, X_train, y_train, y_test, y_train_pred, y_test_pred, 'Linear Regression')

def elastic_net(X_train, X_test, y_train, y_test):
    
    best_model = ElasticNetCV(cv=10, random_state=42, l1_ratio=[.1, .5, .7, .9, .95, .99, 1])
    best_model.fit(X_train, y_train)
    
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    model_results(best_model, X_train, y_train, y_test, y_train_pred, y_test_pred, 'Elastic Net')
    
def randomized_lasso(X_train, X_test, y_train, y_test):
    lasso = LassoCV(cv=10, random_state=42).fit(X_train, y_train)
    model = SelectFromModel(lasso, threshold='mean', prefit=True)
    
    X_train_selected = model.transform(X_train)
    X_test_selected = model.transform(X_test)
    
    best_model = LassoCV(cv=10, random_state=42)
    best_model.fit(X_train_selected, y_train)
    
    y_train_pred = best_model.predict(X_train_selected)
    y_test_pred = best_model.predict(X_test_selected)
    
    model_results(best_model, X_train, y_train, y_test, y_train_pred, y_test_pred, 'Randomized Lasso', model)

def ridge(X_train, X_test, y_train, y_test):
    alphas = np.logspace(-6, 6, 13)
    best_model = RidgeCV(alphas=alphas, cv=10).fit(X_train, y_train)
    
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    model_results(best_model, X_train, y_train, y_test, y_train_pred, y_test_pred, 'Ridge Regression')
    
def huber(X_train, X_test, y_train, y_test, vif = False):
    if vif == True:
        X_train, X_test = vif_delete(X_train, X_test)
        
    best_model = HuberRegressor().fit(X_train, y_train)
    
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    model_results(best_model, X_train, y_train, y_test, y_train_pred, y_test_pred, 'Huber Regression')

#%%
target_column = 'price'
find_correlations(airbnb_df)
X_train, X_test, y_train, y_test = preprocessing(airbnb_df, target_column, outliers=True)
lin_reg(X_train.copy(), X_test.copy(), y_train, y_test, vif=True)
elastic_net(X_train, X_test, y_train, y_test)
randomized_lasso(X_train, X_test, y_train, y_test)
ridge(X_train, X_test, y_train, y_test)
huber(X_train.copy(), X_test.copy(), y_train, y_test, vif = True)
#%%
target_column = 'price'
find_correlations(cairbnb_df)
X_train, X_test, y_train, y_test = preprocessing(cairbnb_df, target_column, outliers = True)
lin_reg(X_train.copy(), X_test.copy(), y_train, y_test, vif=True)
elastic_net(X_train, X_test, y_train, y_test)
randomized_lasso(X_train, X_test, y_train, y_test)
ridge(X_train, X_test, y_train, y_test)
huber(X_train.copy(), X_test.copy(), y_train, y_test, vif = True)
#%%
target_column = 'GDP in current prices (millions of US dollars)'
find_correlations(df_2015)
X_train, X_test, y_train, y_test = preprocessing(df_2015, target_column, outliers = True)
lin_reg(X_train.copy(), X_test.copy(), y_train, y_test, vif=True)
elastic_net(X_train, X_test, y_train, y_test)
randomized_lasso(X_train, X_test, y_train, y_test)
ridge(X_train, X_test, y_train, y_test)
huber(X_train.copy(), X_test.copy(), y_train, y_test, vif = True)