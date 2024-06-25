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

all_gdp_data = pd.read_csv(r"C:\Users\anyas\Desktop\Summer Project\gdp_data.csv")
#%%
#Create histograms of the data to explore the distribution
#GDP Data is pretty obviously skewed
data = imputed_dfs.get("GDP_in_current_prices_millions_of_US_dollars")
health_data = imputed_dfs.get("Domestic_general_government_health_expenditure_percent_of_total_government_expenditure")
#2010 data
gdp_data_2010 = data[data['Year'] == 2010]['GDP in current prices (millions of US dollars)']
plt.hist(gdp_data_2010, bins=30, color='skyblue', edgecolor='black')
 
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('GDP Values in 2010')
 
plt.show()
#2015 data
gdp_data_2015 = data[data['Year'] == 2015]['GDP in current prices (millions of US dollars)']
plt.hist(gdp_data_2015, bins=30, color='skyblue', edgecolor='black')
 
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('GDP Values in 2015')
 
plt.show()

#2010 data
health_data_2010 = health_data[health_data['Year'] == 2010]['Domestic general government health expenditure (% of total government expenditure)']
plt.hist(health_data_2010, bins=30, color='skyblue', edgecolor='black')
hist, bin_edges = np.histogram(health_data_2010, bins=30)

plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Health Expenditure in 2010')
 
plt.show()

#2015 data
health_data_2015 = health_data[health_data['Year'] == 2015]['Domestic general government health expenditure (% of total government expenditure)']
plt.hist(health_data_2015, bins=30, color='skyblue', edgecolor='black')
 
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Health Expenditure in 2015')
 
plt.show()
#%%
#Check the skew and kurtosis of the data
from scipy.stats import skew
from scipy.stats import kurtosis 

skew_2010 = skew(gdp_data_2010)
skew_2015 = skew(gdp_data_2015)
kurtosis_2010 = kurtosis(gdp_data_2010)
kurtosis_2015 = kurtosis(gdp_data_2015)

print("GDP")
print("Skew in 2010: ", skew_2010)

if skew_2010 > .8 or skew_2010 < -.8:
    print("2010 skew value higher than expected for normal distribution.")

print("Skew in 2015: ", skew_2015)

if skew_2015 > .8 or skew_2015 < -.8:
    print("2015 skew value higher than expected for normal distribution.")

print("Kurtosis in 2010: ", kurtosis_2010)

if kurtosis_2010 > 3 or kurtosis_2010 < -3:
    print("2010 kurtosis value higher than expected for normal distribution.")

print("Kurtosis in 2015: ", kurtosis_2015)

if kurtosis_2015 > 3 or kurtosis_2015 < -3:
    print("2015 kurtosis value higher than expected for normal distribution.")

print('')
print('Health Expenditure')

hskew_2010 = skew(health_data_2010)
hskew_2015 = skew(health_data_2015)
hkurtosis_2010 = kurtosis(health_data_2010)
hkurtosis_2015 = kurtosis(health_data_2015)

print("Skew in 2010: ", hskew_2010)

if hskew_2010 > .8 or hskew_2010 < -.8:
    print("2010 skew value higher than expected for normal distribution.")

print("Skew in 2015: ", hskew_2015)

if hskew_2015 > .8 or hskew_2015 < -.8:
    print("2015 skew value higher than expected for normal distribution.")

print("Kurtosis in 2010: ", hkurtosis_2010)

if hkurtosis_2010 > 3 or hkurtosis_2010 < -3:
    print("2010 kurtosis value higher than expected for normal distribution.")

print("Kurtosis in 2015: ", hkurtosis_2015)

if hkurtosis_2015 > 3 or hkurtosis_2015 < -3:
    print("2015 kurtosis value higher than expected for normal distribution.")

#%%
## Is there a relationship between GDP and unemployment? ##
# Example of a scatter plot to investigate relationship between life expectancy and GDP

plt.figure(figsize=(8, 6))
unemploy_data = imputed_dfs.get("Unemployment_rate_-_Total")
plt.scatter(unemploy_data['Unemployment rate - Total'], data['GDP in current prices (millions of US dollars)'], alpha=0.6)
plt.title('Percent unemployed vs GDP')
plt.xlabel('Percent Unemployed')
plt.ylabel('GDP in current prices (millions of US dollars)')
plt.grid(True)
plt.show()
#%%
## Which of these variables, if any, follow a normal distribution? ##
#test variables for normality using Kolmogorov–Smirnov Test
# Shapiro-Wilks Test more appropriate for <50 sample points
print('')
print('Which of these variables follow a normal distribution?')
print('')
from scipy.stats import shapiro
from scipy.stats import kstest
from tabulate import tabulate
import statsmodels.api as sm

def test_normality(data, column, alpha=0.05):
    results = []
    stat, p_value = shapiro(data)
    normal = p_value > alpha
    results.append([column, round(p_value, 3), normal])
    return results

def test_normality_ks(data, column, alpha=0.05):
    results = []
    #can use k-s test to determine other distributions as well
    stat, p_value = kstest(data, 'norm')
    normal = p_value > alpha
    results.append([column, round(p_value, 3), normal])
    return results

normality_results_15 = test_normality(gdp_data_2015, column = "GDP")
normality_results_10 = test_normality(gdp_data_2010, column = "GDP")
normality_kresults_15 = test_normality_ks(gdp_data_2015, column = "GDP")
normality_kresults_10 = test_normality_ks(gdp_data_2010, column = "GDP")

sm.qqplot(data["GDP in current prices (millions of US dollars)"], line='s')  # 's' indicates standardized line (slope = 1, intercept = mean)
plt.title('Q-Q Plot of GDP Data')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.show()

print('Normality results for 2010:')
print('S-W: ' ,normality_results_10)
print('K-S: ' ,normality_kresults_10)
print('')
print('Normality results for 2015:')
print('S-W: ' ,normality_results_15)
print('K-S: ' ,normality_kresults_15)
print('')
print('Normality results for GDP as a whole:')
print('S-W: ' ,test_normality(data["GDP in current prices (millions of US dollars)"], column = "GDP"))
print('K-S: ' ,test_normality_ks(data["GDP in current prices (millions of US dollars)"], column = "GDP"))
print('')
hnormality_results_15 = test_normality(health_data_2015, column = "Health expenditure")
hnormality_results_10 = test_normality(health_data_2010, column = "Health expenditure")
hnormality_kresults_15 = test_normality_ks(health_data_2015, column = "Health expenditure")
hnormality_kresults_10 = test_normality_ks(health_data_2010, column = "Health expenditure")

sm.qqplot(health_data["Domestic general government health expenditure (% of total government expenditure)"], line='s')  # 's' indicates standardized line (slope = 1, intercept = mean)
plt.title('Q-Q Plot of Health Expenditure Data')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.show()

print('Health expenditure normality results for 2010:')
print('S-W: ' , hnormality_results_10)
print('K-S: ' ,hnormality_kresults_10)
print('')
print('Health expenditure normality results for 2015:')
print('S-W: ' ,hnormality_results_15)
print('K-S: ' ,hnormality_kresults_15)
print('')
print('Normality results for health expenditure as a whole:')
print('S-W: ' ,test_normality(health_data["Domestic general government health expenditure (% of total government expenditure)"], column = "Health expenditure"))
print('K-S: ' ,test_normality_ks(health_data["Domestic general government health expenditure (% of total government expenditure)"], column = "Health expenditure"))
#%%
for key, df in imputed_dfs.items():
    df.set_index(['Country', 'Year'], inplace=True)
#%%
## What variables are strongly correlated with one another ##
import seaborn as sns

print('')
print('What variables are strongly correlated with GDP?')
print('')

def find_correlations(dataframes, variable, threshold=0.5):

    # Initialize a DataFrame to store correlation results
    correlation_results = {}

    # Iterate through each DataFrame in the dictionary
    for key, df in dataframes.items():

        # Calculate the correlation between each variable and GDP
        correlations = df.corrwith(variable)
        # Store the correlations in the results DataFrame
        correlation_results[key] = correlations.iloc[0]
            
    return correlation_results

## What variables are strongly correlated with GDP? ##
# Can change threshold of output depending on your analysis
correlations = find_correlations(imputed_dfs, imputed_dfs.get("GDP_in_current_prices_millions_of_US_dollars").iloc[:,0])

for next_key, value in correlations.items():
    if abs(value) >= .3:
        print("Correlation with variable", next_key)
        print("Value: ", value)
        print('')
#%%
## Do the distributions of GDP in 2010 vs. 2015 have equal variance? ##
# Testing whether distributions of GDP in 2015 vs 2010 had equal variance
# Can use Levene's test as it does not assume normality

print('')
print('Do the distributions of GDP in 2010 vs. 2015 have equal variance?')
print('')

from scipy.stats import levene

statistic, p_value = levene(gdp_data_2010, gdp_data_2015)

alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: Variances are not equal.")
else:
    print("Fail to reject the null hypothesis: Variances are equal.")
#%%
## Does the GDP of country A differ from 2010 to 2015 by a statistically significant amount? ##

# Trick question - if we had a significant amount of data we would use a paired t-test
# But we only have one value for GDP in 2010 vs. 2015 so we should just use a simple comparison

#%%
## Does the GDP differ between country A to country B by a 
## statistically significant amount? ##

# Example of One-Sided T-Test using Czechia and Portugal
# Overall GDP is non-normal so for this reduced sample size will use a nonparametric test
# For a larger sample size, could use one-sided t-test
# Mann-Whitney works with unequal sample sizes but the statistical power is reduced as the difference increases

print('')
print('Does the GDP of country A and country B differ by a statistically significant amount?')
print('')

from scipy.stats import mannwhitneyu

gdp_czechia = all_gdp_data[(all_gdp_data['Country'] == 'Czechia') & (all_gdp_data['Series'] == 'GDP in current prices (millions of US dollars)')]['Value']
gdp_portugal = all_gdp_data[(all_gdp_data['Country'] == 'Portugal') & (all_gdp_data['Series'] == 'GDP in current prices (millions of US dollars)')]['Value']

statistic, p_value = mannwhitneyu(gdp_czechia, gdp_portugal)

print('Czechia GDP over the years:', gdp_czechia)
print('Portugal GDP over the years:', gdp_portugal)

alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: GDPs are statistically different.")
else:
    print("Fail to reject the null hypothesis: No significant difference in GDPs.")
#%%
## Does the GDP differ among regions A, B, and C by a statistically 
## significant amount over all years? ##
# For the smaller sample size, will use a nonparametric test
# For a larger sample size, ANOVA will generalize well
# Follow-up w/ Mann-Whitney to find the exact pair(s) w/ significant differences
print('')
print('Does the GDP differ between regions A, B, and C by a statistically significant amount?')
print('')
from scipy.stats import kruskal
from itertools import combinations

region_a_gdp = all_gdp_data[(all_gdp_data['Country'] == 'Africa') & (all_gdp_data['Series'] == 'GDP in current prices (millions of US dollars)')]['Value']
region_b_gdp = all_gdp_data[(all_gdp_data['Country'] == 'Asia') & (all_gdp_data['Series'] == 'GDP in current prices (millions of US dollars)')]['Value']
region_c_gdp = all_gdp_data[(all_gdp_data['Country'] == 'Americas') & (all_gdp_data['Series'] == 'GDP in current prices (millions of US dollars)')]['Value']

groups = [region_a_gdp, region_b_gdp, region_c_gdp]
group_names = ['Americas', 'Asia', 'Africa']

f_statistic, p_value = kruskal(region_a_gdp, region_b_gdp, region_c_gdp)

alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference in GDPs between the regions.")
    print('')
    # Generate all pairwise combinations of groups
    pairwise_combinations = list(combinations(range(len(groups)), 2))
    
    # Perform Mann-Whitney U tests for each pair
    for i, (idx1, idx2) in enumerate(pairwise_combinations):
        stat, p = mannwhitneyu(groups[idx1], groups[idx2])
        # Apply Bonferroni correction
        corrected_p = p * len(pairwise_combinations)
        print(f"Comparison {group_names[idx1]} vs {group_names[idx2]}:")
        print(f"Mann-Whitney U statistic: {stat}, P-value (corrected): {corrected_p}")
        if corrected_p < alpha:
            print("Significant difference found.")
        else:
            print("No significant difference found.")
            
else:
    print("Fail to reject the null hypothesis: No significant difference in GDPs between the regions.")
#%%
# Looked through the correlations between all variables
# Decided to drop these as they are most likely collinear
imputed_dfs_copy = imputed_dfs.copy()
del imputed_dfs_copy['Employment_by_industry__Agriculture_percent_Female']
del imputed_dfs_copy['Employment_by_industry__Agriculture_percent_Male']
del imputed_dfs_copy['Employment_by_industry__Industry_percent_Female']
del imputed_dfs_copy['Employment_by_industry__Industry_percent_Male']
del imputed_dfs_copy['Employment_by_industry__Services_percent_Female']
del imputed_dfs_copy['Employment_by_industry__Services_percent_Male']
del imputed_dfs_copy['Unemployment_rate_-_Male']
del imputed_dfs_copy['Unemployment_rate_-_Female']
del imputed_dfs_copy['Life_expectancy_at_birth_for_females_years']
del imputed_dfs_copy['Life_expectancy_at_birth_for_males_years']
del imputed_dfs_copy['Domestic_general_government_health_expenditure_percent_of_total_government_expenditure']
del imputed_dfs_copy['International_migrant_stock__Female_percent_total_Population']
del imputed_dfs_copy['International_migrant_stock__Male_percent_total_Population']
del imputed_dfs_copy['Labour_force_participation_-_Female']
del imputed_dfs_copy['Labour_force_participation_-_Male']
del imputed_dfs_copy['Other_of_concern_to_UNHCR_number']
del imputed_dfs_copy['Gross_enrollment_ratio_-_Lower_secondary_level_male']
del imputed_dfs_copy['Gross_enrollment_ratio_-_Lower_secondary_level_female']
del imputed_dfs_copy['Gross_enrollment_ratio_-_Primary_male']
del imputed_dfs_copy['Gross_enrollment_ratio_-_Upper_secondary_level_female']
del imputed_dfs_copy['Gross_enrollment_ratio_-_Primary_female']
del imputed_dfs_copy['Gross_enrollment_ratio_-_Upper_secondary_level_male']
del imputed_dfs_copy['Infant_mortality_for_both_sexes_per_1,000_live_births']
#%%
del imputed_dfs_copy['Students_enrolled_in_primary_education_thousands']
del imputed_dfs_copy['All_staff_compensation_as_percent_of_total_expenditure_in_public_institutions_percent']
#%%
for first_key in imputed_dfs_copy:
    correlations = find_correlations(imputed_dfs_copy, imputed_dfs_copy.get(first_key).iloc[:,0])
    print("Significant correlations for variable ", first_key)
    for next_key, value in correlations.items():
        if abs(value) >= .7 and first_key != next_key:
            print("Correlation with variable", next_key)
            print("Value: ", value)
            print('')
#%%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaled_df_dict = {}

for key, df in imputed_dfs_copy.items():
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    scaled_df_dict[key] = scaled_df

target_variable = scaled_df_dict.get('GDP_in_constant_2015_prices_millions_of_US_dollars')
#%%
## Is the number of women in parliament (can be any variable) statistically significant 
## in determining whether a country will have a high GDP? ##
# Univariate Linear Regression example

print('')
print('Is the number of women in parliament statistically significant in determining whether a country will have a high GDP in 2015?')
print('')

import statsmodels.api as sm

X = scaled_df_dict.get("Seats_held_by_women_in_national_parliament,_as_of_February_percent")['Seats held by women in national parliament, as of February (%)']
y = scaled_df_dict.get('GDP_in_current_prices_millions_of_US_dollars')['GDP in current prices (millions of US dollars)']
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

summary = model.summary()
print(summary)
print('The R squared value of this summary shows that this variable is insignificant.')
#%%
data = data.reset_index()
#%%
## What variables are most significant in determining a country’s GDP? ##
# Multivariate Linear Regression example

## TAKING A LONG TIME TO RUN - MIGHT DO THIS MANUALLY ##
import pandas as pd
import itertools
import statsmodels.api as sm

# Add 'Country' and 'Year' columns to each DataFrame in the dictionary
for key in scaled_df_dict:
    value = scaled_df_dict.get(key)
    value['Country'] = data['Country']
    value['Year'] = data['Year']

# Function to get the combined DataFrame for selected features
def get_combined_df(keys, df_dict):
    combined_df = df_dict[keys[0]].set_index(['Country', 'Year'])
    for key in keys[1:]:
        combined_df = combined_df.join(df_dict[key].set_index(['Country', 'Year']), how='inner')
    return combined_df

# Function to calculate AIC for different combinations of features
def calculate_aic(df_dict, target_column):
    features = list(df_dict.keys())
    best_aic = float('inf')
    best_model = None
    best_features = None
    
    # Iterate over all combinations of features
    for i in range(1, len(features) + 1):
        print(i)
        for combo in itertools.combinations(features, i):
            X = get_combined_df(combo, df_dict)
            y = scaled_df_dict.get(target_column).set_index(['Country', 'Year'])
            # Ensure the index aligns
            X, y = X.align(y, join='inner', axis=0)
            
            # Add constant to the model
            X = sm.add_constant(X)
            
            # Fit the model
            model = sm.OLS(y, X).fit()
            aic = model.aic
            
            if aic < best_aic:
                best_aic = aic
                best_model = model
                best_features = combo
                
    return best_model, best_aic, best_features

# Calculate AIC and find the best model
target_column = 'GDP_in_constant_2015_prices_millions_of_US_dollars'
best_model, best_aic, best_features = calculate_aic(scaled_df_dict, target_column)

# Print the best model summary
print(f"Best model features: {best_features}")
print(f"Best AIC: {best_aic}")
if best_model:
    print(best_model.summary())
else:
    print("No valid model found.")
#%%

# p_values = results.pvalues

# alpha = 0.05

# significant_vars = p_values[p_values < alpha].index
# print("Statistically significant variables at alpha = 0.05:")
# print(significant_vars)

#seems the most significant variables are related to labour force participation
# #%%
# #print dataframes to .csvs for use in other applications
# data_2015_df.to_csv('data_2015.csv', index=False)
# data_2010_df.to_csv('data_2010.csv', index=False)
# gdp.to_csv('gdp_data.csv', index=False)
# #%%
# ## WORKING ON CLUSTERING ##
# # ## Suppose we were to cluster these countries. What variables would define each cluster? ##

# # from sklearn.decomposition import PCA

# # pca = PCA()
# # pca.fit(X)

# # # Get explained variance ratio
# # explained_variance = pca.explained_variance_ratio_

# # # Plotting the scree plot
# # plt.figure(figsize=(10, 6))
# # plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
# # plt.title('Scree Plot')
# # plt.xlabel('Principal Component')
# # plt.ylabel('Explained Variance Ratio')
# # plt.xticks(np.arange(1, len(explained_variance) + 1))
# # plt.grid(True)
# # plt.show()
# # #%%
# # from sklearn.cluster import KMeans

# # pca = PCA(n_components=7)
# # X_pca = pca.fit_transform(X)

# # inertia = []

# # # Define range of clusters (adjust as needed)
# # k_range = range(1, 11)

# # for k in k_range:
# #     kmeans = KMeans(n_clusters=k, random_state=0)
# #     kmeans.fit(X_pca)
# #     inertia.append(kmeans.inertia_)

# # # Plot the inertia values
# # plt.plot(k_range, inertia, marker='o')
# # plt.xlabel('Number of Clusters (k)')
# # plt.ylabel('Inertia')
# # plt.title('Elbow Method for Optimal k')
# # plt.xticks(k_range)
# # plt.show()
# # #%%
# # kmeans = KMeans(n_clusters=2, random_state=0)

# # # Fit the model to the scaled data
# # kmeans.fit(X_pca)

# # # Predict the cluster labels
# # cluster_labels = kmeans.labels_

# # # Assign cluster labels back to the original dataframe
# # data_2015_df.loc[:, 'Cluster'] = cluster_labels

# # cluster_means = data_2015_df.groupby('Cluster').mean()

# # # Alternatively, you can visualize the clusters
# # import matplotlib.pyplot as plt

# # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
# # plt.xlabel('GDP')
# # plt.ylabel('Population')
# # plt.title('Clustering of Countries')
# # plt.colorbar()
# # plt.show()
