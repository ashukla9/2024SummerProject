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

gdp_2015_data = pd.read_csv(r"C:\Users\anyas\Desktop\Summer Project\data_2015.csv")
gdp_2015_copy = gdp_2015_data.copy()
all_gdp_data = pd.read_csv(r"C:\Users\anyas\Desktop\Summer Project\gdp_data.csv")
#%%
## What countries are anomalies?  ##

from sklearn.ensemble import IsolationForest

# Extract the country labels
countries = gdp_2015_data['Country']
gdp_2015_copy = gdp_2015_copy.drop(columns='Year')

# Identify numerical columns
numerical_cols = gdp_2015_data.select_dtypes(include=['float64', 'int64']).columns

# Initialize the Isolation Forest model
iso_forest = IsolationForest(contamination='auto', random_state=42)

# Fit the model
gdp_2015_data['Anomaly'] = iso_forest.fit_predict(gdp_2015_data[numerical_cols])

print("Outliers are:")
for index, row in gdp_2015_data.iterrows():
    if row['Anomaly'] == -1:
        print(row['Country'])
#%%      
anomalies = gdp_2015_data['Anomaly']
gdp_2015_data = gdp_2015_data.drop(columns={'Anomaly'})
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

columns=['Country', 'Year',
'GDP in constant 2015 prices (millions of US dollars)',         
'Employment by industry: Agriculture (%) Female',
'Employment by industry: Agriculture (%) Male',
'Employment by industry: Industry (%) Female',
'Employment by industry: Industry (%) Male',
'Employment by industry: Services (%) Female',
'Employment by industry: Services (%) Male',
'Life expectancy at birth for females (years)',
'Life expectancy at birth for males (years)',
'Gross enrollment ratio - Lower secondary level (female)',
'Gross enrollment ratio - Lower secondary level (male)',
'Gross enrollment ratio - Primary (female)',
'Gross enrollment ratio - Upper secondary level (female)',
'Gross enrollment ratio - Upper secondary level (male)',
'International migrant stock: Female (% total Population)',
'International migrant stock: Male (% total Population)',
'Labour force participation - Female',
'Labour force participation - Male',
'Unemployment rate - Female',
'Unemployment rate - Male', 
'Infant mortality for both sexes (per 1,000 live births)',
'International migrant stock: Both sexes (% total population)',
'Other of concern to UNHCR (number)',
'Total population of concern to UNHCR (number)',
'Students enrolled in upper secondary education (thousands)',
'Students enrolled in lower secondary education (thousands)',
'Employment by industry: Services (%) Male and Female'
]
gdp_2015_data = gdp_2015_data.drop(columns=columns)

# Values greater than 10 are generally seen as collinear
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

df = pd.DataFrame(gdp_2015_data)

X = add_constant(df)

# Calculate VIF for each predictor
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Print VIF data
print(vif_data)

#%%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(gdp_2015_data)
scaled_df = pd.DataFrame(scaled_data, columns=gdp_2015_data.columns)

cols = gdp_2015_copy.columns
numerical_cols = gdp_2015_copy.select_dtypes(include=['float64', 'int64']).columns
gdp_2015_copy[numerical_cols] = scaler.fit_transform(gdp_2015_copy[numerical_cols])
kscaled_df = pd.DataFrame(gdp_2015_copy, columns = cols)

target_variable = gdp_2015_data['GDP in current prices (millions of US dollars)']
#%%
## Is the number of women in parliament (can be any variable) statistically significant 
## in determining whether a country will have a high GDP? ##
# Univariate Linear Regression example

print('')
print('Is the number of women in parliament statistically significant in determining whether a country will have a high GDP in 2015?')
print('')

import statsmodels.api as sm

X = scaled_df['Seats held by women in national parliament, as of February (%)']
y = target_variable
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

summary = model.summary()
print(summary)
print('The R squared value of this summary shows that this variable is insignificant.')
#%%
target_column = 'GDP in current prices (millions of US dollars)'
predictors = scaled_df.columns.drop(target_column)
#%%
## What variables are most significant in determining a country’s GDP? ##
## Suppose a country wanted to improve its GDP. What variables should it focus on? ##
# Multivariate Linear Regression example

import pandas as pd
import itertools
import statsmodels.api as sm

# Function to calculate AIC for different combinations of features
def calculate_aic(df, target_column):
    best_aic = float('inf')
    best_model = None
    best_features = None
    
    # Iterate over all combinations of predictors
    # best model uses 4 predictors
    # takes a long time to run over all the features
    # CHANGE VALUE
    for i in range(1, 5):
        for combo in itertools.combinations(predictors, i):
            X = df[list(combo)]
            y = df[target_column]
            
            X = sm.add_constant(X)
            
            model = sm.OLS(y, X).fit()
            aic = model.aic
            
            if aic < best_aic:
                best_aic = aic
                best_model = model
                best_features = combo

    return best_model, best_aic, best_features

best_model, best_aic, best_features = calculate_aic(scaled_df, target_column)

print(f"Best model features: {best_features}")
print(f"Best AIC: {best_aic}")
if best_model:
    print(best_model.summary())
else:
    print("No valid model found.")
#%%
## Suppose we were to cluster these countries. What variables would define each cluster? ##

from sklearn.decomposition import PCA

countries = gdp_2015_copy['Country']
pca = PCA()
X_k = kscaled_df.drop(columns="Country")
pca.fit(X_k)

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
from sklearn.cluster import KMeans

pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_k)

inertia = []

# Define range of clusters (adjust as needed)
k_range = range(1, 50)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)

# Plot the inertia values
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.xticks(k_range)
plt.show()
#%%
kmeans = KMeans(n_clusters=7, random_state=0)

# Fit the model to the scaled data
kmeans.fit(X_pca)

# Predict the cluster labels
cluster_labels = kmeans.labels_

# Assign cluster labels back to the original dataframe
kscaled_df.loc[:, 'Cluster'] = cluster_labels

cluster_means = kscaled_df.groupby('Cluster').mean()

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# Create a DataFrame for the loadings
loadings_df = pd.DataFrame(loadings, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], index=numerical_cols)

for pc in loadings_df.columns:
    print(f"\nTop 5 variables for {pc}:")
    print(loadings_df[pc].abs().sort_values(ascending=False).head(5))

# Alternatively, you can visualize the clusters
plt.figure(figsize=(14, 10))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
for i, country in enumerate(countries):
    plt.annotate(country, (X_pca[i, 0], X_pca[i, 1]), fontsize=8)
plt.title('Clustering of Countries')
plt.colorbar()
plt.show()
