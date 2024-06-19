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
# from statsmodels.imputation import mice

# countries = merged_df['Country']
# numeric_data = merged_df.drop(columns=['Country'])

# imputer = mice.MICEData(numeric_data)

# imputed_mice_df = imputer.data

# imputed_mice_df.insert(0, 'Country', countries)
#%%
## Is there a relationship between GDP and life expectancy? ##
# Example of a scatter plot to investigate relationship between life expectancy and GDP

plt.figure(figsize=(8, 6))
plt.scatter(imputed_df['Life expectancy at birth for both sexes (years)'], imputed_df['GDP per capita (US dollars)'], alpha=0.6)
plt.title('Life Expectancy vs GDP per capita')
plt.xlabel('Life Expectancy')
plt.ylabel('GDP per capita (US dollars)')
plt.grid(True)
plt.show()
#%%
#scale data as some are in percentages and others are raw numbers
from sklearn.preprocessing import StandardScaler

# Separate Country column
countries = imputed_df['Country']
year = imputed_df['Year']
numeric_data = imputed_df.drop(columns=['Country', 'Year'])

scaler = StandardScaler()

scaled_data = scaler.fit_transform(numeric_data)

scaled_df = pd.DataFrame(scaled_data, columns=numeric_data.columns)

scaled_df.insert(0, 'Country', countries)
scaled_df.insert(1, 'Year', year)
#%%
#break data into years
data_2015_df = scaled_df[scaled_df['Year'] == 2015]
data_2010_df = scaled_df[scaled_df['Year'] == 2010]

data_2010_no = data_2010_df.drop(columns = ['Country', 'Year'])
data_2015_no = data_2010_df.drop(columns = ['Country', 'Year'])
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

def test_normality(data, alpha=0.05):
    results = []
    
    for column in data.columns:
        
        col_data = data[column]
        
        stat, p_value = shapiro(col_data)
        
        normal = p_value > alpha
        results.append([column, round(p_value, 3), normal])
    return results

def test_normality_ks(data, alpha=0.05):
    results = []
    
    for column in data.columns:
        
        col_data = data[column]
        
        stat, p_value = kstest(col_data, 'norm')
        
        normal = p_value > alpha
        results.append([column, round(p_value, 3), normal])
    return results

normality_results_15 = test_normality(data_2015_no)
normality_results_10 = test_normality(data_2010_no)
normality_kresults_15 = test_normality_ks(data_2015_no)
normality_kresults_10 = test_normality_ks(data_2010_no)

sm.qqplot(data_2015_no["Unemployment rate - Female"], line='s')  # 's' indicates standardized line (slope = 1, intercept = mean)
plt.title('Q-Q Plot of GDP Data')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.show()

print('Normality results for 2010')
headers = ["Variable", "Shapiro-Wilk p-value", "Shapiro-Wilk Normal?", "K-S p-value", "K-S Normal?"]
combined_results_10 = []

for (name, p_value, normal), (_, k_p_value, k_normal) in zip(normality_results_10, normality_kresults_10):
    combined_results_10.append([name, p_value, normal, k_p_value, k_normal])

print(tabulate(combined_results_10, headers, tablefmt="pretty"))
print('')

# Print results for 2015 data
print('Normality results for 2015')
combined_results_15 = []

for (name, p_value, normal), (_, k_p_value, k_normal) in zip(normality_results_15, normality_kresults_15):
    combined_results_15.append([name, p_value, normal, k_p_value, k_normal])

print(tabulate(combined_results_15, headers, tablefmt="pretty"))
#%%
## What variables are strongly correlated with GDP in 2010? 
## 2015? ##
#test for correlations between variables and gdp

# Assault rate correlation is off - could be because there 
# was not much data provided and the imputations skewed the data

print('')
print('What variables are strongly correlated with GDP in 2010?')
print('')

corr_2015 = data_2015_no.corr()['GDP per capita (US dollars)'].drop('GDP per capita (US dollars)')
corr_2010 = data_2010_no.corr()['GDP per capita (US dollars)'].drop('GDP per capita (US dollars)')

filter_2015 = corr_2015[corr_2015.abs() > 0.05]
filter_2010 = corr_2010[corr_2010.abs() > 0.05]

print(filter_2010)
print('')
print('')
print('What variables are strongly correlated with GDP in 2015?')
print('')
print(filter_2015)
#%%
## Do the distributions of GDP in 2010 vs. 2015 have equal variance? ##
# Testing whether distributions of GDP in 2015 vs 2010 had equal variance
# Assuming non-normality as per previous tests so will use Fligner-Killeen Test
# Otherwise could use Levene's Test, Bartlett's Test

print('')
print('Do the distributions of GDP in 2010 vs. 2015 have equal variance?')
print('')

from scipy.stats import fligner

statistic, p_value = fligner(data_2015_df['GDP per capita (US dollars)'], data_2010_df['GDP per capita (US dollars)'])

alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: Variances are not equal.")
else:
    print("Fail to reject the null hypothesis: Variances are equal.")
#%%
## Does the GDP of country A differ from 2010 to 2015 by a statistically significant amount? ##
#Not a significant amount of data so t-test may not produce accurate results
print('')
print('Does the GDP of country A differ from 2010 to 2015 by a statistically significant amount?')
print('')
from scipy.stats import ttest_rel

#Example of Paired T-Test using Czechia - could be replicated with any country
GDP_2015 = data_2015_df.loc[data_2015_df['Country'] == 'Czechia', 'GDP per capita (US dollars)'].values[0]
GDP_2010 = data_2010_df.loc[data_2010_df['Country'] == 'Czechia', 'GDP per capita (US dollars)'].values[0]
t_stat, p_value = ttest_rel(GDP_2010, GDP_2015)

print('Czechian GDP in 2015:', GDP_2015)
print('Czechian GDP in 2010:', GDP_2010)
print('')

alpha = 0.05
if p_value < alpha:
    print("The difference in Czechian GDP between 2010 and 2015 is statistically significant.")
else:
    print("The difference in Czechian GDP between 2010 and 2015 is not statistically significant.")
#%%
## Does the GDP differ between country A to country B by a 
## statistically significant amount? ##

#Example of One-Sided T-Test using Czechia and Mexico
#Only two data points per group so will use Mann-Whitney Test
#as we most likely cannot assume normality

print('')
print('Does the GDP of country A and country B differ by a statistically significant amount?')
print('')

from scipy.stats import mannwhitneyu

gdp_czechia=[]
gdp_portugal=[]
gdp_czechia.append(data_2015_df.loc[data_2015_df['Country'] == 'Czechia', 'GDP per capita (US dollars)'].values[0])
gdp_czechia.append(data_2010_df.loc[data_2010_df['Country'] == 'Czechia', 'GDP per capita (US dollars)'].values[0])
gdp_portugal.append(data_2015_df.loc[data_2015_df['Country'] == 'Portugal', 'GDP per capita (US dollars)'].values[0])
gdp_portugal.append(data_2010_df.loc[data_2010_df['Country'] == 'Portugal', 'GDP per capita (US dollars)'].values[0])

statistic, p_value = mannwhitneyu(gdp_czechia, gdp_portugal)

print('Czechia GDP in 2015 and 2010:', gdp_czechia)
print('Portugal GDP in 2015 and 2010:', gdp_portugal)

alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: GDPs are statistically different.")
else:
    print("Fail to reject the null hypothesis: No significant difference in GDPs.")
#%%
## Does the GDP differ among regions A, B, and C by a statistically 
## significant amount over all years? ##
#Example of Kruskal-Wallis test between three regions
#Use ANOVA when data is normally distributed
print('')
print('Does the GDP differ between regions A, B, and C by a statistically significant amount?')
print('')
from scipy.stats import kruskal
import scikit_posthocs as sp

gdp.rename(columns={np.nan: 'Country'}, inplace=True)
#gdp = gdp.drop(index = 0)
gdp['Value'] = gdp['Value'].apply(clean_and_convert_to_float)
        
region_a_gdp = gdp[(gdp['Country'] == 'Africa') & (gdp['Series'] == 'GDP per capita (US dollars)')]['Value']
region_b_gdp = gdp[(gdp['Country'] == 'Asia') & (gdp['Series'] == 'GDP per capita (US dollars)')]['Value']
region_c_gdp = gdp[(gdp['Country'] == 'Americas') & (gdp['Series'] == 'GDP per capita (US dollars)')]['Value']

f_statistic, p_value = kruskal(region_a_gdp, region_b_gdp, region_c_gdp)

alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference in GDPs between the regions.")
    data_melted = gdp[(gdp['Country'].isin(['Africa', 'Asia', 'Americas'])) & 
                      (gdp['Series'] == 'GDP per capita (US dollars)')].melt(id_vars=['Country'], 
                                                                            value_vars=['Value'], 
                                                                            var_name='Series', 
                                                                            value_name='Value')
    nemenyi_test = sp.posthoc_nemenyi(data_melted, val_col='Value', group_col='Country')
    print("Nemenyi's test results:")
    print(nemenyi_test)
    print('')
    print('There is a significant difference between the GDP of the Americas and Africa')
else:
    print("Fail to reject the null hypothesis: No significant difference in GDPs between the regions.")
#%%
## Is the number of women in parliament (can be any variable) statistically significant 
## in determining whether a country will have a high GDP in 2015? ##
# Univariate Linear Regression example

print('')
print('Is the number of women in parliament statistically significant in determining whether a country will have a high GDP in 2015?')
print('')

import statsmodels.api as sm

X = data_2015_df['Seats held by women in national parliament, as of February (%)']
y = data_2015_df['GDP per capita (US dollars)']
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

summary = model.summary()
print(summary) #t value demonstrates that this is not significant
#r^2 shows that almost none of the variance in the model is explained by variable
#%%
## What variables are most significant in determining a country’s GDP in 2015? ##
# Multivariate Linear Regression example

print('')
print('What variables are most significant in determining a country’s GDP in 2015?')
print('')

print(data_2015_df.columns)
X = data_2015_df[['Current health expenditure (% of GDP)',
       'Domestic general government health expenditure (% of total government expenditure)',
       'Employment by industry: Agriculture (%) Female',
       'Employment by industry: Agriculture (%) Male',
       'Employment by industry: Agriculture (%) Male and Female',
       'Employment by industry: Industry (%) Female',
       'Employment by industry: Industry (%) Male',
       'Employment by industry: Industry (%) Male and Female',
       'Employment by industry: Services (%) Female',
       'Employment by industry: Services (%) Male',
       'Employment by industry: Services (%) Male and Female',
       'All staff compensation as % of total expenditure in public institutions (%)',
       'Capital expenditure as % of total expenditure in public institutions (%)',
       'Current expenditure other than staff compensation as % of total expenditure in public institutions (%)',
       'Public expenditure on education (% of GDP)',
       'Infant mortality for both sexes (per 1,000 live births)',
       'Life expectancy at birth for both sexes (years)',
       'Life expectancy at birth for females (years)',
       'Life expectancy at birth for males (years)',
       'Maternal mortality ratio (deaths per 100,000 population)',
       'Population annual rate of increase (percent)',
       'Total fertility rate (children per women)',
       'Gross enrollment ratio - Lower secondary level (female)',
       'Gross enrollment ratio - Lower secondary level (male)',
       'Gross enrollment ratio - Primary (female)',
       'Gross enrollment ratio - Primary (male)',
       'Gross enrollment ratio - Upper secondary level (female)',
       'Gross enrollment ratio - Upper secondary level (male)',
       'Students enrolled in lower secondary education (thousands)',
       'Students enrolled in primary education (thousands)',
       'Students enrolled in upper secondary education (thousands)',
       'Seats held by women in national parliament, as of February (%)',
       'Asylum seekers, including pending cases (number)',
       'International migrant stock: Both sexes (% total population)',
       'International migrant stock: Both sexes (number)',
       'International migrant stock: Female (% total Population)',
       'International migrant stock: Male (% total Population)',
       'Other of concern to UNHCR (number)',
       'Total population of concern to UNHCR (number)',
       'Total refugees and people in refugee-like situations (number)',
       'Labour force participation - Female',
       'Labour force participation - Male',
       'Labour force participation - Total', 'Unemployment rate - Female',
       'Unemployment rate - Male', 'Unemployment rate - Total',
       'Assault rate per 100,000 population',
       'Intentional homicide rates per 100,000']]

X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()

print(results.summary())

p_values = model.pvalues

alpha = 0.05

significant_vars = p_values[p_values < alpha].index
print("Statistically significant variables at alpha = 0.05:")
print(significant_vars)

#seems the most significant variables are related to labour force participation
#%%
#print dataframes to .csvs for use in other applications
data_2015_df.to_csv('data_2015.csv', index=False)
data_2010_df.to_csv('data_2010.csv', index=False)
gdp.to_csv('gdp_data.csv', index=False)
#%%
## WORKING ON CLUSTERING ##
# ## Suppose we were to cluster these countries. What variables would define each cluster? ##

# from sklearn.decomposition import PCA

# pca = PCA()
# pca.fit(X)

# # Get explained variance ratio
# explained_variance = pca.explained_variance_ratio_

# # Plotting the scree plot
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
# plt.title('Scree Plot')
# plt.xlabel('Principal Component')
# plt.ylabel('Explained Variance Ratio')
# plt.xticks(np.arange(1, len(explained_variance) + 1))
# plt.grid(True)
# plt.show()
# #%%
# from sklearn.cluster import KMeans

# pca = PCA(n_components=7)
# X_pca = pca.fit_transform(X)

# inertia = []

# # Define range of clusters (adjust as needed)
# k_range = range(1, 11)

# for k in k_range:
#     kmeans = KMeans(n_clusters=k, random_state=0)
#     kmeans.fit(X_pca)
#     inertia.append(kmeans.inertia_)

# # Plot the inertia values
# plt.plot(k_range, inertia, marker='o')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Inertia')
# plt.title('Elbow Method for Optimal k')
# plt.xticks(k_range)
# plt.show()
# #%%
# kmeans = KMeans(n_clusters=2, random_state=0)

# # Fit the model to the scaled data
# kmeans.fit(X_pca)

# # Predict the cluster labels
# cluster_labels = kmeans.labels_

# # Assign cluster labels back to the original dataframe
# data_2015_df.loc[:, 'Cluster'] = cluster_labels

# cluster_means = data_2015_df.groupby('Cluster').mean()

# # Alternatively, you can visualize the clusters
# import matplotlib.pyplot as plt

# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
# plt.xlabel('GDP')
# plt.ylabel('Population')
# plt.title('Clustering of Countries')
# plt.colorbar()
# plt.show()
