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

def test_normal_distribution(variable_data):
    stat, p = shapiro(variable_data)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # Interpret
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')

# Test each variable in the dataset
print('2010 data:')
for column in data_2010_no.columns:
    print(f'Testing {column} for normal distribution:')
    test_normal_distribution(data_2010_no[column])

print('2015 data:')
for column in data_2015_no.columns:
    print(f'Testing {column} for normal distribution:')
    test_normal_distribution(data_2015_no[column])

#%%
# Assuming 'GDP' is the column name for GDP values in 2015
gdp = data_2015_no['GDP per capita (US dollars)']

# Initialize a dictionary to store correlation coefficients
correlations = {}

# Iterate over the columns and calculate the correlation coefficient with GDP
for column in data_2015_no.columns:
    if column != 'GDP per capita (US dollars)':  # Skip the GDP column itself
        correlations[column] = np.corrcoef(gdp, data_2015_no[column])[0, 1]

# Sort the dictionary by absolute value of correlation coefficients in descending order
sorted_correlations = sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)

# Print out the variables and their correlation coefficients
for variable, coefficient in sorted_correlations:
    print(f'{variable}: {coefficient:.2f}')

print('')
print('Strong correlations:')
# You can also filter out only strong correlations, e.g., |coefficient| > 0.5
strong_correlations = {variable: coefficient for variable, coefficient in sorted_correlations if abs(coefficient) > 0.5}
print('Variables strongly correlated with GDP:', strong_correlations)
#%%
from scipy.stats import mannwhitneyu

# Extract GDP data for 2010 and 2015
gdp_2010 = data_2010_df['GDP per capita (US dollars)']
gdp_2015 = data_2015_df['GDP per capita (US dollars)']

# Perform the Mann-Whitney U test
u_statistic, p_value = mannwhitneyu(gdp_2010, gdp_2015)
print(f"Mann-Whitney U test statistic: {u_statistic}, p-value: {p_value}")

# Interpret the p-value
alpha = 0.05
if p_value > alpha:
    print('The test failed to reject the null hypothesis, suggesting that the distributions of the two groups are similar.')
else:
    print('The test rejected the null hypothesis, suggesting that the distributions of the two groups are different.')
#%%
from scipy.stats import ttest_rel

# Extract GDP data for Country A for 2010 and 2015
gdp_2010 = data_2010_df.loc[data_2010_df['Country'] == 'Czechia', 'GDP per capita (US dollars)'].values[0]
gdp_2015 = data_2015_df.loc[data_2015_df['Country'] == 'Portugal', 'GDP per capita (US dollars)'].values[0]

# Perform the paired t-test
t_statistic, p_value = ttest_rel(gdp_2010, gdp_2015)
print(f"Paired t-test statistic: {t_statistic}, p-value: {p_value}")

# Interpret the p-value
alpha = 0.05
if p_value > alpha:
    print('The test failed to reject the null hypothesis, suggesting no significant difference.')
else:
    print('The test rejected the null hypothesis, suggesting a significant difference.')
