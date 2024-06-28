# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 08:04:22 2024

@author: anyas
"""

#%%
import numpy as np
import pandas as pd
from scipy.stats import norm, expon, uniform, gamma, beta, chi2, logistic, pareto, weibull_min, t

## CREATING SIMULATED DISTRIBUTIONS FOR USE IN TESTING ##
num_bins = 10
num_draws = 15

data15 = {
    'Normal': norm.rvs(size=num_draws),
    'Exponential': expon.rvs(size=num_draws),
    'Uniform': uniform.rvs(size=num_draws),
    'Gamma': gamma.rvs(a=2, size=num_draws),
    'Beta': beta.rvs(a=2, b=5, size=num_draws),
    'Chi-squared': chi2.rvs(df=2, size=num_draws),
    'Logistic': logistic.rvs(loc=0, scale=0.954, size=num_draws),
    'Pareto': pareto.rvs(b=2.62, size=num_draws),
    'Weibull': weibull_min.rvs(c=1.5, size=num_draws),
    'Student\'s t': t.rvs(df=10, size=num_draws)
}

df15 = pd.DataFrame(data15)

hist_data15 = {dist: np.histogram(data15[dist], bins=num_bins) for dist in data15}
observed_frequencies15 = {dist: hist_data15[dist][0] for dist in hist_data15}
num_draws = 100

data100 = {
    'Normal': norm.rvs(size=num_draws),
    'Exponential': expon.rvs(size=num_draws),
    'Uniform': uniform.rvs(size=num_draws),
    'Gamma': gamma.rvs(a=2, size=num_draws),
    'Beta': beta.rvs(a=2, b=5, size=num_draws),
    'Chi-squared': chi2.rvs(df=2, size=num_draws),
    'Logistic': logistic.rvs(loc=0, scale=0.954, size=num_draws),
    'Pareto': pareto.rvs(b=2.62, size=num_draws),
    'Weibull': weibull_min.rvs(c=1.5, size=num_draws),
    'Student\'s t': t.rvs(df=10, size=num_draws)
}

df100 = pd.DataFrame(data100)
hist_data100 = {dist: np.histogram(data100[dist], bins=num_bins) for dist in data100}
observed_frequencies100 = {dist: hist_data100[dist][0] for dist in hist_data100}
num_draws = 1000

data1000 = {
    'Normal': norm.rvs(size=num_draws),
    'Exponential': expon.rvs(size=num_draws),
    'Uniform': uniform.rvs(size=num_draws),
    'Gamma': gamma.rvs(a=2, size=num_draws),
    'Beta': beta.rvs(a=2, b=5, size=num_draws),
    'Chi-squared': chi2.rvs(df=2, size=num_draws),
    'Logistic': logistic.rvs(loc=0, scale=0.954, size=num_draws),
    'Pareto': pareto.rvs(b=2.62, size=num_draws),
    'Weibull': weibull_min.rvs(c=1.5, size=num_draws),
    'Student\'s t': t.rvs(df=10, size=num_draws)
}

df1000 = pd.DataFrame(data1000)
hist_data1000 = {dist: np.histogram(data1000[dist], bins=num_bins) for dist in data1000}
observed_frequencies1000 = {dist: hist_data1000[dist][0] for dist in hist_data1000}
#%%
import pandas as pd
import matplotlib.pyplot as plt

## PLOTTING HISTOGRAMS OF GDP DATA ##
gdp = pd.read_csv(r"C:\Users\anyas\Desktop\Summer Project\Modified CSV Files\GDP_in_current_prices_millions_of_US_dollars.csv")

def histogram(col):
        plt.figure(figsize=(8, 6))
        plt.hist(col, bins=30, edgecolor='black')
        plt.title(f'Histogram of {col}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
        
histogram(gdp['GDP in current prices (millions of US dollars)'])
health_expenditure = pd.read_csv(r"C:\Users\anyas\Desktop\Summer Project\Modified CSV Files\Current_health_expenditure_percent_of_GDP.csv")
histogram(health_expenditure['Current health expenditure (% of GDP)'])
#%%

## PLOTTING HISTOGRAMS OF SIMULATED DATA ##
def create_histograms(df):
    for column in df.columns:
        plt.figure(figsize=(8, 6))
        plt.hist(df[column], bins=30, edgecolor='black')
        plt.title(f'Histogram of {column}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

create_histograms(df15)
create_histograms(df100)
create_histograms(df1000)
#%%

## PLOTTING HISTOGRAM OF SIMULATED CATEGORICAL DATA ##
def plot_categorical_histograms(df):
    for dist, freq in df.items():
        plt.figure(figsize=(8, 6))
        plt.bar(range(len(df)), freq, edgecolor='black')
        plt.title(f'Histogram of {dist}')
        plt.xlabel('Bins')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

plot_categorical_histograms(observed_frequencies15)
plot_categorical_histograms(observed_frequencies100)
plot_categorical_histograms(observed_frequencies1000)
#%%

## INITIALIZING DISTRIBUTIONS ##
from scipy.stats import shapiro, anderson, kstest

dfs = [df15, df100, df1000]

dists = ['norm', 'expon', 'uniform', 'gamma', 
         'beta', 'chi2', 'logistic', 'pareto', 
         'weibull_min', 't']
dist_names = ['Normal', 'Exponential', 'Uniform', 'Gamma', 'Beta', 'Chi Squared', 'Logistic', 'Pareto', 'Weibull', 'T Distribution']
python_dists = [norm, expon, uniform, gamma, beta, chi2, logistic, pareto, weibull_min, t]
test_funcs = [anderson, anderson, kstest, kstest, kstest, kstest, anderson, kstest, kstest, kstest]

alpha = 0.05
#%%
## TESTING TO FIND DISTRIBUTIONS OF CONTINUOUS DATA ##
gdp = gdp.drop(columns={"Country", "Year"})
health_expenditure = health_expenditure.drop(columns={"Country", "Year"})

def check_true(i, pvalue, output):
    if pvalue > alpha:
        output[i] = True

def find_dist(dataframe):
    for col in dataframe:
        length = len(dataframe[col])
        column = dataframe[col].dropna().values
        output = [False] * 10
        
        if length < 20:
            print('Tests may not be accurate with small sample sizes.')
        if length > 999:
            print('Tests may not be accurate with extremely large sample sizes.')
        
        if length < 50:
            _, pvalue = shapiro(column)
            check_true(0, pvalue, output)
            for i in range(1, 10):
                if test_funcs[i] == anderson:
                    result = test_funcs[i](column, dists[i])
                    if result.statistic < result.critical_values[2]:  # Compare against the 5% critical value
                        output[i] = True
                elif test_funcs[i] == kstest:
                    params = python_dists[i].fit(column)
                    _, pvalue = test_funcs[i](column, python_dists[i].cdf, args=params)
                    check_true(i, pvalue, output)
        else:
            for i in range(10):
                if test_funcs[i] == anderson:
                    result = test_funcs[i](column, dists[i])
                    if result.statistic < result.critical_values[2]:  # Compare against the 5% critical value
                        output[i] = True
                elif test_funcs[i] == kstest:
                    params = python_dists[i].fit(column)
                    _, pvalue = test_funcs[i](column, python_dists[i].cdf, args=params)
                    check_true(i, pvalue, output)
        
        if any(output):
            true_indices = [dist_names[index] for index, value in enumerate(output) if value]
            print(f"The following distributions match your data for column {col}: " + ", ".join(true_indices))
        else:
            print(f"No distributions match the data for column {col}.")

find_dist(dfs[0])
find_dist(dfs[1])
find_dist(dfs[2])
find_dist(gdp)
find_dist(health_expenditure)
#%%
import numpy as np
from scipy.stats import gamma, kstest

## BOOTSTRAPPING EXAMPLE ##

# Good at estimating the actual distribution but bad
# at knowing when the distribution is something different

shape, loc, scale = gamma.fit(df1000['Gamma'])
n_bootstrap = 1000
bootstrap_statistics = []

for _ in range(n_bootstrap):
    resample = np.random.choice(df1000['Gamma'], size=len(df1000['Gamma']), replace=True)
    b_shape, b_loc, b_scale = gamma.fit(resample)
    simulated_data = gamma.rvs(b_shape, b_loc, b_scale, size=len(df100['Gamma']))
    stat, _ = kstest(simulated_data, 'gamma', args=(b_shape, b_loc, b_scale))
    bootstrap_statistics.append(stat)
    
ks_stat, pvalue = kstest(df1000['Gamma'], 'gamma', args=(shape, loc, scale))

empirical_p_value = np.mean(np.array(bootstrap_statistics) >= ks_stat)

print("K-S Statistic:", ks_stat)
print(pvalue)
print("Empirical P-Value:", empirical_p_value)
#%%
country_counts = gdp['Country'].value_counts().to_frame()
country_counts_array = country_counts.values

crime = pd.read_csv(r"C:\Users\anyas\Desktop\Summer Project\Crime_Data_from_2020_to_Present.csv")
#%%
crime_statuscounts = crime['Status Desc'].value_counts().to_frame()
crime_area = crime['AREA NAME'].value_counts().to_frame()

plot_categorical_histograms(crime_statuscounts)
plot_categorical_histograms(crime_area)
#%%
import numpy as np
import scipy.stats as stats
import pandas as pd

## TESTING TO FIND DISTRIBUTIONS OF CATEGORICAL DATA - ESTIMATING PARAMS FROM DATA ##
def find_cat1_dist(dataframe):
   for col in dataframe:
        output = []
        observed_counts = dataframe[col]
        for dist in python_dists:
            total_observations = np.sum(observed_counts)
            
            bin_edges = np.linspace(min(observed_counts), max(observed_counts), len(observed_counts) + 1)
            params = dist.fit(observed_counts)
            if dist == gamma:
                print(params)
            num_params = len(params)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            expected_probs = np.diff(dist.cdf(bin_edges, *params))
            expected_counts = expected_probs * total_observations
            dof = len(observed_counts) - 1 - num_params
            # if dist == weibull_min:
            #     print(expected_counts)
            #     print(observed_counts)
            expected_counts = expected_counts * np.sum(observed_counts) / np.sum(expected_counts)
            chi_squared_stat, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts, ddof = dof)
            #print(p_value)
            output.append(
                p_value > 0.05
            )
            
            plt.figure()
            plt.bar(range(len(observed_counts)), observed_counts, alpha=0.5, label='Observed')
            plt.plot(range(len(expected_counts)), expected_counts, 'r-', label='Expected')
            plt.title(f'{dist.name} distribution fit for column {col}')
            plt.xlabel('Bins')
            plt.ylabel('Counts')
            plt.legend()
            plt.show()
                
        if any(output):
            true_indices = [dist_names[index] for index, value in enumerate(output) if value]
            print(f"The following distributions match your data for column {col}: " + ", ".join(true_indices))
        else:
            print(f"No distributions match the data for column {col}.")

#find_cat1_dist(observed_frequencies15)
#find_cat1_dist(observed_frequencies100)
#find_cat1_dist(observed_frequencies1000)
#find_cat1_dist(country_counts)
#%%
find_cat1_dist(crime_statuscounts)
#find_cat1_dist(crime_area)

#%%
## DESCRIPTION OF CONTINUOUS DATASET ##
# Mean
# Mode
# Median
# Variance
# Standard deviation
# Outliers as per IQR
# Min and max, range
# Skew (right, left, normal)
# Kurtosis (heavy/light tailed)
# #%%
# ## DESCRIPTION OF CATEGORICAL DATASET ##
# Feature with highest vs. lowest counts
# Range of counts
# Best fit distributions
# Distributions that this dataset could be drawn from