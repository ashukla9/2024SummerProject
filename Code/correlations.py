# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 14:47:52 2024

@author: anyas
"""

import numpy as np
import pandas as pd

shootings_direct = pd.read_csv(r"C:\Users\anyas\Desktop\Summer Project\Correlation CSVs\shootings1.csv")
shootings_indirect = pd.read_csv(r"C:\Users\anyas\Desktop\Summer Project\Correlation CSVs\shootings2.csv")
#%%
shootings_indirect['OCCUR_DATE'] = pd.to_datetime(shootings_indirect['OCCUR_DATE'])
shootings_indirect['OCCUR_TIME'] = pd.to_datetime(shootings_indirect['OCCUR_TIME'])
#%%
import matplotlib.pyplot as plt
import seaborn as sns

shootings_indirect.replace('(null)', np.nan, inplace=True)

direct_correlations = shootings_direct.corr()
rounded_correlation = direct_correlations.round(2)
plt.figure(figsize=(8, 6))
sns.heatmap(rounded_correlation, annot=True, cmap='coolwarm', center=0, linewidths=0.5, annot_kws={"fontsize": 12})
plt.title('Correlation Heatmap', fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
#%%
monthly_gender = shootings_indirect.groupby([pd.Grouper(key='OCCUR_DATE', freq='M'), 'VIC_SEX']).size().unstack(fill_value=0)
monthly_gender = monthly_gender.reset_index()
monthly_gender.columns.name = None
monthly_gender.columns = ['Month', 'Female', 'Male', 'Unknown Gender']

monthly_race = shootings_indirect.groupby([pd.Grouper(key='OCCUR_DATE', freq='M'), 'VIC_RACE']).size().unstack(fill_value=0)
monthly_race = monthly_race.reset_index()
monthly_race.columns.name = None
monthly_race.columns = ['Month', 'AMERICAN INDIAN/ALASKAN NATIVE', "ASIAN / PACIFIC ISLANDER", "BLACK", "BLACK HISPANIC", "UNKNOWN", "WHITE", "WHITE HISPANIC"]

monthly_age = shootings_indirect.groupby([pd.Grouper(key='OCCUR_DATE', freq='M'), 'VIC_AGE_GROUP']).size().unstack(fill_value=0)
monthly_age = monthly_age.reset_index()
monthly_age.columns.name = None
monthly_age.columns = ['Month', '1022', '18-24', '25-44', '45-64', '65+', '<18', 'Unknown Age']

perp_gender = shootings_indirect.groupby([pd.Grouper(key='OCCUR_DATE', freq='M'), 'PERP_SEX']).size().unstack(fill_value=0)
perp_gender = perp_gender.reset_index()
perp_gender.dropna(inplace=True)
perp_gender.columns.name = None
perp_gender.columns = ['Month', 'Perp Female', 'Perp Male', 'Perp Unknown Gender']

jurisd = shootings_indirect.groupby([pd.Grouper(key='OCCUR_DATE', freq='M'), 'JURISDICTION_CODE']).size().unstack(fill_value=0)
jurisd = jurisd.reset_index()
jurisd.columns.name = None
jurisd.columns = ['Month', 'NYPD Patrol', 'NYPD Transit', 'NYPD Housing']

precinct = shootings_indirect.groupby([pd.Grouper(key='OCCUR_DATE', freq='M'), 'PRECINCT']).size().unstack(fill_value=0)
precinct = precinct.reset_index()
precinct.rename(columns=lambda x: f'Precinct_{x}', inplace=True)
precinct.rename(columns={'Precinct_OCCUR_DATE': 'Month'}, inplace=True)

shootings_indirect['Hour'] = shootings_indirect['OCCUR_TIME'].dt.hour
shootings_indirect['Period'] = pd.cut(shootings_indirect['OCCUR_TIME'].dt.hour, bins=[0, 12, 18, 24], labels=['Morning', 'Afternoon', 'Evening'])

hour = shootings_indirect.groupby([pd.Grouper(key='OCCUR_DATE', freq='M'), 'Hour']).size().unstack(fill_value=0)
hour = hour.reset_index()
hour.rename(columns=lambda x: f'Hour_{x}', inplace=True)
hour.rename(columns={'Hour_OCCUR_DATE': 'Month'}, inplace=True)

period = shootings_indirect.groupby([pd.Grouper(key='OCCUR_DATE', freq='M'), 'Period']).size().unstack(fill_value=0)
period = period.reset_index()
period.columns.name = None
period.columns = ['Month', 'Morning', 'Afternoon', 'Evening']

perp_race = shootings_indirect.groupby([pd.Grouper(key='OCCUR_DATE', freq='M'), 'PERP_RACE']).size().unstack(fill_value=0)
perp_race = perp_race.reset_index()
perp_race.dropna(inplace=True)
perp_race.columns.name = None
perp_race.columns = ['Month', 'Perp AMERICAN INDIAN/ALASKAN NATIVE', "Perp ASIAN / PACIFIC ISLANDER", "Perp BLACK", "Perp BLACK HISPANIC", "Perp UNKNOWN", "Perp WHITE", "Perp WHITE HISPANIC"]

def categorize_race_sex(race, sex):
    return race + ' ' + sex

shootings_indirect['Victim Category'] = shootings_indirect.apply(lambda x: categorize_race_sex(x['VIC_RACE'], x['VIC_SEX']), axis=1)
monthly_aggreg = shootings_indirect.groupby([pd.Grouper(key='OCCUR_DATE', freq='M'), 'Victim Category']).size().unstack(fill_value=0)
monthly_aggreg = monthly_aggreg.reset_index()
monthly_aggreg.rename(columns={'OCCUR_DATE': 'Month'}, inplace=True)
#%%
print(shootings_indirect.columns)
shootings_indirect = shootings_indirect.drop(columns={'SHOOTING_NOT_MURDER', 'SHOOTING_MURDER',
       'PERP_SEX', 'PERP_RACE', 'VIC_AGE_GROUP', 'VIC_SEX', 'VIC_RACE',
       'Victim Category'})
#%%
merged_df = pd.merge(monthly_race, monthly_gender, on='Month')
merged_df = pd.merge(merged_df, monthly_age, on='Month')
merged_df = pd.merge(merged_df, monthly_aggreg, on='Month')
merged_df = pd.merge(merged_df, perp_race, on='Month')
merged_df = pd.merge(merged_df, perp_gender, on='Month')
merged_df = pd.merge(merged_df, precinct, on='Month')
merged_df = pd.merge(merged_df, jurisd, on='Month')
merged_df = pd.merge(merged_df, hour, on='Month')
merged_df = pd.merge(merged_df, period, on='Month')
#%%
correlations = merged_df.corrwith(shootings_direct['Murder'])
correlations = correlations[(correlations.abs() > 0.5) & (correlations != 1.0)]
print(correlations)
#%%
correlations = shootings_indirect.corrwith(shootings_direct['Murder'])
print(correlations)
