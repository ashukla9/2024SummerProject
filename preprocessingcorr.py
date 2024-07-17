# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:02:34 2024

@author: anyas
"""
#%%
import pandas as pd
listings = pd.read_csv(r"C:\Users\anyas\Downloads\listings.csv.gz", compression='gzip')
calendar = pd.read_csv(r"C:\Users\anyas\Downloads\calendar.csv.gz", compression='gzip')
#%%
calendar.rename(columns={'listing_id': 'id'}, inplace=True)
filtered_calendar = calendar[calendar['date'] == '2024-05-06']
#%%
merged_df = pd.merge(listings, filtered_calendar, on=['id'], how='inner')
#%%
print(merged_df.columns)
#%%
unique_values = calendar['date'].unique()
print(unique_values)
value_counts = calendar['date'].value_counts()
#%%
numeric_columns = calendar.select_dtypes(include=['float64', 'int64']).columns
final_df = merged_df[numeric_columns]
print(final_df.columns)
#%%
# Get the count for the specific date
count_specific_date = value_counts.get('2024-05-06', 0)

print(count_specific_date)

#%%
country = pd.read_csv(r"C:\Users\anyas\Desktop\Summer Project\Cars_Country.csv")
cars = pd.read_csv(r"C:\Users\anyas\Desktop\Summer Project\Cars_Multi.csv")
price = pd.read_csv(r"C:\Users\anyas\Desktop\Summer Project\Cars_Price.csv")

merged_cars = pd.merge(cars, price, on=['ID'], how='inner')

#%%
gdp = pd.read_csv(r"C:\Users\anyas\Desktop\Summer Project\gdp_data.csv")
gdp = gdp[(gdp['Year'] == 2005) & (gdp['Series'] == 'GDP in constant 2015 prices (millions of US dollars)')]
#%%
globe1 = pd.read_csv(r"C:\Users\anyas\Desktop\Summer Project\GLOBE-Phase-2-Aggregated-Leadership-Data.csv")
globe2 = pd.read_csv(r"C:\Users\anyas\Desktop\Summer Project\GLOBE-Phase-2-Aggregated-Societal-Culture-Data.csv")

merged_globe = pd.merge(globe1, globe2, on=['Country Name'], how='inner')
pd.set_option('display.max_rows', None)

# Print all values in a column
print(globe1['Country Name'])
print('South Africa' in globe1['Country Name'])
#%%
pd.set_option('display.max_rows', None)

# Print all values in a column
print(gdp['Country'])
#%%
gdp.rename(columns={'Country': 'Country Name'}, inplace=True)
merged_globe = pd.merge(merged_globe, gdp, on=['Country Name'], how='inner')
#%%
import pyarrow.parquet as pq
trips = pq.read_table(r"C:\Users\anyas\Desktop\Summer Project\yellow_tripdata_2024-01 (1).parquet")
trips = trips.to_pandas()
print(trips.columns)