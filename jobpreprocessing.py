# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:47:16 2024

@author: anyas
"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
#%%
df = pd.read_csv(r"C:\Users\anyas\Desktop\Summer Project\Correlation CSVs\USvideos.csv")
import json

# Step 1: Load the JSON file
with open(r"C:\Users\anyas\Downloads\US_category_id.json", 'r') as f:
    categories_data = json.load(f)
#%%
categories = [{'category_id': item['id'], 'title': item['snippet']['title']} for item in categories_data['items']]
# Step 2: Convert JSON data to DataFrame (if applicable)
# For demonstration, let's assume categories_data is a list of dictionaries
categories_df = pd.DataFrame(categories)
#%%
df['category_id'] = df['category_id'].astype(str)
#%%
print(categories_df['title'].unique)
#%%
merged_df = pd.merge(df, categories_df, on='category_id')
print(merged_df)
#%%
merged_df.to_csv(r"C:\Users\anyas\Desktop\Summer Project\Correlation CSVs\video_titles.csv")
#%%
title_desc = pd.read_csv(r"C:\Users\anyas\Downloads\Categorized_Video_Data.csv")
lda = pd.read_csv(r"C:\Users\anyas\Downloads\LDA_Categorized_Video_Data.csv")
#%%
title_selected = title_desc[['video_id', 'category']]
title_selected.rename(columns={'category': 'title_desc category'}, inplace=True)
print(title_selected)
# #%%
# lda_selected = lda[['video_id', 'category']]
# lda_selected.rename(columns={'category': 'lda_category'}, inplace=True)
# print(lda_selected)
#%%
merged_df_unique = merged_df.drop_duplicates(subset='video_id')
title_unique = title_selected.drop_duplicates(subset='video_id')
new_merged_df = pd.merge(merged_df_unique, title_unique, on='video_id')
print(new_merged_df)
#%%
print(new_merged_df.columns)
new_merged_df['Pred Correct'] = new_merged_df['title_desc category'] == new_merged_df['title_y']
print((new_merged_df['Pred Correct'] == True).sum() / len(new_merged_df['Pred Correct']))
# #%%
# new_merged_df = pd.merge(new_merged_df, lda_selected, on='video_id')
#%%
print(new_merged_df)