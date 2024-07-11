# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 14:58:26 2024

@author: anyas
"""

import numpy as np
import pandas as pd

pet_outtake = pd.read_csv(r"C:\Users\anyas\Desktop\Summer Project\Correlation CSVs\Austin_Animal_Center_Outcomes_20240703.csv")
pet_intake = pd.read_csv(r"C:\Users\anyas\Desktop\Summer Project\Correlation CSVs\Austin_Animal_Center_Intakes_20240703.csv")
#%%
pet_outtake['DateTime'] = pd.to_datetime(pet_outtake['DateTime'])
pet_intake['DateTime'] = pd.to_datetime(pet_intake['DateTime'])
pet_outtake['Date of Birth'] = pd.to_datetime(pet_outtake['Date of Birth'])
#%%
df1_unique = pet_intake.drop_duplicates(subset='Animal ID')
df2_unique = pet_outtake.drop_duplicates(subset='Animal ID')
# #%%
# df2_unique = df2_unique.drop(columns={'Name', 'Breed', 'Color', 'Animal Type'})
# df2_unique.rename(columns={'Outcome Type': 'Outcome_Type'}, inplace=True)
# df1_unique.rename(columns={'MonthYear': 'Intake Month/Year'}, inplace=True)
# df2_unique.rename(columns={'MonthYear': 'Outtake Month/Year'}, inplace=True)

# def outcome(outcome):
#     if outcome == 'Adoption':
#         return 'Adoption'
#     elif outcome == 'Euthanasia':
#         return 'Euthanasia'
#     elif outcome == 'Return to Owner':
#         return 'Return to Owner'
#     else:
#         return 'Other Outcome'

# df2_unique['Outcome_Type'] = df2_unique['Outcome_Type'].apply(outcome)

# df2_unique = df2_unique.drop(columns={'Outcome Subtype'})
# df1_unique['Name'] = df1_unique['Name'].fillna('None')
# df2_unique = df2_unique.dropna()
# df1_unique = df1_unique.dropna()

# # Save the DataFrame to a CSV file
# df1_unique.to_csv(r"C:\Users\anyas\Desktop\Summer Project\Correlation CSVs\intake.csv", index=False)
# df2_unique.to_csv(r"C:\Users\anyas\Desktop\Summer Project\Correlation CSVs\outtake.csv", index=False)

merged_pets = pd.merge(df1_unique, df2_unique, on='Animal ID', how='inner')
merged_pets = merged_pets.drop(columns={'Name_y', 'Breed_y', 'Color_y', 'Animal Type_y'})
#%%
merged_pets.rename(columns={'Outcome Type': 'Outcome_Type'}, inplace=True)
merged_pets.rename(columns={'Name_x': 'Intake Name'}, inplace=True)
merged_pets.rename(columns={'MonthYear_x': 'Intake Month/Year'}, inplace=True)
merged_pets.rename(columns={'MonthYear_y': 'Outtake Month/Year'}, inplace=True)
merged_pets.rename(columns={'Animal Type_x': 'Intake Animal Type'}, inplace=True)
merged_pets.rename(columns={'Breed_x': 'Intake Breed'}, inplace=True)
merged_pets.rename(columns={'Color_x': 'Intake Color'}, inplace=True)

def outcome(outcome):
    if outcome == 'Adoption':
        return 'Adoption'
    elif outcome == 'Euthanasia':
        return 'Euthanasia'
    elif outcome == 'Return to Owner':
        return 'Return to Owner'
    else:
        return 'Other Outcome'

merged_pets['Outcome_Type'] = merged_pets['Outcome_Type'].apply(outcome)

merged_pets = merged_pets.drop(columns={'Outcome Subtype'})
merged_pets['Intake Name'] = merged_pets['Intake Name'].fillna('None')
merged_pets = merged_pets.dropna()
#%%
## ANOVA CORRELATIONS - CONTINUOUS / CATEGORICAL VARIABLES ##
## DIRECT CORRELATIONS ##
import statsmodels.api as sm
from statsmodels.formula.api import ols
import datetime as dt
from statsmodels.stats.multicomp import pairwise_tukeyhsd

## low p-value in any of these tests - sign that there is association between groups ##

# standardize the dates so that it can be treated as a continuous variable
def anova_tukey(column, x):
    merged_pets[column] = (merged_pets[x] - merged_pets[x].min()).dt.days
    merged_pets[column] = pd.to_numeric(merged_pets[column], errors='coerce')
    model = ols(f'{column} ~ C(Outcome_Type)', data=merged_pets).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print('ANOVA results for Date of Intake/Outcome Type')
    print(anova_table)
    tukey = pairwise_tukeyhsd(endog=merged_pets[column], groups=merged_pets['Outcome_Type'], alpha=0.05)
    print(tukey)
    print('')

anova_tukey('DateNumeric_Intake', 'DateTime_x')
anova_tukey('DateNumeric_Outtake', 'DateTime_y')
anova_tukey('DateNumeric_Birth', 'Date of Birth')
#%%
## CHI SQUARED CORRELATIONS - CATEGORICAL / CATEGORICAL VARIABLES ##
## DIRECT CORRELATIONS ##

## low p-value in any of these tests - sign that there is association between groups ##
## Cramer's V provides info about strength of association ##

from scipy.stats import chi2_contingency
import math

column_names = ['Animal ID', 'Intake Name', 'Intake Month/Year', 'Found Location',
       'Intake Type', 'Intake Condition', 'Intake Animal Type', 'Sex upon Intake',
       'Age upon Intake', 'Intake Breed', 'Intake Color',
       'Outtake Month/Year', 'Date of Birth',
       'Sex upon Outcome', 'Age upon Outcome']

def contcont(column_names, df):
    for col in column_names:
        contingency_table = pd.crosstab(df[col], df['Outcome_Type'])
        chi2, p, dof, ex = chi2_contingency(contingency_table)
    
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = math.sqrt(chi2 / (n * min_dim))
        
        alpha = 0.05
        if p < alpha:
            print("Reject the null hypothesis. There is an association between Outcome_Type and", col)
            if cramers_v < .3:
                print("There is a weak association.")
                print(f"Cramer's V: {cramers_v}")
            elif cramers_v < .5:
                print("There is a moderate association.")
                print(f"Cramer's V: {cramers_v}")
            else:
                print("There is a strong association.")
                print(f"Cramer's V: {cramers_v}")
        else:
            print("Fail to reject the null hypothesis. There is no association between Outcome_Type and", col)
#%%

## ONE-HOT ENCODING LOGISTIC REGRESSION ##

# column_names = ['Intake Name', 'Intake Month/Year', 'Found Location',
#        'Intake Type', 'Intake Condition', 'Intake Animal Type', 'Sex upon Intake',
#        'Age upon Intake', 'Intake Breed', 'Intake Color', 
#        'Outtake Month/Year',
#        'Sex upon Outcome', 'Age upon Outcome']

# def rename_small_categories(df, col):
#     outcome_counts = df[col].value_counts()
#     categories_to_rename = outcome_counts[outcome_counts < 10].index
#     df[col] = df[col].apply(lambda x: f'Other_{col.replace(" ", "_")}' if x in categories_to_rename else x)

# for col in column_names:
#     rename_small_categories(merged_pets, col)
# #%%
# pets_encoded = pd.get_dummies(merged_pets, columns=column_names)
# #%%
# pets_encoded = pets_encoded.drop(columns={'DateTime_x'
#        'DateTime_y', 'Date of Birth', 'Outcome_Type'})
# #%%
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report

# X = pets_encoded.drop(columns=['Animal ID'])
# y = merged_pets['Outcome_Type']

# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# print("Classification report:")
# print(classification_report(y_test, y_pred))

# coefficients = model.coef_

# feature_names = X.columns
# outcome_labels = model.classes_
# coef_df = pd.DataFrame(coefficients.T, index=feature_names, columns=[f'Outcome_{label}' for label in outcome_labels])

# for i, outcome in enumerate(outcome_labels):
#     print(f"\nTop 10 features most associated with Outcome '{outcome}' based on absolute value:")
#     top_features = coef_df[f'Outcome_{outcome}'].abs().sort_values(ascending=False).head(10)
#     print(top_features)
#%%

## TARGET ENCODING LOGISTIC REGRESSION ##

from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

def logistic_reg(df):
    df['Outcome_Type'] = df['Outcome_Type'].astype('category')
    X = df.drop(columns={'DateTime_x',
            'DateTime_y', 'Date of Birth', 'Outcome_Type'})
    y = df['Outcome_Type']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    encoder = TargetEncoder(smooth=0.1, target_type = 'multiclass')
    #creates features * target categories number of columns
    X_train_encoded = encoder.fit_transform(X_train, y_train)
    X_test_encoded = encoder.transform(X_test)
    
    feature_names = encoder.feature_names_in_
    
    if encoder.target_type_ == 'multiclass':
        column_names = []
        for feature in feature_names:
            for class_label in encoder.classes_:
                column_names.append(f"{feature}_class_{class_label}")
    else:
        column_names = feature_names
    
    X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=column_names)
    X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=column_names)

## MODIFY TO HAPPEN ABOVE 
    def calculate_vif(df):
        vif = pd.DataFrame()
        vif['Feature'] = df.columns
        vif['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
        return vif
    
    while True:
        vif = calculate_vif(X_train_encoded_df)
        max_vif = vif["VIF"].max()
        if max_vif > 10:
            feature_to_drop = vif.loc[vif["VIF"].idxmax(), "Feature"]
            print(f"Dropping feature '{feature_to_drop}' with VIF {max_vif}")
            X_train_encoded_df.drop(columns=[feature_to_drop], inplace=True)
            X_test_encoded_df.drop(columns=[feature_to_drop], inplace=True)
        else:
            break

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter = 1000)
    model.fit(X_train_encoded_df, y_train)
    
    y_pred = model.predict(X_test_encoded_df)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Accuracy Score:")
    print(accuracy_score(y_test, y_pred))
    
    coefficients = model.coef_
    coef_df = pd.DataFrame(coefficients.T, index=X_train_encoded_df.columns, columns=[f'Outcome_{label}' for label in model.classes_])
    
    for i, outcome in enumerate(model.classes_):
        print(f"\nTop 10 features most associated with Outcome '{outcome}' based on absolute value:")
        top_features = coef_df[f'Outcome_{outcome}'].abs().sort_values(ascending=False).head(10)
        print(top_features)
#%%
logistic_reg(merged_pets)
#%%
from catboost import CatBoostClassifier
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from catboost import Pool, EFstrType

def catboost_predict(df):
    X = df.drop(columns={'Outcome_Type', 'Date of Birth', 'DateTime_y', 'DateTime_x'})
    y = df['Outcome_Type']
    
    cat_features = [X.columns.get_loc(col) for col in X.select_dtypes(include=['object']).columns]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    #Change loss function depending on classification type
    model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, loss_function='MultiClass', verbose=False)
    
    model.fit(X_train, y_train, cat_features=cat_features)
    
    y_pred = model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Accuracy Score:")
    print(accuracy_score(y_test, y_pred))
    
    feature_importances = model.get_feature_importance()
    
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    print(importance_df)
    
    shap_values = model.get_feature_importance(Pool(X, y, cat_features=cat_features), type=EFstrType.ShapValues)
    shap_values_transposed = shap_values.transpose(1, 0, 2)
    shap.summary_plot(list(shap_values_transposed[:,:,:-1]), features=X, class_names=y.unique(), plot_type='bar')
#%%
contcont(column_names, merged_pets)
catboost_predict(merged_pets)
#%%

## ADDING INDIRECT VARIABLES ##
def categorize_time_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'
    
def categorize_time_of_year(month):
    if 3 <= month < 5:
        return 'Spring'
    elif 6 <= month < 8:
        return 'Summer'
    elif 9 <= month < 11:
        return 'Fall'
    else:
        return 'Winter'
    
cmerged_pets = merged_pets.copy()
cmerged_pets['First Letter'] = cmerged_pets['Intake Name'].str[0]
cmerged_pets['Name Length'] = cmerged_pets['Intake Name'].str.len()
cmerged_pets['No Name'] = cmerged_pets['Intake Name'] == 'None'
cmerged_pets['Austin'] = cmerged_pets['Found Location'].str.contains('Austin \(TX\)')
cmerged_pets['Intake Season'] = cmerged_pets['DateTime_x'].dt.month.apply(categorize_time_of_year)
cmerged_pets['Intake Time of Day'] = cmerged_pets['DateTime_x'].dt.hour.apply(categorize_time_of_day)
cmerged_pets['Outtake Season'] = cmerged_pets['DateTime_y'].dt.month.apply(categorize_time_of_year)
cmerged_pets['Outtake Time of Day'] = cmerged_pets['DateTime_y'].dt.hour.apply(categorize_time_of_day)
cmerged_pets['Age Breed Combo'] = cmerged_pets['Intake Breed'] + ' ' + cmerged_pets['Age upon Intake']
cmerged_pets['Age Condition Combo'] = cmerged_pets['Intake Condition'] + ' ' + cmerged_pets['Age upon Intake']
cmerged_pets['Intake Type Condition Combo'] = cmerged_pets['Intake Type'] + ' ' + cmerged_pets['Intake Condition']
cmerged_pets['Age Intake Time Combo'] = cmerged_pets['Intake Month/Year'] + ' ' + cmerged_pets['Age upon Intake']
cmerged_pets['Time in Shelter'] = (cmerged_pets['DateTime_y'] - cmerged_pets['DateTime_x']).dt.days
#%%
column_names_ind = ['Animal ID', 'Intake Name', 'Intake Month/Year',
       'Found Location', 'Intake Type', 'Intake Condition',
       'Intake Animal Type', 'Sex upon Intake', 'Age upon Intake',
       'Intake Breed', 'Intake Color', 'Outtake Month/Year', 'Sex upon Outcome', 'Age upon Outcome',
       'First Letter', 'Name Length', 'No Name', 'Austin', 'Intake Season',
       'Intake Time of Day', 'Outtake Season', 'Outtake Time of Day',
       'Age Breed Combo', 'Age Condition Combo', 'Intake Type Condition Combo',
       'Age Intake Time Combo', 'Time in Shelter']
contcont(column_names_ind, cmerged_pets)
#%%
catboost_predict(cmerged_pets)
#%%
logistic_reg(cmerged_pets)