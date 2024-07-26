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
import gensim
import seaborn as sns
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

# airbnb_df = merged_df.select_dtypes(include=[int, float])
#airbnb_df = airbnb_df.dropna(subset=['price_duplicate'])
airbnb_df = merged_df.loc[:, ['price_duplicate', 'description']]
airbnb_df.rename(columns={'price_duplicate': 'price'}, inplace=True)
airbnb_df = airbnb_df.dropna(subset=['price'])
#%%
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import gensim.downloader as api

# Using smallest model for my machine but there are much larger models available
model = api.load('glove-wiki-gigaword-100')

# Check the dimension of word vectors
print("Vector size:", model.vector_size)
#%%
airbnb_df['description'] = airbnb_df['description'].fillna('')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    doc = word_tokenize(text)
    doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word.isalpha()]
    return doc

# Function to check if document has vector representation
def has_vector_representation(word2vec_model, doc):
    """Check if at least one word of the document is in the word2vec dictionary."""
    return any(word in word2vec_model.key_to_index for word in doc)

# Function to compute the document vector
def document_vector(word2vec_model, doc):
    # Remove out-of-vocabulary words
    doc = [word for word in doc if word in word2vec_model.key_to_index]
    return np.mean(word2vec_model[doc], axis=0)

# Preprocess descriptions
airbnb_df['processed_descriptions'] = airbnb_df['description'].apply(preprocess)

# Filter out descriptions with no vector representations
filtered_indices = [i for i, doc in enumerate(airbnb_df['processed_descriptions']) if has_vector_representation(model, doc)]

# Compute document vectors for each description
filtered_docs = [airbnb_df['processed_descriptions'].iloc[i] for i in filtered_indices]
document_vectors = np.array([document_vector(model, doc) for doc in filtered_docs])

# Create a DataFrame with the document vectors
doc_vec_df = pd.DataFrame(document_vectors, index=filtered_indices)

# Ensure the target variable matches the filtered documents
filtered_target = airbnb_df['price'].iloc[filtered_indices]

#%%
# # Example usage of document vectors with t-SNE for visualization (optional)
# from sklearn.manifold import TSNE
# import seaborn as sns
# import matplotlib.pyplot as plt
# from adjustText import adjust_text

# tsne = TSNE(n_components=2, init='random', random_state=10, perplexity=100)
# tsne_df = tsne.fit_transform(document_vectors[:400])  # Use only the first 400 document vectors for t-SNE

# # Plotting
# fig, ax = plt.subplots(figsize=(14, 10))
# sns.scatterplot(x=tsne_df[:, 0], y=tsne_df[:, 1], alpha=0.5)

# # Initialize list of texts
# texts = []
# titles_to_plot = list(np.arange(0, len(tsne_df), 40))  # Plot every 40th title in the first 400 titles

# # Append words to list
# for title in titles_to_plot:
#     texts.append(plt.text(tsne_df[title, 0], tsne_df[title, 1], ' '.join(filtered_docs[title][:3]), fontsize=14))

# # Plot text using adjust_text (because overlapping text is hard to read)
# adjust_text(texts, force_points=0.4, force_text=0.4,
#             expand_points=(2, 1), expand_text=(1, 2),
#             arrowprops=dict(arrowstyle="-", color='black', lw=0.5))

# plt.show()
#%%
from sklearn.decomposition import PCA

def pca(df):
    pca = PCA()
    X = df
    pca.fit(X)
    
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

def preprocessing(df, target_column, outliers=False):
    X = df
    y = target_column
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    
    # train_indices = X_train.index
    # test_indices = X_test.index
    # test_columns = X_test.columns
    # train_columns = X_train.columns
    
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

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
X_train, X_test, y_train, y_test = preprocessing(doc_vec_df, filtered_target)
# pca(X_train)
#lin_reg(X_train.copy(), X_test.copy(), y_train, y_test, vif=True)
elastic_net(X_train, X_test, y_train, y_test)
randomized_lasso(X_train, X_test, y_train, y_test)
ridge(X_train, X_test, y_train, y_test)
#huber(X_train.copy(), X_test.copy(), y_train, y_test, vif = True)