# **Part #1: Statistical Analysis & Prediction Models**

**The first goal of my project was to determine “gold standard” statistical analysis pipelines for multi-table datasets.** Throughout this section of the project, I researched various statistical tools and predictive modeling techniques. I also compared my findings to those of various LLMs (GPT, Copilot, etc.) and found that LLMs often did not analyze data correctly. I pre-processed and worked with multi-table datasets I found online. A summary document can be found under “Statistics Summary.pdf”.

First, I focused on tests to analyze continuous and categorical distributions. I researched the Shapiro-Wilk, Anderson-Darling, and Kolmogorov-Smirnov tests; I then generated data from various distributions at different sample sizes and ran these tests to better understand their accuracy. Code can be found in "exploratory_data_analysis.py".

Second, I researched various statistical tests (t-tests, ANOVA, etc.) for both parametric and non-parametric datasets. Code can be found in "statistical_tests.py".

Third, we wanted to understand which variables had the greatest influence on an outcome. I researched regression and boosting algorithms for this purpose. I spent the most time on multivariate regression because of the model’s coefficient interpretability; I discussed the various types of regression models (elastic net, lasso, ridge, vanilla) and the conditions under which we would use each. We decided to use elastic net regression in the broader project as it was the most multipurpose but noted that different models may perform better under certain circumstances. I also researched methods for encoding categorical variables. 

We noted that models like CatBoost had higher predictive accuracy (and didn’t require pre-processing of categorical variables) but had lower coefficient interpretability. We also explored PCA and K-means clustering, as well as time-series analysis. Code can be found in "regression_boosting.py".

# **Part #2: Natural Language to SQL Benchmarking**

**The second goal of my project was to improve LLM classification of SQL queries.** In the broader project, we used LLM-generated SQL code. However, sometimes this code wouldn’t return the right data or return no data at all. We wanted to see if GPT models could classify LLM-generated SQL queries as correct or incorrect and generate the correct SQL queries for incorrect responses. A summary document can be found under “LLM Queries Summary.pdf” and my code can be found in “llm_query_optimization.py”.

To do this, we first created a benchmark of 50 natural-language queries from the [Spider](https://yale-lily.github.io/spider) and [BIRD](https://bird-bench.github.io/) datasets, as well as the corresponding LLM-generated SQL queries. About half of these queries were correct and the other half were incorrect. With a vanilla prompt and GPT-4, we had an accuracy of 69%.

| Test                              | Accuracy (%) | 
| --------------------------------- | -------- |
| GPT-4 baseline                    | .69      |
| GPT-4o baseline                   | .75      | 
| CodeLLaMA baseline                | .54      |
| Results after prompt engineering  | .89      | 

I tried different prompt engineering strategies and models and eventually got the accuracy up to 89%. 

Then, I added more complex queries from a synthetic dataset. Unfortunately, the  baseline GPT-4o accuracy dropped to 60%. I found that most of the classification errors could be resolved in ChatGPT's Playground after some back-and-forth with the model, but I couldn’t figure out a way to standardize this in my prompt.

| Test                              | Accuracy (%) | 
| --------------------------------- | -------- |
| GPT-4o baseline                   | .60      |
| Results after prompt engineering  | .70      | 

The highest accuracy I could get after prompt engineering was 70%. 



