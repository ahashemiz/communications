# Communications

Team Members: Richard Aw, Lakshmi Manne, Shambhavi Gupta, Arman Hashemizadeh

## Getting Started

In this linear regression notebook, we use several python packages. Here is how you can install them from the command-line:

`pip install pandas, matplotlib, seaborn, statsmodels`

## Description

This jupyter notebook contains a simple example of how to perform and analyze the results from a simple linear regression model using the statsmodel python package.
The dataset is the WHO Life Expectancy Dataset, which can be found on Kaggle: <a href="https://www.kaggle.com/kumarajarshi/life-expectancy-who?select=Life+Expectancy+Data.csv">WHO Life Expectancy Dataset</a>

We regress a country's average life expectancy on its citizens' average number of years spent in school. Surprisingly, we found a high level of correlation
between these two variables in this dataset. 

This notebook also includes analysis of the assumptions of the linear model as well as to what degree the data 
follows these assumptions.

## About This Guide

This guide is one of the deliverables of a group project for Communications for Analytics class (MSDS 610). MSDS 610 is a core class in the Master of Science in Data Science program at the University of San Francisco.

The goal of this guide is to provide a concise and “hands-on” introduction to linear regression, with an emphasis on simple linear regression (SLR). This is to keep things as simple as possible whilst still conveying the essence of linear regression. 

This guide is meant to be accessible to those with just enough (bittersweet) memories of introductory probability, statistics, linear algebra, and Python programming. To this end, we have structured the guide as follows: 

(1) SLR in a Nutshell — “No-frills” overview of the essentials and motivation for SLR.

(2) SLR in Practice — End-to-end tutorial on implementing SLR in Python.

## SLR in a nutshell

First...what is linear regression?

Linear regression is the process of estimating the coefficients in the equation for the straight line (2D case) or plane (3D and higher dimensions) that best fits some given data. Thus, to perform linear regression is to (i) find the numbers that specify the line/plane of best fit and (ii) determine how confident we can be in those numbers. 

(Remember from Physics labs that whenever we are estimating the value of something like height, we don’t just state our guess but provide error bounds for our guess as well? This is because there is no guarantee that our guess is the true value. It is exactly the same idea in linear regression!) (Could put in a textbox with a light color hue; something like ‘FYI’ boxes in textbooks.)
There are two types of linear regression: simple linear regression (SLR) and multiple linear regression (MLR). 

SLR is performed when we regress the response variable (Y) against only one predictor variable (X) — this is why it is dubbed “simple”. Example: regressing “price of milk” against “milk’s shelf life”.

MLR is performed when there are multiple predictor variables (two or more X’s). Example: regressing “price of milk” against “milk’s shelf life”, “milk’s tastiness score”, and “milk’s nutrition score”.

An intuitive way to grasp the difference between SLR and MLR is to visualize both:

![Alt text](readme_imgs/SlrVsMlr.png?raw=true "SLR Vs MLR")
SLR: fitting a line of best fit vs MLR: fitting a plane of best fit. Note that the 3D plot on the right depicts MLR in the case of two predictor variables; MLR also includes cases where there are even more predictor variables. (<a href="https://www.keboola.com/blog/linear-regression-machine-learning">Image Source</a>)

## Why Care About Linear Regression?

Linear regression enables us to carry out inference and prediction.

Inference is about understanding the relationship between variables in the given data. Regressing a chosen response variable against one or more predictor variables would enable us to evaluate not only the significance of each of the predictor variables on the response variable (e.g., does “milk’s shelf life” have a significant influence on “price of milk”?), but also the relative importance of the predictor variables (e.g., which has a greater impact on “price of milk” — “milk’s shelf life” or “milk’s tastiness score”?).

Prediction is about computing projected values for the response variable based on new input values of the predictor variables. Since linear regression gives us an equation that relates the response variable (e.g., “price of milk”) to the predictor variables (e.g., “milk’s shelf life”, “milk’s tastiness score”, etc.), we can ‘feed’ new values for “milk’s shelf life” and “milk’s tastiness score” to the equation and obtain a projected value for “price of milk”. In this way, linear regression enables us to predict the future given new data!

Of course, it goes without saying that linear regression is practically useful. Inference and prediction are both critical to the commercial success of many companies, and increasingly so given that the world economy is becoming increasingly ‘data-driven’! (Could put in a textbox with a light color hue; something like ‘FYI’ boxes in textbooks.)

## SLR in Practice

Let’s start with a “real-world” data set - something with many potential predictor variables. For the sole purpose of seeing SLR in practice, we want to find two variables with values that are sufficiently correlated. To do so, we can plot a correlation matrix:

```python
df = pd.read_csv('Life Expectancy Data.csv')
# Focusing on 2015
df_2015 = df[df['Year'] == 2015].drop(columns='Year')
df_2015_subset = df_2015[['Country', 'Status', 'Life expectancy ',
                          'Adult Mortality', 'infant deaths', 'Hepatitis B',
                          'Measles ', ' BMI ', 'Diphtheria ',
                          ' HIV/AIDS', 'GDP', 'Schooling']]
corr_matrix_2015 = df_2015_subset.corr()
plt.figure(figsize=(12, 5))
sns.heatmap(data=corr_matrix_2015, annot=True, cbar=False)
```

![Alt text](readme_imgs/corrMatrix.png?raw=true "seaborn heatmap of the correlation matrix")

How do we obtain the numbers that specify the line of best fit? There are various methods to do so. Ordinary least squares estimation (OLSE) — wherein “best” means minimizing the sum of squared errors — is probably the most well-known method. Maximum likelihood estimation (MLE) — wherein “best” means maximizing the probability that each of the random error terms in our linear model is normally distributed — is another method. (Coincidentally, MLE and OLSE lead to the same results in SLR!) 

(OLSE should not be conflated with linear regression itself! The latter is a general process that may be initiated using a specific method, while the former is simply a specific method.) (Could put in a textbox with a light color hue; something like ‘FYI’ boxes in textbooks.)

Having determined that there is a linear relationship between schooling and life expectancy, we used the python statsmodels package for fitting a linear regression model using ordinary least squares. Here is the summary of the linear regression model fitted. 

```python
model = smf.ols('target~schooling_predictor', data=df_2015).fit()
model.summary()
```

![Alt text](readme_imgs/olsSummary.png?raw=true "statsmodels OLS Summary")

The p value is 0.000, this shows that the schooling is a significant predictor of life expectancy. The coefficient indicates that for every one year increase in average no of school years, the life expectancy increases by 2.3387 years. 
