# (Simple) Linear Regression: A Practical Guide

Authors: Richard Aw, Lakshmi Manne, Shambhavi Gupta, Arman Hashemizadeh

> In fulfillment of the requirements for the Final Project of [Communications for Analytics (MSDS 610)](https://catalog.usfca.edu/preview_course_nopop.php?catoid=35&coid=533754). MSDS 610 is a core class under the MS in Data Science program at the University of San Francisco. 



## About This Guide

The goal of this guide is to provide a concise and “hands-on” introduction to linear regression, with an emphasis on **simple linear regression (SLR)**. It is meant to be accessible to those with *just enough* (bittersweet) memories of introductory probability, statistics, linear algebra, and Python programming. To this end, we have structured the guide as follows: 

1. **Linear Regression in a Nutshell** — “No-frills” overview of the essentials and motivation for linear regression.

2. **Linear Regression in Practice** — Gentle tutorial on implementing SLR in Python, using [the WHO Life Expectancy Dataset](https://www.kaggle.com/kumarajarshi/life-expectancy-who?select=Life+Expectancy+Data.csv). 



## Linear Regression in a Nutshell

### What *is* linear regression?

Linear regression is the process of *estimating* the coefficients in the equation for the straight line or plane that best fits some given data. Thus, to perform linear regression is to (i) find the numbers that specify the line/plane of best fit and (ii) determine how confident we can be in those numbers. 

> Remember from Physics 101 that whenever we are estimating the value of something like height, we don’t just state our guess but provide error bounds for it as well? This is because there is no guarantee that our guess *is* the true value. Linear regression is based on the same logic!

There are two types of linear regression: simple linear regression (SLR) and multiple linear regression (MLR). 

* **SLR** is performed when we regress the response variable (*Y*) against only *one* predictor variable (*X*) — this is why it is dubbed “simple”. Example: regressing “price of milk” against “milk’s shelf life”.

* **MLR** is performed when there are *multiple* predictor variables (two or more *X*’s). Example: regressing “price of milk” against “milk’s shelf life”, “milk’s tastiness score”, and “milk’s nutrition score”.

An intuitive way to grasp the difference between SLR and MLR is to visualize them both, side by side:

![SLR vs MLR](readme_imgs/SlrVsMlr.png?raw=true "SLR Vs MLR")
*Note that the 3D plot on the right depicts MLR in the case of two predictor variables; MLR also includes cases where there are even more predictor variables (X's)!* (<a href="https://www.keboola.com/blog/linear-regression-machine-learning">Image Source</a>)

In this way, we can see that SLR is about fitting the best *line* through a plot of 2D data, while MLR is about fitting the best *plane* through a plot of 3D or higher-dimensional data. (More about what 'best' means, exactly, in the tutorial section.)



### Why *care* about linear regression?

Linear regression enables us to carry out inference and prediction.

> This makes linear regression **practically useful**: Inference and prediction are both critical to the commercial success of many companies, and increasingly so given that [the world economy is becoming more ‘data-driven’](https://www2.deloitte.com/mt/en/pages/technology/articles/mt-what-is-digital-economy.html). If you want to [stay relevant](https://hbr.org/2020/02/boost-your-teams-data-literacy), or just [earn a *healthy* salary](https://datasciencedegree.wisconsin.edu/data-science/data-scientist-salary/), you *should* care about linear regression!

**Inference** is about understanding the relationship between variables in the given data. Regressing a chosen response variable against one or more predictor variables would enable us to evaluate not only the significance of *each* of the predictor variables on the response variable (e.g., does “milk’s shelf life” have a significant influence on “price of milk”?), but also the *relative* importance of the predictor variables (e.g., which has a greater impact on “price of milk” — “milk’s shelf life” or “milk’s tastiness score”?).

**Prediction** is about computing *projected* values for the response variable based on new input values of the predictor variables. Since linear regression gives us an equation that relates the response variable (e.g., “price of milk”) to the predictor variables (e.g., “milk’s shelf life”, “milk’s tastiness score”, etc.), we can ‘feed’ new values for “milk’s shelf life” and “milk’s tastiness score” to the equation and obtain a projected value for “price of milk”. In this way, linear regression enables us to *predict* the future given new data!



## Linear Regression in Practice

### Getting started

- We will need to install some Python packages. You can do so by running the following line of code in the command-line:

  `pip install pandas, matplotlib, seaborn, statsmodels`

-  We will be using the WHO Life Expectancy Dataset, which can be downloaded [here on Kaggle](https://www.kaggle.com/kumarajarshi/life-expectancy-who?select=Life+Expectancy+Data.csv). 

  - If you're planning to run the same block of code below in a Jupyter/CoLab notebook, be sure to *save the dataset in the same folder as your notebook*. Otherwise, you will have to modify the file path in `df = pd.read_csv('Life Expectancy Data.csv')`.


### Extracting two variables from the dataset for SLR

The dataset contains many features, so we have lots of options for choosing which features to include as the response variable and predictor variable in our SLR model. For pedagogical reasons, we want to find two variables that are sufficiently correlated. To do so, we can plot a correlation matrix:

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

It appears that `Schooling` has a pretty strong correlation of 0.82 with `Life expectancy`. As such, let us use `Life expectancy` and `Schooling` as our response variable and predictor variable, respectively.


### Creating a scatter plot of the two variables

We can easily visualize the relationship between `Schooling` and `Life expectancy` by producing a scatter plot as follows:

```python
schooling_predictor = df_2015['Schooling']
target = df_2015['Life expectancy ']
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(schooling_predictor, target)
ax.set_xlabel("Schooling (Avg. number of years)", fontsize='x-large')
ax.set_ylabel("Life Expectancy (years)", fontsize='x-large')
ax.set_title("WHO Country Life Expectancy Dataset, 2015", fontsize='x-large', y=1.05);
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
```

![Scatter Plot](readme_imgs/scatter_plot.png?raw=true "Scatter Plot")


### Finding the line of best fit 

From the scatter plot, it seems that we could definitely fit a straight line through those points. How, then, can we obtain the numbers that specify the direction and position of this line?

There are various methods to do so. Ordinary least squares estimation (OLSE) — wherein “best” means minimizing the sum of squared errors — is probably the most well-known method. Maximum likelihood estimation (MLE) — wherein “best” means maximizing the probability that each of the random error terms in our linear model is normally distributed — is another method. (Coincidentally, MLE and OLSE lead to the same results in SLR!) 

> OLSE should not be conflated with linear regression itself! The latter is a general process that may be initiated using a specific method, while the former is simply a specific method.


### Assessing the validity of the line of best fit

Having determined that there is a linear relationship between schooling and life expectancy, we used the python statsmodels package for fitting a linear regression model using ordinary least squares. Here is the summary of the linear regression model fitted. 

```python
model = smf.ols('target~schooling_predictor', data=df_2015).fit()
model.summary()
```

![OLS Summary](readme_imgs/olsSummary.png?raw=true "statsmodels OLS Summary")

The p value is 0.000, this shows that the schooling is a significant predictor of life expectancy. The coefficient indicates that for every one year increase in average no of school years, the life expectancy increases by 2.3387 years.


### Assessing the assumptions of the SLR model

1. The mean of error terms must be 0. 

For a perfect model, we would expect all the error terms to be 0, that is there is no difference between the actual value and the fitted line value. However, this is rarely seen in practice. Alternatively, we can say that the expected value of the error terms must be 0, that is the mean of error terms must be 0. 

How do we know if the assumption is met? 
Once we model the data, we can extract the residuals from the fitted model.

```python
residuals = model.resid
predicted_values = model.predict()
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(predicted_values, residuals)
ax.axhline(np.mean(residuals), color='red')
ax.set_xlabel('Predicted values')
ax.set_ylabel('Residuals')
ax.set_title('Residuals v/s Predicted values')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
```

![Residual Plot](readme_imgs/residuals.png?raw=true "Residual plot with matplotlib")

We see that the residuals are scattered evenly around the bold red line – the calculated mean of all residuals. The mean is exactly at 0 which means that our assumption is met. 
 
2. The variance of error terms must be constant. 
 
The second model is assumption is the variance of error terms must be a fixed value. This assumption is particularly important because we perform numerous statistical tests to determine the significance/worth of the model. Most of these significance tests rely on the assumption of constant variance. If this is not met, we cannot call our test results/model reliable. 

How do we know if the assumption is met? 
The same residuals v/s fitted value plot is good enough to verify this assumption. About the mean at 0, we see that the spread of the data points is almost equal. There is no drastic change in the spread as we move along the x-axis. The spread is capped within a range –15 to 15. 

3. The error terms must be independent and identically distributed (I.I.D)

The error terms from the model are random in nature. At any point, we cannot predict the future error term value. We can say that they are stochastic in nature. All the error terms must have the same probability of turning up as an error from the model. That is, they must all have the same probability distribution. Also, the error terms must be independent, that is the occurrence of one must not affect the occurrence of another error.   
How do we know if the assumption is met? 
While there are numerous complex ways to test if the errors are dependent on each other or if they are equiprobable. We will use our residuals v/s fitted values plot to draw a conclusion. We can clearly observe that, the residuals are randomly scattered around the mean and there is no clear pattern the residuals follow. Therefore, our assumption, that error terms are I.I.D is met. 

4. The error terms should be normally distributed. 

A QQ plot compares a given dataset distribution to the distribution of a standard normal. It allows us to compare how close/deviated our distribution is compared to an ideally distribution.  A plot that is linear and set at a 45-degree angle tells us that our data distribution is normal. 
Below, is the plot we obtained from the dataset, it is almost linear with slight deviations at the ends, which means that our error term distribution is close to a normal distribution, however, it could have heavy tails. 

When performing linear regression, we are not determining the true coefficients (or intercept value). We are merely estimating them, and this is already assuming that our conception of the ground truth (i.e., the SLR model Y = B_0 + B_1*X + E) is correct. The true relationship between variables might not be perfectly linear — we never know!
