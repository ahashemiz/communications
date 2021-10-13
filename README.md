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
