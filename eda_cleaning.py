#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 13:42:18 2020

@author: michaelliu
"""

# Based off of Derek Jedamski's LinkedIn Learning Cource on Applied Machine Learning: Foundations
# Exploratory data analysis (EDA) & Cleaning

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Read in the data
titanic = pd.read_csv('train.csv')
titanic.head()

# First let's explore the continous features, to do that, we first simpplify the data by dropping the categorcial features
cat_feat = ['PassengerId', 'Name', 'Sex',  'Ticket', 'Cabin', 'Embarked']
titanic_cont_feat = titanic.drop(cat_feat, axis=1)
titanic_cont_feat.head()

# Now let's explore the countinous features

describe_cont = titanic_cont_feat.describe()
# Notice here that the "Age" feature is missing a significant amount of data

survive_cont = titanic_cont_feat.groupby('Survived').mean()
# It appears that indivudlaas that were in a higher class, therfore, had a higher fare, was more likely to survive

# Now let's use a histogram to plot the age and fare features 
for i in ['Age', 'Fare']:
    died = list(titanic_cont_feat[titanic_cont_feat['Survived'] == 0][i].dropna())
    survived = list(titanic_cont_feat[titanic_cont_feat['Survived'] == 1][i].dropna())
    xmin = min(min(died), min(survived))
    xmax = max(max(died), max(survived))
    width = (xmax - xmin) / 40
    sns.distplot(died, color='r', kde=False, bins=np.arange(xmin, xmax, width))
    sns.distplot(survived, color='g', kde=False, bins=np.arange(xmin, xmax, width))
    plt.legend(['Did not survive', 'Survived'])
    plt.title('Overlaid histogram for {}'.format(i))
    plt.show()
    
for i, col in enumerate(['Pclass', 'SibSp', 'Parch']):
    plt.figure(i)
    sns.catplot(x=col, y='Survived', data=titanic_cont_feat, kind='point', aspect=2)
    
# Now that we have explored the continous features, let's start cleaning them
# Start by filling in the "null" value of "Age" with the mean of the "Age"
titanic['Age'].fillna(titanic['Age'].mean(), inplace=True)
# Let's check if there are anymore bull values in the Age feature
null_sum = titanic.isnull().sum()

# Combine SibSp and Parch into one variable

# Now let's explore the categorical features
cont_feat = ['PassengerId', 'Pclass', 'Name', 'Age', 'SibSp', 'Parch', 'Fare']
titanic_cat_feat = titanic.drop(cont_feat, axis=1)

# Let's get some more information of the Sex, Cabin and Embarked features
