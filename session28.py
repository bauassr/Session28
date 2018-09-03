# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 23:27:04 2018

"""



# Load libraries

# Core Libraries - Data manipulation and analysis
import pandas as pd
import numpy as np
import math
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
  
# Core Libraries - Machine Learning
import sklearn
 

# Importing Regressor - Modelling
from sklearn.neighbors import KNeighborsRegressor
 

## Importing train_test_split,cross_val_score,KFold - Validation and Optimization
from sklearn.model_selection import  train_test_split, cross_val_score, KFold 


# Importing Metrics - Performance Evaluation
from sklearn import metrics

# Warnings Library - Ignore warnings
import warnings
warnings.filterwarnings('ignore')

import pickle



pd.set_option("display.max_columns",100) 


#Load Data

nba = pd.read_csv('nba_2013.csv')


# Understand the Dataset and Data


nba.shape



columns = nba.columns.values
columns



nba.head()



nba.tail()



nba.info()


#There are NULL values in the dataset which need to be imputed or removed. Also, there are NULL values in numerical columns but none in the categorical columns


nba.get_dtype_counts()


#Clean the data


# Store the numerical and categorical columns list
num_cols = nba.select_dtypes(exclude = 'object').columns.values
cat_cols = nba.select_dtypes(include = 'object').columns.values
num_cols, cat_cols


# Clean Column Names

#There are no column names that need cleaning

#Clean Numerical Columns

nba[num_cols].info()


#It is clear that column 'fg.', 'x3p.', 'x2p.', 'efg.', 'ft.' have null values



nba[num_cols].nunique()


#season_end column has only one unique values throughout the rows the dataset. In other words, its a constant through out the dataset. So, we will choose to ignore the column in our regression.

# Null values

nba_null = pd.DataFrame({'total_null_values': nba[num_cols].isna().sum(), 'null_percentage': (nba[num_cols].isna().sum()/nba.shape[0])*100})
nba_null 

null_col_list = nba_null.loc[nba_null.total_null_values>0,:].index.values


null_col_list


(nba.loc[(nba.isna()).any(axis=1),:].shape[0]/nba.shape[0])*100


#Since one of the rows has higher percentage of NaN/Null values and also because this is a small dataset, it is better to impute.


nba.loc[(nba.isna()).any(axis=1),:].head()


#We remove columns ending with period(['fg.', 'x3p.', 'x2p.', 'efg.', 'ft.']) because, those columns are calculated with the help of the 2 preceeding them and don't add any new information to the existing data and also contain null values which cannot be imputed MEANINGFULLY. Moreover, they might cause feature interactions in the data when we perform regression.


#nba.interpolate(value=np.NaN, method='nearest', axis=0, inplace=True)
nba.drop(columns = null_col_list,inplace =True) 



num_cols = num_cols = nba.select_dtypes(exclude = 'object').columns.values



nba.head()



nba.loc[(nba.isna()).any(axis=1),:].shape

nba.loc[(nba==0).all(axis=1),:].shape


# No rows with all columns values == 0

nba.loc[(nba==0).any(axis=1),:].shape


nba.loc[(nba==0).any(axis=1),:].head()


#The zeroes in the dataset seem to be valid zeroes. So, no cleaning is required

#Nonsensical values


# Checking for negative numbers
nba.loc[(nba[num_cols].values<0).any(axis=1),:].shape


#There don't seem to be any nonsensical values in the numerical columns of the dataset

#Clean Categorical Columns


nba[cat_cols].info()


#There are no null values in the categorical columns


nba[cat_cols].nunique()


#season column has only one unique values throughout the rows the dataset. In other words, its a constant through out the dataset. So, we will choose to ignore the column in our regression.

#Since, there are no null values in the categorical columns, no need to check for null values again

#Empty Strings


nba.loc[(nba=="").any(axis=1),:].shape


#There are no empty strings in the categorical columns

#Nonsensical values 


nba['player'].unique()



nba['player'].nunique()



nba['bref_team_id'].unique()


nba['bref_team_id'].nunique()


plt.figure(figsize = (20,5))
sns.countplot(x = nba['bref_team_id'], data = nba)


#It is interesting that there many players from the Team with Team ID == TOT. This might introduce a bias in training of the regression. Therefore, it needs checking. When checking the number of teams in NBA from http://stats.nba.com/teams/ under Team List Heading, and doing a simple count of the number of teams revealed there are only 30 teams. TOT is used when a player has represented more than one team in a season.And the values associated with that player's row are the combined stats(Refer: https://www.reddit.com/r/nba/comments/7lt7qz/what_does_tot_mean_on_basketballreferencecom/)

#The value TOT is a valid value in this dataset


nba['pos'].unique()



nba['pos'].nunique()



plt.figure(figsize = (20,5))
sns.countplot(x= nba['pos'], data = nba)


#There are very less number of players having positions == G, F. After reading about the player positions in basketball from https://www.myactivesg.com/sports/basketball/how-to-play/basketball-rules/basketball-positions-and-roles, I have identified that the G,F are invalid values 


# Identify the players for whom these invalid values G and F
nba.loc[(nba['pos'].isin(['G','F'])),['player','pos']]


# Looking up for the position values for the players above in 'https://www.basketball-reference.com/leagues/NBA_2014_totals.html' which contains the above dataset(but slightly corrected), it can be seen that Damion James's Position was 'SF' and Josh Powell's Position was 'PF'.


# Replace the invalid values with the ones mentioned above
nba['pos'].replace(to_replace ='G',value= 'SF',inplace=True)
nba['pos'].replace(to_replace ='F',value= 'PF',inplace=True)



# Check the players for whom these invalid values G and F
nba.loc[(nba['pos'].isin(['G','F'])),['player','pos']]


# Identify the players for whom these invalid values G and F
nba.loc[[224,356],['player','pos']]


#The replacement of the values was successful and the column is now clean


nba['season'].unique()



nba['season'].nunique()



plt.figure(figsize = (20,5))
sns.countplot(x= nba['season'], data = nba)


#The season column has only one value and rightly so, as the data pertains to the season 2013-2014

# Get Basic Statistical Information



print(nba.describe())


nba.describe(include ='object')


plt.figure(figsize=(20,20))
sns.heatmap(nba.corr(), annot= True)


#Explore Data

# Uni-variate

#Uni-variate - Numerical columns


len(nba[num_cols].columns)



nba[num_cols].hist(bins=50,figsize=(20,40), layout= (9,3))


# It is interesting the note the majority of the columns have values which behave similar to exponential distribution

# Also, the season_end columns has only one value repeated for every row

# Uni-variate - Categorical Columns


for i,col in enumerate(nba[cat_cols]):
    plt.figure(i,figsize = (20,5))
    sns.countplot(x=col, data=nba[cat_cols])
    print()


#It is expected that every name has the same count == 1, as no player name was repeated. Also, season column value for the year 2013-2014 is expected to be a constant for all values in that column. After all, it is a dataset for the year 2013-2014

# However, it is interesting that there many players from the Team with Team ID == TOT. These players are players who played for more than one team in a season because of team transfers. It shows that there were 63 transfers in 2013 assuming every player transfered only one. The data however can't convey if a player had more than one team transfer in 2013.

# Bi-variate

#Individual Numerical Columns vs pts


for i, col in enumerate(num_cols):
    plt.figure(i,figsize = (20,5))
    sns.regplot(x = col, y = 'pts', data = nba)


#Individual Categorical Columns vs pts


cat_cols



for i, col in enumerate(cat_cols):
    plt.figure(i,figsize = (20,5))
    sns.barplot(x = col,y ='pts', data=nba)


#Multi-variate


plt.figure(figsize=(30,20))
sns.heatmap(nba.corr(), annot = True, cmap= "PRGn")


nba.corr().iloc[-2:-1,:]


print(nba.corr().iloc[-2:-1,:])


#Engineer Features


print(num_cols)
print(cat_cols)


# Encode Categorical Columns


to_encode = ['pos','bref_team_id']
prefixes = ['pos','team']
nba_encoded= pd.get_dummies(data = nba, prefix = prefixes, columns = to_encode, prefix_sep = '_', drop_first = True)



nba_encoded.columns


nba_encoded.drop(['player','season','season_end'], axis = 1,inplace = True)



nba_encoded_cols = nba_encoded.columns.values
nba_encoded_cols


#Data Preprocessing - Normalization(MinMaxScaler)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
nba_scaled_array = scaler.fit_transform(nba_encoded)


nba_scaled = pd.DataFrame(data = nba_scaled_array, columns = nba_encoded_cols)


nba_scaled.describe()


#Generate Input Vector X and Output Y, and Split the Data for Training and Testing


X = nba_scaled.drop('pts', axis = 1)
Y = nba_scaled['pts']



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)


#Fit the Base Models and Collect the Metrics

# Distance Metric = Euclidean Distance

k_values =[]
r2_train_values =[]
r2_test_values =[]
rmse_train_values = []
rmse_test_values =[]
accuracy_test =[]
accuracy_train =[]

import math

for k in range(1,51):
    knn = KNeighborsRegressor(n_neighbors = k, weights='uniform', algorithm='auto')
    model = knn.fit(x_train, y_train) 
    y_pred = model.predict(x_test)
    k_values.append(k)
    r2_train_values.append(metrics.r2_score(model.predict(x_train), y_train))
    r2_test_values.append(metrics.r2_score(model.predict(x_test), y_test))
    rmse_train_values.append(math.sqrt(metrics.mean_squared_error(model.predict(x_train), y_train)))
    rmse_test_values.append(math.sqrt(metrics.mean_squared_error(model.predict(x_test), y_test)))
    accuracy_train.append(model.score(x_train, y_train))
    accuracy_test.append(model.score(x_test, y_test))

    print("The RMSE is ", math.sqrt(metrics.mean_squared_error(model.predict(x_test), y_test)),"for K-Value:",k)



plt.figure(figsize = (20,10))
plt.title('Accuracy, R2-Score and RMSE for Train and Test sets by K-Value - EUCLIDEAN DISTANCE' )
plt.xlabel('K-Value')
plt.ylabel('Score Value')

plt.scatter(k_values, accuracy_train, label = 'accuracy_train')
plt.scatter(k_values, accuracy_test, label = 'accuracy_test')

plt.scatter(k_values, r2_train_values, label = 'r2_train')
plt.scatter(k_values, r2_test_values, label = 'r2_test')

plt.scatter(k_values, rmse_train_values, label = 'rmse_train')
plt.scatter(k_values, rmse_test_values, label = 'rmse_test')

plt.legend()
plt.show()
     


# Distance Metric = Manhattan Distance  

k_values =[]
r2_train_values =[]
r2_test_values =[]
rmse_train_values = []
rmse_test_values =[]
accuracy_test =[]
accuracy_train =[]

import math

for k in range(1,51):
    knn = KNeighborsRegressor(n_neighbors = k, weights='uniform', algorithm='auto',p=1)
    model = knn.fit(x_train, y_train) 
    y_pred = model.predict(x_test)
    k_values.append(k)
    r2_train_values.append(metrics.r2_score(model.predict(x_train), y_train))
    r2_test_values.append(metrics.r2_score(model.predict(x_test), y_test))
    rmse_train_values.append(math.sqrt(metrics.mean_squared_error(model.predict(x_train), y_train)))
    rmse_test_values.append(math.sqrt(metrics.mean_squared_error(model.predict(x_test), y_test)))
    accuracy_train.append(model.score(x_train, y_train))
    accuracy_test.append(model.score(x_test, y_test))
    
    
plt.figure(figsize = (20,10))
plt.title('Accuracy, R2-Score and RMSE for Train and Test sets by K-Value - MANHATTAN DISTANCE' )
plt.xlabel('K-Value')
plt.ylabel('Score Value')

plt.scatter(k_values, accuracy_train, label = 'accuracy_train')
plt.scatter(k_values, accuracy_test, label = 'accuracy_test')

plt.scatter(k_values, r2_train_values, label = 'r2_train')
plt.scatter(k_values, r2_test_values, label = 'r2_test')

plt.scatter(k_values, rmse_train_values, label = 'rmse_train')
plt.scatter(k_values, rmse_test_values, label = 'rmse_test')

plt.legend()
plt.show()
     


#Distance Metric = Minkowski Distance

k_values =[]
r2_train_values =[]
r2_test_values =[]
rmse_train_values = []
rmse_test_values =[]
accuracy_test =[]
accuracy_train =[]

import math

for k in range(1,51):
    knn = KNeighborsRegressor(n_neighbors = k, weights='uniform', algorithm='auto', p = 3)
    model = knn.fit(x_train, y_train) 
    y_pred = model.predict(x_test)
    k_values.append(k)
    r2_train_values.append(metrics.r2_score(model.predict(x_train), y_train))
    r2_test_values.append(metrics.r2_score(model.predict(x_test), y_test))
    rmse_train_values.append(math.sqrt(metrics.mean_squared_error(model.predict(x_train), y_train)))
    rmse_test_values.append(math.sqrt(metrics.mean_squared_error(model.predict(x_test), y_test)))
    accuracy_train.append(model.score(x_train, y_train))
    accuracy_test.append(model.score(x_test, y_test))

    
plt.figure(figsize = (20,10))

plt.title('Accuracy, R2-Score and RMSE for Train and Test sets by K-Value - MINKOWSKI DISTANCE(p>2)' )
plt.xlabel('K-Value')
plt.ylabel('Score Value')

plt.scatter(k_values, accuracy_train, label = 'accuracy_train')
plt.scatter(k_values, accuracy_test, label = 'accuracy_test')

plt.scatter(k_values, r2_train_values, label = 'r2_train')
plt.scatter(k_values, r2_test_values, label = 'r2_test')

plt.scatter(k_values, rmse_train_values, label = 'rmse_train')
plt.scatter(k_values, rmse_test_values, label = 'rmse_test')

plt.legend()
plt.show()


#Select Features

from sklearn.ensemble import RandomForestRegressor
rndf = RandomForestRegressor(n_estimators=150)
rndf.fit(x_train, y_train)
importance = pd.DataFrame.from_dict({'cols':x_train.columns, 'importance': rndf.feature_importances_})
importance = importance.sort_values(by='importance', ascending=False)
plt.figure(figsize=(20,15))
sns.barplot(importance.cols, importance.importance)
plt.xticks(rotation=90)


imp_cols = importance[importance.importance > 0.001].cols.values
imp_cols


x1_train,x1_test, y1_train, y1_test = train_test_split(X[imp_cols],Y,test_size=0.3,random_state =100)


#Fit Features Selected Model and Collect the Metrics

#Distance Metric = Euclidean Distance


k_values =[]
r2_train1_values =[]
r2_test1_values =[]
rmse_train1_values = []
rmse_test1_values =[]
accuracy_test1 =[]
accuracy_train1 =[]


import math

for k in range(1,51):
    knn = KNeighborsRegressor(n_neighbors = k, weights='uniform', algorithm='auto')
    model = knn.fit(x1_train, y1_train) 
    y_pred1 = model.predict(x1_test)
    k_values.append(k)
    r2_train1_values.append(metrics.r2_score(model.predict(x1_train), y1_train))
    r2_test1_values.append(metrics.r2_score(model.predict(x1_test), y1_test))
    rmse_train1_values.append(math.sqrt(metrics.mean_squared_error(model.predict(x1_train), y1_train)))
    rmse_test1_values.append(math.sqrt(metrics.mean_squared_error(model.predict(x1_test), y1_test)))
    accuracy_train1.append(model.score(x1_train, y1_train))
    accuracy_test1.append(model.score(x1_test, y1_test))

    print("The RMSE is ", math.sqrt(metrics.mean_squared_error(model.predict(x1_test), y1_test)),"for K-Value:",k)


plt.figure(figsize = (20,10))
plt.title('Accuracy, R2-Score and RMSE for Feature Selected Train and Test sets by K-Value - EUCLIDEAN DISTANCE' )
plt.xlabel('K-Value')
plt.ylabel('Score Value')

plt.scatter(k_values, accuracy_train1, label = 'accuracy_train  - Feature Selected')
plt.scatter(k_values, accuracy_test1, label = 'accuracy_test - Feature Selected')

plt.scatter(k_values, r2_train1_values, label = 'r2_train - Feature Selected')
plt.scatter(k_values, r2_test1_values, label = 'r2_test - Feature Selected')

plt.scatter(k_values, rmse_train1_values, label = 'rmse_train - Feature Selected')
plt.scatter(k_values, rmse_test1_values, label = 'rmse_test - Feature Selected')

plt.legend()
plt.show()


#Distance metric = Manhattan Distance


k_values =[]
r2_train1_values =[]
r2_test1_values =[]
rmse_train1_values = []
rmse_test1_values =[]
accuracy_test1 =[]
accuracy_train1 =[]


import math

for k in range(1,51):
    knn = KNeighborsRegressor(n_neighbors = k, weights='uniform', algorithm='auto', p =1)
    model = knn.fit(x1_train, y1_train) 
    y_pred1 = model.predict(x1_test)
    k_values.append(k)
    r2_train1_values.append(metrics.r2_score(model.predict(x1_train), y1_train))
    r2_test1_values.append(metrics.r2_score(model.predict(x1_test), y1_test))
    rmse_train1_values.append(math.sqrt(metrics.mean_squared_error(model.predict(x1_train), y1_train)))
    rmse_test1_values.append(math.sqrt(metrics.mean_squared_error(model.predict(x1_test), y1_test)))
    accuracy_train1.append(model.score(x1_train, y1_train))
    accuracy_test1.append(model.score(x1_test, y1_test))

plt.figure(figsize = (20,10))
plt.title('Accuracy, R2-Score and RMSE for Feature Selected Train and Test sets by K-Value - MANHATTAN DISTANCE' )
plt.xlabel('K-Value')
plt.ylabel('Score Value')


plt.scatter(k_values, accuracy_train1, label = 'accuracy_train  - Feature Selected')
plt.scatter(k_values, accuracy_test1, label = 'accuracy_test - Feature Selected')

plt.scatter(k_values, r2_train1_values, label = 'r2_train - Feature Selected')
plt.scatter(k_values, r2_test1_values, label = 'r2_test - Feature Selected')

plt.scatter(k_values, rmse_train1_values, label = 'rmse_train - Feature Selected')
plt.scatter(k_values, rmse_test1_values, label = 'rmse_test - Feature Selected')

plt.legend()
plt.show()


#Distance metric = Minkowski Distance


k_values =[]
r2_train1_values =[]
r2_test1_values =[]
rmse_train1_values = []
rmse_test1_values =[]
accuracy_test1 =[]
accuracy_train1 =[]


import math

for k in range(1,51):
    knn = KNeighborsRegressor(n_neighbors = k, weights='uniform', algorithm='auto', p =3)
    model = knn.fit(x1_train, y1_train) 
    y_pred1 = model.predict(x1_test)
    k_values.append(k)
    r2_train1_values.append(metrics.r2_score(model.predict(x1_train), y1_train))
    r2_test1_values.append(metrics.r2_score(model.predict(x1_test), y1_test))
    rmse_train1_values.append(math.sqrt(metrics.mean_squared_error(model.predict(x1_train), y1_train)))
    rmse_test1_values.append(math.sqrt(metrics.mean_squared_error(model.predict(x1_test), y1_test)))
    accuracy_train1.append(model.score(x1_train, y1_train))
    accuracy_test1.append(model.score(x1_test, y1_test))

plt.figure(figsize = (20,10))
plt.title('Accuracy, R2-Score and RMSE for Feature Selected Train and Test sets by K-Value - MINKOWSKI DISTANCE' )
plt.xlabel('K-Value')
plt.ylabel('Score Value')


plt.scatter(k_values, accuracy_train1, label = 'accuracy_train  - Feature Selected')
plt.scatter(k_values, accuracy_test1, label = 'accuracy_test - Feature Selected')

plt.scatter(k_values, r2_train1_values, label = 'r2_train - Feature Selected')
plt.scatter(k_values, r2_test1_values, label = 'r2_test - Feature Selected')

plt.scatter(k_values, rmse_train1_values, label = 'rmse_train - Feature Selected')
plt.scatter(k_values, rmse_test1_values, label = 'rmse_test - Feature Selected')

plt.legend()
plt.show()


#Validate Model

# Validate Base Model

cv_scores_euclid =[]
cv_scores_manhattan =[]
cv_scores_minkowski =[]

k_values =[]

for k in range(1, 51):
    
    k_values.append(k)
    
    knn_euclid = KNeighborsRegressor(n_neighbors = k, weights='uniform', algorithm='auto')
    scores = cross_val_score(knn_euclid, x_train, y_train, cv=10, scoring='neg_mean_squared_error')
    cv_scores_euclid.append(scores.mean())
    
    knn_manhattan = KNeighborsRegressor(n_neighbors = k, weights='uniform', algorithm='auto', p = 1 )
    scores = cross_val_score(knn_manhattan, x_train, y_train, cv=10, scoring='neg_mean_squared_error')
    cv_scores_manhattan.append(scores.mean())
    
    knn_minkowski = KNeighborsRegressor(n_neighbors = k, weights='uniform', algorithm='auto', p = 3)
    scores = cross_val_score(knn_minkowski, x_train, y_train, cv=10, scoring='neg_mean_squared_error')
    cv_scores_minkowski.append(scores.mean())
 


# Validate Feature Selected Model
cv_scores1_euclid =[]
cv_scores1_manhattan =[]
cv_scores1_minkowski =[]

for k in range(1, 51):
         
    knn1_euclid = KNeighborsRegressor(n_neighbors = k, weights='uniform', algorithm='auto')
    scores = cross_val_score(knn1_euclid, x1_train , y1_train, cv=10, scoring='neg_mean_squared_error')
    cv_scores1_euclid.append(scores.mean())
    
    knn1_manhattan = KNeighborsRegressor(n_neighbors = k, weights='uniform', algorithm='auto', p = 1 )
    scores = cross_val_score(knn1_manhattan, x1_train , y1_train, cv=10, scoring='neg_mean_squared_error')
    cv_scores1_manhattan.append(scores.mean())
    
    knn1_minkowski = KNeighborsRegressor(n_neighbors = k, weights='uniform', algorithm='auto', p = 3)
    scores = cross_val_score(knn1_minkowski, x1_train , y1_train, cv=10, scoring='neg_mean_squared_error')
    cv_scores1_minkowski.append(scores.mean())
 


plt.figure(figsize = (20,10))
plt.title('CV Scores by K-Value for Feature Selected and Base Models' )
plt.xlabel('K-Value')
plt.ylabel('CV Scores - Mean')
plt.scatter(k_values,cv_scores1_euclid, label = 'Euclid - Feature Selected')
plt.scatter(k_values,cv_scores1_manhattan, label = 'Manhattan - Feature Selected')
plt.scatter(k_values,cv_scores1_minkowski, label = 'Minkowski - Feature Selected')

plt.scatter(k_values,cv_scores_euclid, label = 'Euclid')
plt.scatter(k_values,cv_scores_manhattan, label = 'Manhattan')
plt.scatter(k_values,cv_scores_minkowski, label = 'Minkowski')

plt.legend()
plt.show()


#The CV score for different distance measures are better for feature selected models and are very close to zero

#Compare Performance Metrics of Different Models  - Euclidean Distance based Models 


# Base Model

k_values =[]
r2_train_values =[]
r2_test_values =[]
rmse_train_values = []
rmse_test_values =[]
accuracy_test =[]
accuracy_train =[]

import math

for k in range(1,51):
    knn = KNeighborsRegressor(n_neighbors = k, weights='uniform', algorithm='auto')
    model = knn.fit(x_train, y_train) 
    y_pred = model.predict(x_test)
    k_values.append(k)
    r2_train_values.append(metrics.r2_score(model.predict(x_train), y_train))
    r2_test_values.append(metrics.r2_score(model.predict(x_test), y_test))
    rmse_train_values.append(math.sqrt(metrics.mean_squared_error(model.predict(x_train), y_train)))
    rmse_test_values.append(math.sqrt(metrics.mean_squared_error(model.predict(x_test), y_test)))
    accuracy_train.append(model.score(x_train, y_train))
    accuracy_test.append(model.score(x_test, y_test))



# Features Selected Model

k_values =[]
r2_train1_values =[]
r2_test1_values =[]
rmse_train1_values = []
rmse_test1_values =[]
accuracy_test1 =[]
accuracy_train1 =[]


import math

for k in range(1,51):
    knn = KNeighborsRegressor(n_neighbors = k, weights='uniform', algorithm='auto')
    model = knn.fit(x1_train, y1_train) 
    y_pred1 = model.predict(x1_test)
    k_values.append(k)
    r2_train1_values.append(metrics.r2_score(model.predict(x1_train), y1_train))
    r2_test1_values.append(metrics.r2_score(model.predict(x1_test), y1_test))
    rmse_train1_values.append(math.sqrt(metrics.mean_squared_error(model.predict(x1_train), y1_train)))
    rmse_test1_values.append(math.sqrt(metrics.mean_squared_error(model.predict(x1_test), y1_test)))
    accuracy_train1.append(model.score(x1_train, y1_train))
    accuracy_test1.append(model.score(x1_test, y1_test))



plt.figure(figsize = (20,10))
plt.title('Accuracy(Test Set Only) for Feature Selected and Base Models by K-Value - EUCLIDEAN DISTANCE' )
plt.xlabel('K-Value')
plt.ylabel('Score Value')


plt.scatter(k_values, accuracy_test1, label = 'Accuracy  - Feature Selected')
plt.scatter(k_values, accuracy_test , label = 'Accuracy  - Base')

plt.legend()
plt.show()



plt.figure(figsize = (20,10))
plt.title('R2-Score(Test Set Only) for Feature Selected and Base Models by K-Value - EUCLIDEAN DISTANCE' )
plt.xlabel('K-Value')
plt.ylabel('Score Value')
 

plt.scatter(k_values, r2_test1_values,label = 'R2-Score - Feature Selected')
plt.scatter(k_values, r2_test_values, label = 'R2-Score - Base')

plt.legend()
plt.show()



plt.figure(figsize = (20,10))

plt.title('RMSE (Test Set Only) for Feature Selected and Base Models by K-Value - EUCLIDEAN DISTANCE' )
plt.xlabel('K-Value')
plt.ylabel('Error Value')
 
plt.scatter(k_values, rmse_test1_values, c ='blue', label = 'rmse_test  - Feature Selected')
plt.scatter(k_values, rmse_test_values,  c ='red', label = 'rmse_test  - Base')

plt.legend()
plt.show()


#Models with k = 4,5,6 provide us with better metrics. So we choose the models where k =4,5,6 and select one among them that has best metrics


knn_4 = KNeighborsRegressor(n_neighbors = 4, weights='uniform', algorithm='auto')
model_4 = knn_4.fit(x1_train, y1_train) 
print('K = 4')
print('-'*60)
print('Accuracy: ', model_4.score(x1_test, y1_test))
print('R2-score: ', metrics.r2_score(model_4.predict(x1_test), y1_test))
print('RMSE: ', math.sqrt(metrics.mean_squared_error(model_4.predict(x1_test), y1_test)))
print( )
print( )

knn_5 = KNeighborsRegressor(n_neighbors = 5, weights='uniform', algorithm='auto')
model_5 = knn_5.fit(x1_train, y1_train) 
print('K = 5')
print('-'*60)
print('Accuracy: ', model_5.score(x1_test, y1_test))
print('R2-score: ', metrics.r2_score(model_5.predict(x1_test), y1_test))
print('RMSE: ', math.sqrt(metrics.mean_squared_error(model_5.predict(x1_test), y1_test)))
print( )
print( )


knn_6 = KNeighborsRegressor(n_neighbors = 6, weights='uniform', algorithm='auto')
model_6 = knn_6.fit(x1_train, y1_train) 
print('K = 6')
print('-'*60)
print('Accuracy: ', model_6.score(x1_test, y1_test))
print('R2-score: ', metrics.r2_score(model_6.predict(x1_test), y1_test))
print('RMSE: ', math.sqrt(metrics.mean_squared_error(model_6.predict(x1_test), y1_test)))


#The feature selected model (for k = 6) performs better for better than the base model. So this should be the model we choose to deploy

#Choose the model for deployment

imp_cols


# Saving the the chosen model in the pickle object
chosen_model_pkl = pickle.dumps(knn_6)


#To Load:
chosen_model = pickle.loads(chosen_model_pkl)


#Predicting the Points

# Base Model for k = 6
knn = KNeighborsRegressor(n_neighbors = 6, weights='uniform', algorithm='auto')
model = knn.fit(x_train, y_train)


def de_scale(num):
    num = num*(nba.pts.max() - nba.pts.min()) + nba.pts.min()
    return num


Y_pred = pd.Series(model_6.predict(X[imp_cols])).apply(de_scale) # K = 6 Model with imp_cols features
Y_pred_base= pd.Series(model.predict(X)).apply(de_scale) # K = 6 Model with all features


pred_points_df = pd.DataFrame({'Player': nba.player ,
                               'Y': nba.pts ,
                               'Y_Pred': Y_pred,
                               'Y_Pred_Base': Y_pred_base,
                              })
pred_points_df


plt.figure(figsize =(20,300))
plt.ylim(-1,482)
plt.yticks(range(481))
print()
plt.title("Model(K=6): Scatter plot of Player Names Vs Actual Points, Points Predicted by Models with Selected and All Features")
plt.ylabel("Player Names")
plt.xlabel("Points")
plt.scatter(pred_points_df.Y , pred_points_df.Player, label = 'Actual Points')
plt.scatter(pred_points_df.Y_Pred , pred_points_df.Player, label= 'Predicted Points - Selected Features' )
plt.scatter(pred_points_df.Y_Pred_Base , pred_points_df.Player, c = 'lightgreen', label = 'Predicted Points - All Features' )
plt.gca().invert_yaxis()
plt.legend()
plt.show()

