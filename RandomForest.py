import pandas as pd

import finlab.ml as ml

# Load fundamental features dataset using finlab.ml
dataset = ml.fundamental_features()

# Remove columns with more than 50% NaN values and rows with any NaN values
dataset = dataset.dropna(thresh=int(len(dataset) * 0.5), axis=1).dropna(how='any')

# Get a list of feature columns
features = dataset.columns

# Add a profit prediction column to the dataset
ml.add_profit_prediction(dataset)

# Remove rows with NaN values
dataset = dataset.dropna()

# Create a date_arr based on a date condition
date_arr = dataset.index.get_level_values('date') < '2017'

# Split the dataset into training and testing sets based on the date condition
dataset_train = dataset[date_arr]
dataset_test = dataset[~date_arr]

# Import the RandomForestClassifier from sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier
from finlab.data import Data

# Define a subset of returns1 based on a prediction condition
returns1 = dataset_test['return'][prediction == True]

# Extract dates and calculate cumulative returns for returns1
dates = returns1.index.get_level_values('date')
return1 = returns1.groupby(dates).mean().cumprod()
print(return1)

# Define a subset of returns2 based on a prediction condition
returns2 = dataset_test['return'][prediction == False]

# Extract dates and calculate cumulative returns for returns2
dates = returns2.index.get_level_values('date')
returns2 = returns2.groupby(dates).mean().cumprod()
print(returns2)

# Create a RandomForestClassifier
cf = RandomForestClassifier()

# Fit the classifier on the training dataset
cf.fit(dataset_train[features], dataset_train['return'] > 1)

# Calculate and print the score of the classifier on the testing dataset
score = cf.score(dataset_test[features], dataset_test['return'] > 1)
print(score)

# Make predictions using the trained classifier
prediction = cf.predict(dataset_test[features])

# Calculate feature importance using the RandomForestClassifier
importance = pd.Series(cf.feature_importances_, index=features).sort_values(ascending=False)
items = list(importance.head(10).index)

# Initialize a Data object
data = Data()

# Get the '收盤價' (closing price) data
close = data.get('收盤價')

# Calculate the Simple Moving Average (SMA) with a rolling window of 60 and minimum periods of 10
sma = close.rolling(60, min_periods=10).mean()

# Calculate the BIAS indicator
bias = close / sma

# Add the 'bias' feature to the dataset
ml.add_feature(dataset, 'bias', bias)

# Define a function to select stocks based on specific conditions
def select_stock(df):
    rank = df[items].rank(pct=True).sum(axis=1)
    condition1 = rank.rank(pct=True) > 0.9
    condition2 = df['bias'] > 1
    return df[condition1 & condition2]['return'].mean()

# Extract dates and calculate cumulative returns based on the selected stocks
dates = dataset.index.get_level_values('date')
print(dataset.groupby(dates).apply(select_stock).cumprod())

# Calculate and print the cumulative returns of 'return' column grouped by dates
print(dataset['return'].groupby(dates).mean().cumprod())
