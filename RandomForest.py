import pandas as pd

import finlab.ml as ml

dataset = ml.fundamental_features()
dataset = dataset.dropna(thresh=int(len(dataset) * 0.5), axis=1).dropna(how='any')
features = dataset.columns

ml.add_profit_prediction(dataset)
dataset = dataset.dropna()

date_arr = dataset.index.get_level_values('date') < '2017'
dataset_train = dataset[date_arr]
dataset_test = dataset[~date_arr]

from sklearn.ensemble import RandomForestClassifier
from finlab.data import Data
# returns1 = dataset_test['return'][prediction == True]
# dates = returns1.index.get_level_values('date')
# return1 = returns1.groupby(dates).mean().cumprod()
# print(return1)
#
# returns2 = dataset_test['return'][prediction == False]
# dates = returns2.index.get_level_values('date')
# returns2 = returns2.groupby(dates).mean().cumprod()
# print(returns2)

cf = RandomForestClassifier()
cf.fit(dataset_train[features], dataset_train['return'] > 1)
score = cf.score(dataset_test[features], dataset_test['return'] > 1)
print(score)

prediction = cf.predict(dataset_test[features])

importance = pd.Series(cf.feature_importances_, index=features).sort_values(ascending=False)
items = list(importance.head(10).index)

data = Data()
close = data.get('收盤價')
sma = close.rolling(60, min_periods=10).mean()
bias = close / sma
ml.add_feature(dataset, 'bias', bias)


def select_stock(df):
    rank = df[items].rank(pct=True).sum(axis=1)
    condition1 = rank.rank(pct=True) > 0.9
    condition2 = df['bias'] > 1
    return df[condition1 & condition2]['return'].mean()


dates = dataset.index.get_level_values('date')
print(dataset.groupby(dates).apply(select_stock).cumprod())
print(dataset['return'].groupby(dates).mean().cumprod())
