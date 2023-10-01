import finlab.ml as ml

dataset = ml.fundamental_features()
features = ['R103_ROE稅後', 'R402_營業毛利成長率']
dataset = dataset[features].dropna(how='any')
ml.add_profit_prediction(dataset)


def is_valid(feature, nstd):
    ub = feature.mean() + nstd * feature.std()
    lb = feature.mean() - nstd * feature.std()

    return (feature > lb) & (feature < ub)


valid = is_valid(dataset['R103_ROE稅後'], 2) & is_valid(dataset['R402_營業毛利成長率'], 0.05)
dataset_rmoutliers = dataset[valid].dropna()

import pandas as pd
import sklearn.preprocessing as preprocessing

dataset_scaled = pd.DataFrame(preprocessing.scale(dataset_rmoutliers), index=dataset_rmoutliers.index, columns=dataset_rmoutliers.columns)

from sklearn.model_selection import train_test_split
dataset_train, dataset_test = train_test_split(dataset_scaled, test_size=0.1, random_state=0)

from sklearn.svm import SVC
cf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto',
  kernel='linear', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)

cf.fit(dataset_train[features], dataset_train['return'] > dataset_train['return'].quantile(0.5))
score = cf.score(dataset_test[features], dataset_test['return'] > dataset_test['return'].quantile(0.5))
print(score)

history = dataset_test.copy()
history['svm_prediction'] = cf.predict(dataset_test[features])
history = history.reset_index()
dates = sorted(list(set(history['date'])))

seasonal_returns1 = []
seasonal_returns2 = []
for date in dates:
    print(date)
    current_stocks = history[history['date'] == date]
    buy_stock = current_stocks[current_stocks['svm_prediction'] == True]
    sell_stock = current_stocks[current_stocks['svm_prediction'] == False]
    seasonal_return1 = buy_stock['return'].mean()
    seasonal_return2 = sell_stock['return'].mean()

    seasonal_returns1.append(seasonal_return1)
    seasonal_returns2.append(seasonal_return2)

import matplotlib.pyplot as plt
plt.style.use("ggplot")

pd.Series(seasonal_returns1, index=dates).cumprod().plot(color='red')
pd.Series(seasonal_returns2, index=dates).cumprod().plot(color='blue')