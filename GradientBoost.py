import pandas as pd

import finlab.ml as ml

dataset = ml.fundamental_features()
dataset = dataset.dropna(thresh=int(len(dataset) * 0.5), axis=1).dropna(how='any')
features = list(dataset.columns)

ml.add_profit_prediction(dataset)
dataset = dataset.dropna()

date_arr = dataset.index.get_level_values('date') < '2017'
dataset_train = dataset[date_arr]
dataset_test = dataset[~date_arr]

train = dataset_train[features], dataset_train['return'] > 1
test = dataset_test[features], dataset_test['return'] > 1

import lightgbm
fit_params={"early_stopping_rounds":30,
            "eval_metric": 'auc',
            "eval_set" : [test],
            'eval_names': ['valid'],
            'verbose': 100,
            'categorical_feature': 'auto'}

# 裏頭的參數為參數優化後之結果
cf_lgbm = lightgbm.LGBMClassifier(colsample_bytree=0.8708421484750899, metric='None',
               min_child_samples=475, min_child_weight=1e-05, n_estimators=5000,
               n_jobs=4, num_leaves=22, random_state=314, reg_alpha=0.1,
               reg_lambda=1, subsample=0.5872540295820521)

cf_lgbm.fit(dataset_train[features], dataset_train['return'] > 1, **fit_params)
cf_lgbm.score(dataset_test[features], dataset_test['return'] > 1)

prediction = cf_lgbm.predict(dataset_test[features])

returns1 = dataset_test['return'][prediction == True]
dates = returns1.index.get_level_values('date')
print(returns1.groupby(dates).mean().cumprod())

returns2 = dataset_test['return'][prediction == False]
dates = returns2.index.get_level_values('date')
print(returns2.groupby(dates).mean().cumprod())