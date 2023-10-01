import finlab.ml as ml
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

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

fit_params = {"early_stopping_rounds": 30,
              "eval_metric": 'auc',
              "eval_set": [test],
              'eval_names': ['valid'],
              'verbose': 100,
              'categorical_feature': 'auto'}

param_test = {'num_leaves': sp_randint(6, 50),
              'min_child_samples': sp_randint(100, 500),
              'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
              'subsample': sp_uniform(loc=0.2, scale=0.8),
              'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
              'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
              'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

n_HP_points_to_test = 100

clf = lgb.LGBMClassifier(max_depth=-1, random_state=314, silent=True, metric='None', n_jobs=4, n_estimators=5000)
gs = RandomizedSearchCV(
    estimator=clf, param_distributions=param_test,
    n_iter=n_HP_points_to_test,
    scoring='roc_auc',
    cv=3,
    refit=True,
    random_state=314,
    verbose=True)

gs.fit(*train, **fit_params)
print(gs.best_estimator_)
colsample_bytree=0.9233783583096781, metric='None',
               min_child_samples=399, min_child_weight=0.1, n_estimators=5000,
               n_jobs=4, num_leaves=13, random_state=314, reg_alpha=2,
               reg_lambda=5, subsample=0.8548443912440804

colsample_bytree=0.8708421484750899, metric='None',
               min_child_samples=475, min_child_weight=1e-05, n_estimators=5000,
               n_jobs=4, num_leaves=22, random_state=314, reg_alpha=0.1,
               reg_lambda=1, subsample=0.5872540295820521