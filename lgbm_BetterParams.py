import finlab.ml as ml
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# Load fundamental features dataset
dataset = ml.fundamental_features()

# Remove columns with more than 50% NaN values and rows with any NaN values
dataset = dataset.dropna(thresh=int(len(dataset) * 0.5), axis=1).dropna(how='any')

# Get a list of feature columns
features = list(dataset.columns)

# Add a profit prediction column to the dataset
ml.add_profit_prediction(dataset)

# Remove rows with NaN values
dataset = dataset.dropna()

# Split the dataset into training and testing sets based on a date condition
date_arr = dataset.index.get_level_values('date') < '2017'
dataset_train = dataset[date_arr]
dataset_test = dataset[~date_arr]

# Define the training and testing data
train = dataset_train[features], dataset_train['return'] > 1
test = dataset_test[features], dataset_test['return'] > 1

# Define fit parameters for the LightGBM model
fit_params = {"early_stopping_rounds": 30,
              "eval_metric": 'auc',
              "eval_set": [test],
              'eval_names': ['valid'],
              'verbose': 100,
              'categorical_feature': 'auto'}

# Define hyperparameter search space for RandomizedSearchCV
param_test = {'num_leaves': sp_randint(6, 50),
              'min_child_samples': sp_randint(100, 500),
              'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
              'subsample': sp_uniform(loc=0.2, scale=0.8),
              'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
              'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
              'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

# Number of hyperparameter combinations to test
n_HP_points_to_test = 100

# Initialize the LightGBM classifier
clf = lgb.LGBMClassifier(max_depth=-1, random_state=314, silent=True, metric='None', n_jobs=4, n_estimators=5000)

# Perform RandomizedSearchCV to find the best hyperparameters
gs = RandomizedSearchCV(
    estimator=clf, param_distributions=param_test,
    n_iter=n_HP_points_to_test,
    scoring='roc_auc',
    cv=3,
    refit=True,
    random_state=314,
    verbose=True)

# Fit the model on the training data with the best hyperparameters
gs.fit(*train, **fit_params)

# Print the best estimator found by RandomizedSearchCV
print(gs.best_estimator_)