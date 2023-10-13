# tw_stock_predictor

## download data from: https://www.twse.com.tw/exchangeReport/MI_5MINS_INDEX
use auto_update.py to update the data.

## SVM predictor
use the features (['R103_ROE稅後', 'R402_營業毛利成長率']) to train SVM model.

## GradientBoost predictor
use GradientBoost.py to compute the result of lightgbm model.

### lgbm_BetterParams.py
use lgbm_BetterParams.py to compute the best parameters for GradientBoost fit the data.

## RandomForest predictor
use RandomForest.py to output the best features that influence stock the most. Then, combine them with tradition strategy.
