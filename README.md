# tw_stock_predictor

## Dataset
### auto_update.py
download data from: https://www.twse.com.tw/exchangeReport/MI_5MINS_INDEX  
This file is responsible for automatically updating stock datafrom TWSE(Taiwan Stock Exchange).  
It utilizes finlab.crawler to fetch data. 
```mermaid
flowchart LR
A(TWSE) -->|auto_update.py| B(collect data)
B(collect data) -->|data wrangling| C(save)
```

## traditional strategy
### traditional_strategy.py
calculate the deviation rate between the quarterly moving average and the daily chart.  
When the stock price is greater than one standard deviation above the deviation rate, buy it.  
When it's lower than one standard deviation below the deviation rate, sell it (buy high, sell low).  
At the same time, including the transaction fees of Taiwan Stock Exchange into the cost.  
```mermaid
graph LR
A[get moving average] -->|calculate| B(standard deviation)
B --> C{greater or lower?}
C -->|greater| D[Buy]
C -->|lower| E[Sell]
```

## GradientBoost predictor
### GradientBoost.py 
This file uses GradientBoostingRegressor to predict stock prices. It first fetches data from finlab.data, performs feature engineering, and then trains the model.
```mermaid
flowchart LR
A(data) -->|GradientBoost.py| B(training complete)
B(training complete) -->|backtest| C(return)
```

### lgbm_BetterParams.py
use to optimize the parameters
compute the best parameters for GradientBoost to fit the data.
website: https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc

## RandomForest and traditional strategy predictor
### RandomForest.py
This file uses RandomForestRegressor to find the best features that influence stock the most. Then, combine them with tradition strategy.
theory: https://link.springer.com/content/pdf/10.1023/A:1010933404324.pdf
```mermaid
flowchart LR
A(data) -->|RandomForest.py| B(training complete)
B(training complete) -->|backtest| C(return)
```

## Finlab
```mermaid
graph TD
    finlab.backtest -->|提供| 回測功能
    finlab.crawler -->|提供| 抓取股票數據
    finlab.data -->|使用| finlab.crawler
    finlab.data -->|提供| 獲取和處理股票數據
    finlab.labels -->|提供| 生成股票數據的標籤
    finlab.ml -->|提供| 機器學習工具
    finlab.plot_candles -->|提供| 繪製蠟燭圖
    finlab.utility -->|提供| 實用工具
    lgbm_BetterParams -->|使用| LGBMRegressor
    traditional_strategy -->|提供| 傳統股票交易策略
```
