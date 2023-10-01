import requests
import datetime
import pandas as pd
import io
import numpy as np

# 下載某天的資料
date = datetime.date(2019, 1, 2)
datestr = date.strftime("%Y%m%d")

res = requests.get(
    "https://www.twse.com.tw/exchangeReport/MI_5MINS_INDEX?response=csv&date=" + datestr + "&_=1631534526262")
print(res.text[:1000])

# 將資料用pandas整理成表格
df = pd.read_csv(io.StringIO(res.text.replace("=", "")), header=1, index_col="時間")

# 資料處理
df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
df.index = pd.to_datetime(datestr + ' ' + df.index)
df = df.applymap(lambda s: float(str(s).replace(",", "")))
print(df)
