# pip install finance-datareader prophet matplotlib seaborn plotly bs4
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from numpy import array
import pyupbit
import FinanceDataReader as fdr
from prophet import Prophet as prh
from prophet.plot import add_changepoints_to_plot

app = FastAPI()


app.add_middleware(             # cors보안 규칙 인가
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello, Prophet!!!"}


@app.post("/responsePrice")
def read_root():
    return {"days":date,"pred_price":pred_price}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}


# AI API

df = pyupbit.get_ohlcv("KRW-ETH", count=10000, interval="minute1")      # 업비트 데이터 
df['y'] = df['close']
df['ds'] = df.index
df['cap'] = df['low']
df['floor'] = df['high']
df

sum = 0
for i in range(len(df)):
    if df['cap'][i] >= df['floor'][i]:
        print(df['floor'][i])
        df['floor'][i] = df['floor'][i] + 1
        print(df['floor'][i])
        sum = sum + 1
        
print(sum)

print(df['floor'][9] + 1)

#changepoint_prior_scale=0.1,
#holidays_prior_scale=5.0,
#interval_width=0.95,

m = prh(                            # Prophet 파라미터 설정
    changepoint_prior_scale=1,
    growth="linear"
)

m.fit(df)                                          # Prophet 모델링

future = m.make_future_dataframe(periods=1)       # 예상 주기 설정


forecast = m.predict(future)                        # 예측한 값을 forecast변수에 저장

forecast['yhat'] = forecast['yhat'].astype('float') # 숫자형식을 int로 변환

#forecast['ds'] = forecast['ds'].astype('str')       # datetime형식에서 문자열 타입으로 변환
forecast

fig = m.plot(forecast)

plt.show()

pred_price = []                                     # 데이터 프레임에 담겨있는 예측한 가격 데이터를 리스트에 보관
for i in forecast['yhat']:
    pred_price.append(i)
pred_price


date = []                                           # 데이터 프레임에 담겨있는 날짜 데이터를 리스트에 보관
for i in forecast['ds']:
    date.append(i)
date

inputs1 = Input(shape=(3, 1))
lstm1 = LSTM(1)(inputs1)
model = Model(inputs=inputs1, outputs=lstm1)

data = array([0.1, 0.2, 0.3]).reshape((1,3,1))

print(model.predict(data)) # [[0.04836999]]