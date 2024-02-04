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


@app.get("/responsePrice/{ticker}")
async def read_root(ticker: str):

    pred_price, date = get_predict_crypto_price(ticker)     # 전달받은 가상화폐 ticker를 함수에 인자값으로 전달

    return {"days":date, "pred_price":pred_price}           # 일시와 예측 가격데이터를 spring서버로 전달


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}


# AI API

# BTC = fdr.DataReader("BTC/USD")     # 비트코인 종가 데이터 수집

# BTC['y'] = BTC['Close']             #  종가
# BTC['ds'] = BTC.index               #  기간

# sum = 0
# for i in range(len(df)):
#     if df['cap'][i] >= df['floor'][i]:
#         print(df['floor'][i])
#         df['floor'][i] = df['floor'][i] + 1
#         print(df['floor'][i])
#         sum = sum + 1
 

#changepoint_prior_scale=0.1,
#holidays_prior_scale=5.0,
#interval_width=0.95,

def get_predict_crypto_price(ticker):                   # 가상화폐의 가격을 예측하는 사용자 함수

    df = pyupbit.get_ohlcv(f"KRW-{ticker}", count=2000, interval="minute1")     # 원화 단위의 가상화폐, 시간 단위는 분 단위, 현재 시점부터 2000분 전의 데이터를 요청
    df['y'] = df['close']
    df['ds'] = df.index

    m = prh(                                            # Prophet 파라미터 설정
    changepoint_prior_scale=1,
    growth="linear"
    )

    m.fit(df)                                           # Prophet 모델링

    future = m.make_future_dataframe(periods=365)       # 예상 주기 설정

    forecast = m.predict(future)                        # 예측한 값을 forecast변수에 저장

    forecast['yhat'] = forecast['yhat'].astype('float') # 숫자형식을 int로 변환

    forecast['yhat'] = forecast['yhat'].astype('str')

    pred_price = []                                     # 데이터 프레임에 담겨있는 예측한 가격 데이터를 리스트에 보관
    for i in forecast['yhat']:
        pred_price.append(i)
    pred_price


    date = []                                           # 데이터 프레임에 담겨있는 날짜 데이터를 리스트에 보관
    for i in forecast['ds']:
        date.append(i)
    date

    return pred_price, date                             # 예측 가격과 일시를


