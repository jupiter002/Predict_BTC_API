# pip install finance-datareader prophet matplotlib seaborn plotly bs4 fastapi pyupbit
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

# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input
# from tensorflow.keras.layers import LSTM
# from numpy import array

import pyupbit
import FinanceDataReader as fdr
from prophet import Prophet as prh
from prophet.plot import add_changepoints_to_plot
import time

app = FastAPI()


app.add_middleware(             # cors보안 규칙 인가
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/responsePrice/{ticker}")
async def read_root(ticker: str):

    pred_price, real_price, date = get_predict_crypto_price(ticker)     # 전달받은 가상화폐 ticker를 함수에 인자값으로 전달

    return {"days":date, "pred_price":pred_price, "real_price":real_price}           # 일시와 예측 가격데이터를 spring서버로 전달


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}


# AI API


def fitting_to_real_price(df):                          # 학습 데이터를 Fitting 시키는 사용자 함수
    
    
    m = prh(                                            
    changepoint_prior_scale=0.3,
    growth="linear"
    )


    m.fit(df)                                           # 학습데이터 Fitting

    future = m.make_future_dataframe(periods=3000)      # 예상 주기 설정

    forecast = m.predict(future)                        # 예측한 값을 forecast변수에 저장

    forecast['yhat'] = forecast['yhat'].astype('float') # 숫자형식을 float로 변환

    # forecast['ds'] = forecast['ds'].astype('str')     # 예측과 실제 가격 추세 그래프를 양쪽으로 나눠서 그릴 수 있음
    
    return forecast                                     # 예측값 반환


def get_predict_crypto_price(ticker):                   # 가상화폐의 가격을 예측하는 사용자 함수

    df = pyupbit.get_ohlcv(f"KRW-{ticker}", count=3000, interval="minute60", period=0.1)     # 원화 단위의 가상화폐, 시간 단위는 분 단위, 현재 시점부터 2000분 전의 데이터를 요청
    df['y'] = df['close']
    df['ds'] = df.index

    real_price = df['y']                                # 실제 가격 추세

    # search_space = {
    # 'changepoint_prior_scale': [0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    # 'seasonality_prior_scale': [0.05, 0.1, 1.0, 10.0],
    # 'holidays_prior_scale': [0.05, 0.1, 1.0, 10.0],
    # 'seasonality_mode': ['additive', 'multiplicative'],
    # }






def get_crypto_price(ticker="BTC"):                   # 가상화폐의 가격을 가져오는 사용자 함수

    df = pyupbit.get_ohlcv(f"KRW-{ticker}", count=2000, interval="minute1")     # 원화 단위의 가상화폐, 시간 단위는 분 단위, 현재 시점부터 2000분 전의 데이터를 요청
    df['y'] = df['close']
    df['ds'] = df.index

    forecast = fitting_to_real_price(df)
    
    pred_price = []                                     # 데이터 프레임에 담겨있는 예측한 가격 데이터를 리스트에 보관
    for i in forecast['yhat']:
        pred_price.append(i)
    pred_price



    date = []                                           # 데이터 프레임에 담겨있는 날짜 데이터를 리스트에 보관
    for i in forecast['ds']:
        date.append(i)
    date

    real_price = df['y']

    return pred_price, real_price, date                             # 예측 가격, 실제 가격 추세 일시를 반환


def Real_time_Prediction(ticker="BTC"):                  # 실시간 가격을 계속 추가하는 사용자 함수
    pred_price, real_price, date = get_crypto_price()

    while True:
        current_crypto = pyupbit.get_current_price(f"KRW-{ticker}")
        



pyupbit.get_current_price(f"KRW-BTC")
a, b, c, d, e, f, g, h, i = time.localtime()
a
print(pyupbit.get_daily_ohlcv_from_base("KRW-BTC", base=13))