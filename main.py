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

app = FastAPI()


app.add_middleware(             # cors보안 규칙 인가
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

price = pyupbit.get_current_price("KRW-XRP")
print(price)


@app.get("/responsePrice/{ticker}")
async def read_root(ticker: str):

    pred_price, real_price, date = get_predict_crypto_price(ticker)     # 전달받은 가상화폐 ticker를 함수에 인자값으로 전달

    return {"days":date, "pred_price":pred_price, "real_price":real_price}           # 일시와 예측 가격데이터를 spring서버로 전달



def get_predict_crypto_price(ticker):                   # 가상화폐의 가격을 예측하는 사용자 함수

    df = pyupbit.get_ohlcv(f"KRW-{ticker}", count=3000, interval="minute60", period=0.1)     # 원화 단위의 가상화폐, 시간 단위는 분 단위, 현재 시점부터 2000분 전의 데이터를 요청
    df['y'] = df['close']
    df['ds'] = df.index

    real_price = df['y']                                # 실제 가격 추세

    search_space = {
    'changepoint_prior_scale': [0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    'seasonality_prior_scale': [0.05, 0.1, 1.0, 10.0],
    'holidays_prior_scale': [0.05, 0.1, 1.0, 10.0],
    'seasonality_mode': ['additive', 'multiplicative'],
    }


    m = prh(seasonality_mode='multiplicative')          # Prophet 파라미터 설정

    m.fit(df)                                           # Prophet 모델링

    future = m.make_future_dataframe(periods=3000)      # 예상 주기 설정

    forecast = m.predict(future)                        # 예측한 값을 forecast변수에 저장

    forecast['yhat'] = forecast['yhat'].astype('float') # 숫자형식을 int로 변환

    forecast['ds'] = forecast['ds'].astype('str')

    pred_price = []                                     # 데이터 프레임에 담겨있는 예측한 가격 데이터를 리스트에 보관
    for i in forecast['yhat']:
        pred_price.append(i)
    pred_price



    date = []                                           # 데이터 프레임에 담겨있는 날짜 데이터를 리스트에 보관
    for i in forecast['ds']:
        date.append(i)
    date
    return pred_price, real_price, date                 # 예측한 가격 추세, 실제 가격 추세, 처음 일시부터 마지막 예측 일시까지 반환