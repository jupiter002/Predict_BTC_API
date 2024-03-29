# pip install finance-datareader prophet matplotlib seaborn plotly bs4 fastapi pyupbit
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pyupbit
from prophet import Prophet as prh
from prophet.plot import add_changepoints_to_plot
from function import *

import math

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

app = FastAPI()


app.add_middleware(             # cors보안 규칙 인가
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/responsePrice/{ticker1}")
async def read_root(ticker1: str):

    pred_price1, real_price1 = get_crypto_price(ticker1)     # 전달받은 가상화폐 ticker를 함수에 인자값으로 전달
    # pred_price2, real_price2 = get_crypto_price(ticker2)
    pred_price1, real_price1, mse, mae, mape = get_crypto_price(ticker1)     # 전달받은 가상화폐 ticker를 함수에 인자값으로 전달
    #pred_price2, real_price2 = get_crypto_price(ticker2)


    #return [{"days":date1, "value":pred_price1}, {"days":date1, "value":real_price1},                        
    #        {"days":date1, "value":pred_price2}, {"days":date1, "value":real_price2}]           # 일시와 예측 가격데이터를 spring서버로 전달
    # return [{"days":date1, "value":pred_price1}, {"days":date1,"value":real_price1}, {"days":date2,"value":pred_price2}, {"days":date2,"value":real_price2}]           # 일시와 예측 가격데이터를 spring서버로 전달
    return [pred_price1, real_price1, mse, mae, mape]




# AI API
# df = pyupbit.get_ohlcv(f"KRW-BTC", count=2000, interval="minute1")


def fitting_to_real_price(df):                          # 학습 데이터를 Fitting 시키는 사용자 함수
    m = prh(                                            
    growth="linear",
    interval_width=0.95
    )

    m.fit(df)                                           # 학습데이터 Fitting

    future = m.make_future_dataframe(periods=240, freq='H')      # 예상 주기 설정

    forecast = m.predict(future)                        # 예측한 값을 forecast변수에 저장

    # forecast['yhat'] = forecast['yhat'].astype('float') # 숫자형식을 float로 변환

    # forecast['ds'] = forecast['ds'].astype('str')     # 예측과 실제 가격 추세 그래프를 양쪽으로 나눠서 그릴 수 있음
    
    return forecast                                     # 예측값 반환

# def fitting_to_real_price(df):                          # 학습 데이터를 Fitting 시키는 사용자 함수

#     window_size,forecast_size=240,24
#     ''' 1. 가공되지 않은 데이터 전처리 '''
#     date, data=split_data(df,'close')
#     ''' 2. dataloader 빌드 '''
#     dataloader=build_dataLoader(data,
#                                 window_size=window_size,
#                                 forecast_size=forecast_size,
#                                 batch_size=8)
#     ''' 3. 학습 및 검증 '''
#     pred=trainer(data,
#                 dataloader,
#                 window_size=window_size,
#                 forecast_size=forecast_size).implement() 
#     ''' 4. 결과 반환 '''
#     return date[4296:], data[4296:], pred


def get_crypto_price(ticker="BTC"):                   # 가상화폐의 가격을 가져오는 사용자 함수

    df = pyupbit.get_ohlcv(f"KRW-{ticker}", count=4320, interval="minute60")     # 원화 단위의 가상화폐, 시간 단위는 분 단위, 현재 시점부터 2000분 전의 데이터를 요청
    df.reset_index(inplace=True)
    df['y'] = df['close']
    df['ds'] = df['index']

    forecast = fitting_to_real_price(df)
    
    #pred_price = forecast['yhat']                             # 데이터 프레임에 담겨있는 예측한 가격 데이터

    #real_price = df['y']                                      # 실제 가격 데이터

    date = forecast['ds']                                     # 데이터 프레임에 담겨있는 날짜 데이터
    
    real_price_list = []
    pred_price_list = []

    for i in range(len(date)):
        pred_price = {"days":date[i],"value":forecast['yhat'][i]}

        pred_price_list.append(pred_price)

    for i in range(len(df['y'])):
        real_price = {"days":date[i],"value":df['y'][i]}

        real_price_list.append(real_price)

    mse = round(math.sqrt(mean_squared_error(df['y'], forecast['yhat'][:4320])),2)
    mae = round(math.sqrt(mean_absolute_error(df['y'], forecast['yhat'][:4320])),2)
    mape = round(math.sqrt(mean_absolute_percentage_error(df['y'], forecast['yhat'][:4320])),2)

    output_mse = f"평균 제곱 오차 (MSE) : {mse}"
    output_mae = f"평균 절대 오차 (MAE) : {mae}" 
    output_mape = f"평균 절대 오차 비율 (MAPE) : {mape}%" 

    # return pred_price, real_price
    return pred_price_list, real_price_list, output_mse, output_mae, output_mape                # 예측 가격, 실제 가격 추세 일시를 반환


# def get_crypto_price(ticker):                   # 가상화폐의 가격을 가져오는 사용자 함수
#     df = pyupbit.get_ohlcv(f"KRW-{ticker}", count=4320, interval="minute60")     # 원화 단위의 가상화폐, 시간 단위는 분 단위, 현재 시점부터 2000분 전의 데이터를 요청

#     date, data, pred = fitting_to_real_price(df)

#     real_price_list = []
#     pred_price_list = []

#     print(f"""
# date: {date}
# data: {data[0]}
# pred: {pred}
# date: {type(date)}
# data: {type(data[0])}
# pred: {type(pred)}
# """)

#     pred_Series = pd.Series(pred)
#     real_Series = pd.Series(data)

#     # pred_Series = []
#     # real_Series = []

#     # for value in pred:
#     #     value = value.astype("int")
#     #     pred_Series.append(value)

#     # for value in data:
#     #     value = value.astype("int")
#     #     real_Series.append(value)


#     print(f"""
# pred_list: {type(pred_Series)}
# real_list: {type(real_Series)}
# """)

#     for i in range(len(date)):
#         pred_price = {"days":date[i],"value":pred_Series[i]}

#         pred_price_list.append(pred_price)

#     for i in range(len(date)):
#         real_price = {"days":date[i],"value":real_Series[i]}

#         real_price_list.append(real_price)

#     print(f"""
# pred_price_list: {type(pred_price_list[0]["value"])}
# real_price_list: {real_price_list}
# """)

#     return pred_price_list, real_price_list
    # return pred_price_list, real_price_list                 # 예측 가격, 실제 가격 추세 일시를 반환



