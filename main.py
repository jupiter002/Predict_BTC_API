# pip install finance-datareader prophet matplotlib seaborn plotly bs4
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request

import pandas as pd
import numpy as np

import FinanceDataReader as fdr
from prophet import Prophet as prh
from prophet.plot import add_changepoints_to_plot
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

templates = Jinja2Templates(directory="templates")

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/")
def read_root(request: Request):
    return JSONResponse(content=pred_price)


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}


# AI API

BTC = fdr.DataReader("BTC/KRW")     # 비트코인 종가 데이터 수집

BTC['y'] = BTC['Close']             #  종가
BTC['ds'] = BTC.index               #  기간

BTC

m = prh(                            # Prophet 파라미터 설정
    growth="linear",
    interval_width=0.95,
)   

m.fit(BTC)                                          # Prophet 모델링

future = m.make_future_dataframe(periods=365)       # 예상 주기 설정


forecast = m.predict(future)                        # 예측한 값을 forecast변수에 저장

forecast['yhat'] = forecast['yhat'].astype('float')   # 숫자형식을 int로 변환

forecast['ds'] = forecast['ds'].astype('str')

pred_price = []
for i in forecast['yhat']:
    pred_price.append(i)
pred_price


date = []
for i in forecast['ds']:
    date.append(i)
date