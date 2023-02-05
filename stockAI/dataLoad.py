import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')



# 종목 불러오기
def get_tickers(markets:list, date:str): 
    df = pd.read_csv("https://media.githubusercontent.com/media/stockAI-py/data-test/main/KOR/data_2001_2022.csv")
    df_market = df[df['Market'].isin(markets)]
    lst_tickers = df_market.loc[df_market['Date'] >= f'{date}-01-01', 'Code'].unique().tolist() 
    return lst_tickers


# 주가 데이터 불러오기
def load_data(date, tickers):
    df = pd.read_csv("https://media.githubusercontent.com/media/stockAI-py/data-test/main/KOR/data_2001_2022.csv")
    df_date = df[(df['Date'] >= date[0]) & (df['Date'] <= date[1])]
    df_code = df_date[df_date['Code'].isin(tickers)].reset_index(drop=True)
    
    return df_code