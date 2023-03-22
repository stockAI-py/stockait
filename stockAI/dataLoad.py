import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def get_tickers(markets:list, date:str): 
    
    '''
    
    [ Explanation ]
    Bring up the tickers
    
    
    [ input ] 
    - markets: (list) Put the market you want to bring in in on the list.
    - date: (str) 
    
    
    [ output ]
    - lst_tickers: (list) Bring up the tickers that are being traded based on this date
  
    '''
    
    df = pd.read_parquet("https://media.githubusercontent.com/media/stockAir-py/stock_data/main/v1_krx_2001_2022/krx_20010101_20221231.parquet")
    df_market = df[df['Market'].isin(markets)]
    lst_tickers = df_market.loc[df_market['Date'] >= f'{date}-01-01', 'Code'].unique().tolist() 
    return lst_tickers



def load_data(date:list, tickers:list):
    
    '''
    
    [ Explanation ]
    Bring up the data
    
    
    [ input ] 
    - date: (list) Specify the date you want to load like [start_date, end_date]
    - tickers: (list) List the tickers you want to bring in.
    
    
    [ output ]
    - df_code: (pd.DataFrame) Daily stock price data for dates and stocks entered as input values.
    
    '''
    
    df = pd.read_parquet("https://media.githubusercontent.com/media/stockAI-py/stock_data/main/v1_krx_2001_2022/krx_20010101_20221231.parquet")
    df_date = df[(df['Date'] >= date[0]) & (df['Date'] <= date[1])]
    df_code = df_date[df_date['Code'].isin(tickers)].reset_index(drop=True)
    
    return df_code
