import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import boto3
import os
import sqlalchemy 
from sqlalchemy import create_engine
from dotenv import load_dotenv
load_dotenv()


def get_tickers(markets:list): 
    
    '''
    
    [ Explanation ]
    Bring up the tickers
    
    
    [ input ] 
    - markets: (list) Put the markets you want to bring in in on the list.
    
    
    [ output ]
    - lst_tickers: (list) Outputs the tickers for the entered markets.
  
    '''
    AWS_ACCESS_KEY_ID=os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY=os.getenv("AWS_SECRET_ACCESS_KEY")
    REGION=os.getenv("REGION")
    
    dynamodb = boto3.resource('dynamodb',
                              region_name=REGION,
                              aws_access_key_id=AWS_ACCESS_KEY_ID,
                              aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    table = dynamodb.Table("MARKET_CODE")
    
    lst_tickers = []
    for market in markets:
        lst_tickers += table.get_item(Key={"key_market":market})["Item"]['Code'] ## 코넥스의 데이터만 불러온다면
    
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
    host=os.getenv("HOST")
    user=os.getenv("USERNAME")
    passwd=os.getenv("PASSWORD")
    db="STOCK_DATA"
    
    db_connection_str = f'mysql+pymysql://{user}:{passwd}@{host}/{db}'
    db_connection = create_engine(db_connection_str)
    conn = db_connection.connect()

    df_code = pd.DataFrame() 
    for code in tqdm(tickers): 
        sql_query = f'''
                    SELECT * 
                    FROM stock_{code}
                    WHERE Date BETWEEN "{date[0]}" AND "{date[1]}"
                    '''
        stock_code = pd.read_sql(sql = sql_query, con = conn) 
        df_code = pd.concat([df_code, stock_code], axis=0)

    df_code = df_code.sort_values(by=["Code", 'Date'])
    
    return raw_stock_df