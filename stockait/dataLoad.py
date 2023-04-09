import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import boto3
import os
import sqlalchemy 
from sqlalchemy import create_engine



def get_countries(): 
    
    '''
    
    [ Explanation ]
    Get the country provided by stockait
    
    [ output ]
    - lst_countries: (list) Returns the list of countries provided.

    '''
    
    return ['Argentina', 'Australia', 'Austria', 'Belgium', 'Brazil', 'Canada', 'China', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hong Kong', 'Iceland', 'India', 'Indonesia', 'Ireland', 'Israel', 'Italy', 'Latvia', 'Lithuania', 'Malaysia', 'Mexico', 'Netherlands', 'New Zealand', 'Norway', 'Portugal', 'Qatar', 'Russia', 'Singapore', 'South Korea', 'Spain', 'Sweden', 'Switzerland', 'Taiwan', 'Thailand', 'Turkey', 'USA', 'United Kingdom', 'Venezuela']


    
def get_markets(country: str):
    
    '''
    
    [ Explanation ]
    Bring up the tickers
    
    
    [ input ] 
    - markets: (str) Put the country you want 
    
    
    [ output ]
    - lst_tickers: (list) Outputs the markets for the entered country.
  
    '''
    
    dic_country2market = {'Argentina': ['BUE'], 'Australia': ['ASX'], 'Austria': ['VIE'], 'Belgium': ['BRU'], 'Brazil': ['SAO'], 'Canada': ['CNQ', 'TOR', 'VAN'], 'China': ['SHH', 'SHZ'], 'Denmark': ['CPH'], 'Estonia': ['TAL'], 'Finland': ['HEL'], 'France': ['ENX', 'FRA', 'PAR'], 'Germany': ['BER', 'DUS', 'EUX', 'GER', 'HAM', 'HAN', 'MUN', 'STU'], 'Greece': ['ATH'], 'Hong Kong': ['HKG'], 'Iceland': ['ICE'], 'India': ['BSE', 'NSI'], 'Indonesia': ['JKT'], 'Ireland': ['ISE'], 'Israel': ['TLV'], 'Italy': ['MIL', 'TLO'], 'Latvia': ['RIS'], 'Lithuania': ['LIT'], 'Malaysia': ['KLS'], 'Mexico': ['MEX'], 'Netherlands': ['AMS'], 'New Zealand': ['NZE'], 'Norway': ['OSL'], 'Portugal': ['LIS'], 'Qatar': ['DOH'], 'Russia': ['MCX'], 'Singapore': ['SES'], 'South Korea': ['KOSPI', 'KOSDAQ', 'KONEX'], 'Spain': ['MAD', 'MCE'], 'Sweden': ['STO'], 'Switzerland': ['EBS'], 'Taiwan': ['TAI', 'TWO'], 'Thailand': ['SET'], 'Turkey': ['IST'], 'USA': ['ASE', 'NCM', 'NGM', 'NMS', 'NYQ', 'OBB', 'PCX', 'PNK'], 'United Kingdom': ['IOB', 'LSE'], 'Venezuela': ['CCS']}

    return dic_country2market[country]
    
    
def get_tickers(markets:list): 
    
    '''
    
    [ Explanation ]
    Bring up the tickers
    
    
    [ input ] 
    - markets: (list) Put the markets you want to bring in in on the list.
    
    
    [ output ]
    - lst_tickers: (list) Outputs the tickers for the entered markets.
  
    '''
    
    # read_only account 
    AWS_ACCESS_KEY_ID="AKIAUCRIFWMNVPXPQCZ6"
    AWS_SECRET_ACCESS_KEY="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"
    REGION="ap-southeast-2"
    
    dynamodb = boto3.resource('dynamodb',
                              region_name=REGION,
                              aws_access_key_id=AWS_ACCESS_KEY_ID,
                              aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    table = dynamodb.Table("MARKET_CODE")
    
    lst_tickers = []
    for market in markets:
        lst_tickers += table.get_item(Key={"key_market":market})["Item"]['Code'] 

    return [t for t in lst_tickers if "K" not in t]



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

    # read_only account 
    host="146.56.39.107"
    user="stockuser"
    passwd="stockai123"
    db="STOCK_DATA"
    
    db_connection_str = f'mysql+pymysql://{user}:{passwd}@{host}/{db}'
    db_connection = create_engine(db_connection_str)
    conn = db_connection.connect()

    df_code = pd.DataFrame() 
    for code in tqdm(tickers): 
        sql_query = f'''
                    SELECT * 
                    FROM stock_{code}
                    WHERE (Date BETWEEN "{date[0]}" AND "{date[1]}") AND (Open NOT IN (0)) AND (Low != High) 
                    '''
        stock_code = pd.read_sql(sql = sql_query, con = conn) 
        df_code = pd.concat([df_code, stock_code], axis=0)

    df_code = df_code.sort_values(by=["Code", 'Date'])
    
    return df_code.reset_index(drop=True)
