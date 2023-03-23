import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm 
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


def decision(lst_trader:list, dtype='test', data=None, data_scaled=None):
    
    '''
    
    [ Explanation ]
    A function that creates a data frame that records buying and selling behavior for all dates
    
    
    [ input ] 
    - lst_trader: (pd.DataFrame) Data to convert into time series data
    - dtype: (str) If 'test', calculate the rate of return from the test dataset stored in the trader, 
                   If 'none' is data, data_sclared is used to calculate the rate of return with the desired data set.
    - data & data_scaled (pd.DataFrame) Dataset to calculate yield if dtype is set to 'none'
    
    
    [ output ]
    - df_signal_all: (pd.DataFrame) Selling, buying records for all dates
    
    '''  
    
        
    df_signal_all = pd.DataFrame()
    for trader in lst_trader:
        
        # Write a buying diary, a selling diary
        df_signal_buy = trader.buyer.decision_all(trader, dtype='test', data=None, data_scaled=None)
        df_signal_sell = trader.seller.decision_all(trader, dtype='test', data=None, data_scaled=None)
        
        # dataframe concatenate
        df_signal = pd.concat([df_signal_buy, df_signal_sell], axis=0)
        
        # df_signal_all: trader_id, date, code, +(buy)/-(sell), amount, close
        df_signal_all = df_signal_all.append(df_signal)
        
        print(f'== {trader.name} completed ==')
    
    return df_signal_all 
        

    
def simulation(df_signal_all, init_budget, init_stock, fee=0.01): 
    
    '''
    
    [ Explanation ]
    A function that calculates the rate of return from the trading log (df_signal_all).
    
    
    [ input ] 
    - df_signal_all: (pd.DataFrame) decision The resulting value of the function. Sales Diary Data Frame.
    - init_budget: (int) an initial budget
    - init_stock: (dict) Type of {'ticker': count}. It is a dictionary containing the number of stocks purchased. You can put in an empty dictionary unless it's a special case.
    
    
    [ output ]
    - df_history_all: (pd.DataFrame) Data frames of buy, sell records for all dates
    
    '''  
    
    df_history_all = pd.DataFrame()
    for trader_id, df_signal in df_signal_all.groupby('Trader_id'):

        budget = init_budget  
        stock = init_stock          

        df_signal = df_signal.sort_values(by='Date') 
        
        lst_history = [] # [[trader_id, date, budget, day_yield, stock], ...]
        for idx, (date, df_signal_date) in enumerate(tqdm(df_signal.groupby('Date'))):  
            df_signal_date = df_signal_date.sort_values(by='Code')

            temp_stock = stock.copy() 
            lst_code = list(temp_stock.keys())
            
            df_sell = df_signal_date[(df_signal_date['+(buy)/-(sell)'] == '-') & (df_signal_date['Code'].isin(lst_code))]
            for row_sell in df_sell.values.tolist():
                code, amount, close = row_sell[2], row_sell[4], row_sell[5]
                budget += close * stock[code] * (1.0-fee)
                stock[code] *= (1-amount)
                
                if stock[code] == 0.0:
                    del stock[code]

            day_yield = (budget-init_budget) / init_budget * 100 
            lst_history.append([trader_id, date, int(budget), day_yield, temp_stock])
        
            if idx == len(df_signal.groupby('Date'))-1:
                break  
                
            temp_budget = budget

            # buy for each code  
            df_buy = df_signal_date[(df_signal_date['+(buy)/-(sell)'] == '+') & (df_signal_date['Amount']>0)]
            for row_buy in df_buy.values.tolist():
                code, amount, close = row_buy[2], row_buy[4], row_buy[5]
                
                if code not in stock.keys():
                    stock[code] = 0
                
                amount /= len(df_buy)
                stock[code] += int( temp_budget * amount / close )
                budget -= close * stock[code] * (1.0+fee)
            
        df_history_all = df_history_all.append(pd.DataFrame(lst_history, columns=['Trader_id', 'Sell_date', 'Budget', 'Yield', 'Stock']))
        print(f"== {trader_id} completed ==")

    return df_history_all



def leaderboard(df):
    
    '''
    
    [ Explanation ]
    A function that outputs a per-trader yield leaderboard data frame.
    
    
    [ input ] 
    - df: (pd.DataFrame) Sales record (df_history_all)
    
    
    [ output ]
    - df_rank: (pd.DataFrame) Data frames with yields sorted in descending order
    
    '''  
    
    df_rank = pd.DataFrame()
    for trader_id in df['Trader_id'].unique():
        sr1 = df.loc[df['Trader_id']==trader_id, ['Trader_id', 'Yield']].iloc[-1]
        df_rank = df_rank.append(sr1)
    return df_rank.sort_values(by='Yield', ascending=False).reset_index(drop=True)



def yield_plot(df):
    
    '''
    
    [ Explanation ]
    A function that outputs a graph that visualizes returns for all traders and dates    
    
    [ input ] 
    - df: (pd.DataFrame) Sales record (df_history_all)
    
    
    [ output ]
    - df_rank: (pd.DataFrame) Yield visualization graph
    
    '''  
    
    mpl.style.use("seaborn-whitegrid")

    mpl.rcParams['font.family'] ='NanumSquareRound'
    mpl.rcParams['axes.unicode_minus'] =False
    
    
    f,ax = plt.subplots(1,1,figsize=(25,12),sharex=False)
    df = df.reset_index(drop=True)
        
    
        
    ax = sns.lineplot(data=df, x='Sell_date', y='Yield', hue='Trader_id', linewidth=4)

    _=plt.xticks(range(0, 250, 20), fontsize=20, rotation = 35)
    _=plt.yticks(range(int(df.Yield.min())-15,int(df.Yield.max())+15,20), fontsize=20)
    ax.set_xlabel('Date', fontsize=25, labelpad=25)
    ax.set_ylabel('Yield', fontsize=25, labelpad=30)
    _=plt.legend(fontsize=20)