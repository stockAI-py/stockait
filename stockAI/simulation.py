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

# 매매일지 작성
def decision(lst_trader:list, dtype='test', data=None, data_scaled=None):
    
    '''
    
    lst_trader
    - 트레이더들이 저장된 리스트 
    
    dtype
    - [default] 'test' -> 트레이더에 저장되어있는 test dataset으로 계산 
    - None -> 사용자 지정 데이터 
    
    data
    - [default] None 
    - 원본 데이터 
    
    data_scaled
    - [default] None 
    - 표준화 데이터 
    
    '''
        
    df_signal_all = pd.DataFrame()
    for trader in lst_trader:
        # 매수일지, 매도일지 작성 
        df_signal_buy = trader.buyer.decision_all(trader, dtype='test', data=None, data_scaled=None)
        df_signal_sell = trader.seller.decision_all(trader, dtype='test', data=None, data_scaled=None)
        
        # dataframe concatenate
        df_signal = pd.concat([df_signal_buy, df_signal_sell], axis=0)
        
        # df_signal_all: trader_id, date, code, +(buy)/-(sell), amount, close
        df_signal_all = df_signal_all.append(df_signal)
        
        print(f'== {trader.name} complete ==')
    
    return df_signal_all 
        
# 수익률 계산
def simulation(df_signal_all, init_budget, init_stock): 
    
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
                budget += close * stock[code]
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
                budget -= close * stock[code] 
            
        df_history_all = df_history_all.append(pd.DataFrame(lst_history, columns=['Trader_id', 'Sell_date', 'Budget', 'Yield', 'Stock']))
        print(f"== {trader_id} complete ==")

    return df_history_all

# 리더보드
def leaderboard(df):
    df_rank = pd.DataFrame()
    for trader_id in df['Trader_id'].unique():
        sr1 = df.loc[df['Trader_id']==trader_id, ['Trader_id', 'Yield']].iloc[-1]
        df_rank = df_rank.append(sr1)
    return df_rank.sort_values(by='Yield', ascending=False).reset_index(drop=True)

# 수익률 시각화
def yield_plot(df):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
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