import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import ta
from tqdm import tqdm 


def add_index(data, index_list:list):
    
    '''
    
    [ Explanation ]
    Adding multiple secondary indicators to stock price data
    
    
    [ input ] 
    - data: (pd.DataFrame) Data to add secondary indicators
    - index_list: (list) List of secondary indicators to add
    
    
    [ output ]
    - result: (pd.DataFrame) Data Frames with Additional Indicators
    
    '''
    
    result = pd.DataFrame()
    
    tickers = list(data['Code'].unique())
    
    for code, stock_df in tqdm(data.groupby('Code')):
        stock_df = stock_df.sort_values(by="Date")
        H, L, C, V = stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Volume']
        
        stock_df['Change'] = (stock_df['Close'] - stock_df['Close'].shift(1))/stock_df['Close'].shift(1)
        
        
        add_form = {'MA5':stock_df['Close'].rolling(window = 5).mean(),
                    'MA20':stock_df['Close'].rolling(window = 20).mean(),
                    'MA60':stock_df['Close'].rolling(window = 60).mean(),
                    'MA120':stock_df['Close'].rolling(window = 120).mean(),
                    
#                     'trading_value':stock_df['Close']*stock_df['Volume'],
                    'next_change':stock_df['Change'].shift(-1),
                    
#                     'MFI':ta.volume.money_flow_index(high=H, low=L, close=C, volume=V, fillna=True),
#                     'OBV':ta.volume.on_balance_volume(close=C, volume=V, fillna=True),
#                     'FI':ta.volume.force_index(close=C, volume=V, fillna=True),
#                     'EOM_EMV':ta.volume.ease_of_movement(high=H, low=L, volume=V, fillna=True),
#                     'NVI':ta.volume.negative_volume_index(close=C, volume=V, fillna=True),
                    'ADI':ta.volume.acc_dist_index(high=H, low=L, close=C, volume=V, fillna=True),
                    'CMF':ta.volume.chaikin_money_flow(high=H, low=L, close=C, volume=V, fillna=True),
                    'VPT':ta.volume.volume_price_trend(close=C, volume=V, fillna=True),
                    'VMAP':ta.volume.volume_weighted_average_price(high=H, low=L, close=C, volume=V, fillna=True),
                    
#                     'ATR':ta.volatility.average_true_range(high=H, low=L, close=C, fillna=True),
                    'BHB':ta.volatility.bollinger_hband(close=C, fillna=True),
                    'BLB':ta.volatility.bollinger_lband(close=C, fillna=True),
                    'KCH':ta.volatility.keltner_channel_hband(high=H, low=L, close=C, fillna=True),
                    'KCL':ta.volatility.keltner_channel_lband(high=H, low=L, close=C, fillna=True),
                    'KCM':ta.volatility.keltner_channel_mband(high=H, low=L, close=C, fillna=True),
                    'DCH':ta.volatility.donchian_channel_hband(high=H, low=L, close=C, fillna=True),
                    'DCL':ta.volatility.donchian_channel_lband(high=H, low=L, close=C, fillna=True),
                    'DCM':ta.volatility.donchian_channel_mband(high=H, low=L, close=C, fillna=True),
                    'UI':ta.volatility.ulcer_index(close=C, fillna=True),
                    
#                     'ADX':ta.trend.adx(high=H, low=L, close=C, fillna=True),                    
                    'SMA':ta.trend.sma_indicator(close=C, fillna=True),
                    'EMA':ta.trend.ema_indicator(close=C, fillna=True),
                    'WMA':ta.trend.wma_indicator(close=C, fillna=True),
                    'MACD':ta.trend.macd(close=C, fillna=True),
                    'VIneg':ta.trend.vortex_indicator_neg(high=H, low=L, close=C, fillna=True),
                    'VIpos':ta.trend.vortex_indicator_pos(high=H, low=L, close=C, fillna=True),
                    'TRIX':ta.trend.trix(close=C, fillna=True),
                    'MI':ta.trend.mass_index(high=H, low=L, fillna=True),
                    'CCI':ta.trend.cci(high=H, low=L, close=C, fillna=False),
                    'DPO':ta.trend.dpo(close=C, fillna=True),
                    'KST':ta.trend.kst(close=C, fillna=True),
                    'Ichimoku':ta.trend.ichimoku_a(high=H, low=L, fillna=True),
                    'ParabolicSAR':ta.trend.psar_down(high=H, low=L, close=C, fillna=True),
                    'STC':ta.trend.stc(close=C, fillna=True),
                    
                    'RSI':ta.momentum.rsi(close=C, fillna=True),
                    'SRSI':ta.momentum.stochrsi(close=C, fillna=True),
                    'TSI':ta.momentum.tsi(close=C, fillna=True),
                    'UO':ta.momentum.ultimate_oscillator(high=H, low=L, close=C, fillna=True),
                    'SR':ta.momentum.stoch(close=C, high=H, low=L, fillna=True),
                    'WR':ta.momentum.williams_r(high=H, low=L, close=C, fillna=True),
                    'AO':ta.momentum.awesome_oscillator(high=H, low=L, fillna=True),
                    # 'KAMA':ta.momentum.kama(close=C, fillna=False),
                    'ROC':ta.momentum.roc(close=C, fillna=True),
                    'PPO':ta.momentum.ppo(close=C, fillna=True),
                    'PVO':ta.momentum.pvo(volume=V, fillna=True)
                   }
        
        for index in index_list:
            stock_df[index] = add_form[index]
            
        result = result.append(stock_df)
        result = result.dropna()
    
    return result.reset_index(drop=True)




def scaling(data, scaler_type):
    
    '''
    
    [ Explanation ]
    Functions that standardize stock price data
    
    
    [ input ] 
    - data: (pd.DataFrame) Data to Standardize
    - scaler_type: (str) Choose between 'minmax', 'standard', 'robust', and 'div-close'.
    
    
    [ output ]
    - df_result: (pd.DataFrame) Data Frames with Additional Indicators
    
    '''    
    
    
    class UserMinMaxScaler:
        def __init__(self):
            self.max_num = -np.inf
            self.min_num = np.inf

        def fit(self, arr):
            if arr is None:
                print("fit() missing 1 required positional argument: 'X'")

            self.max_num = np.min(arr)
            self.min_num = np.max(arr)

        def fit_transform(self, arr):
            if arr is None:
                print("fit_transform() missing 1 required positional argument: 'X'")

            self.max_num = np.max(arr)
            self.min_num = np.min(arr)

            return (arr - self.min_num) / (self.max_num - self.min_num)

        def transform(self, arr):
            return (arr - self.min_num) / (self.max_num - self.min_num)
        
        
    class UserStandardScaler:
        def __init__(self):
            self.mean_num = None
            self.std_num = None

        def fit(self, arr):
            if arr is None:
                print("fit() missing 1 required positional argument: 'X'")

            self.mean_num = np.mean(arr)
            self.std_num = np.std(arr)

        def fit_transform(self, arr):
            if arr is None:
                print("fit_transform() missing 1 required positional argument: 'X'")

            self.mean_num = np.mean(arr)
            self.std_num = np.std(arr)

            return (arr - self.mean_num) / self.std_num

        def transform(self, arr):
            return (arr - self.mean_num) / self.std_num
    
    
    class UserRobustScaler:
        def __init__(self):
            self.q3 = None
            self.q1 = None
            self.median_num = None

        def fit(self, arr):
            if arr is None:
                print("fit() missing 1 required positional argument: 'X'")

            self.q3 = np.percentile(arr, 75)
            self.q1 = np.percentile(arr, 25)
            self.median_num = np.median(arr)

        def fit_transform(self, arr):
            if arr is None:
                print("fit_transform() missing 1 required positional argument: 'X'")

            self.q3 = np.percentile(arr, 75)
            self.q1 = np.percentile(arr, 25)
            self.median_num = np.median(arr)

            return (arr - self.median_num) / (self.q3 - self.q1)

        def transform(self, arr):
            return (arr - self.median_num) / (self.q3 - self.q1)


    class DivCloseScaler:
        
        def __init__(self):
            self.prev_close = None
        
        def fit_transform(self, arr):
            self.prev_close = stock_df['Close'].shift(1).fillna(method="bfill")
            return arr.apply(lambda x: x /self.prev_close)
    
    
    scaler_dict = {"minmax":UserMinMaxScaler(), "standard":UserStandardScaler(), "robust":UserRobustScaler(), "div-close":DivCloseScaler()}

    
    col_temp = data.columns
    
    col_scaling = ['Open', 'High', 'Low', 'MA5', 'MA20', 'MA60', 'MA120', #'trading_value',
                   'VMAP', 'BHB', 'BLB', 'KCH', 'KCL', 'KCM', 'DCH', 'DCL', 'DCM',
                   'SMA', 'EMA', 'WMA', 'Ichimoku', 'ParabolicSAR',  'KAMA', 'MACD']
    
    column_list_scaling = list(set(col_temp) & set(col_scaling))
    
    column_list_no_scaling = list(set(col_temp) - set(col_scaling))    
    
    df_result = pd.DataFrame()
    
    for ticker, stock_df in tqdm(data.groupby('Code')):
        if scaler_type in ("minmax", "standard", "robust"):
            scaler = scaler_dict[scaler_type]
            df_scaling = pd.DataFrame(scaler.fit_transform(stock_df[column_list_scaling]), columns=column_list_scaling).reset_index(drop=True)
            df_no_scaling = stock_df[column_list_no_scaling].reset_index(drop=True)
            concat_df = pd.concat([df_scaling, df_no_scaling], axis=1)
            df_result = df_result.append(concat_df)
            
        elif scaler_type == "div-close":
            scaler = scaler_dict[scaler_type]
            df_scaling = pd.DataFrame(scaler.fit_transform(stock_df[column_list_scaling]), columns=column_list_scaling).reset_index(drop=True)
            df_no_scaling = stock_df[column_list_no_scaling].reset_index(drop=True)
            concat_df = pd.concat([df_scaling, df_no_scaling], axis=1)
            df_result = df_result.append(concat_df)
            
        else:
            print('Check scaler_type : ["minmax", "standard", "robust", "div-close"]')
            
            
    return df_result[col_temp]




def time_series(data, day=10): 
    
    
    '''
    
    [ Explanation ]
    A function that converts a daily stock price dataset into a time series datasets
    
    
    [ input ] 
    - data: (pd.DataFrame) Data to convert into time series data
    - day: (str) Set how many days to convert to time series data
    
    
    [ output ]
    - df: (pd.DataFrame) Data frames converted into time series data
    
    '''  
    
    

    df = pd.DataFrame()
    for code, df_stock in tqdm(data.groupby('Code')): 
        df_stock = df_stock.sort_values(by='Date')
        
        lst_nc =  df_stock['next_change'].values
        df_stock = df_stock.drop(columns=['next_change'])

        lst_stock = df_stock.values 

        lst_result_total = []
        for idx, (row, next_change) in enumerate(zip(lst_stock, lst_nc)): 
            if (idx < day-1) or (idx >= len(lst_stock)-1):  
                continue

            date = row[1]

            # lst_sub_stock: D-9 ~ D0 data
            lst_sub_stock = lst_stock[idx-day+1:idx+1]
            lst_sub_stock = lst_sub_stock[:, 2:] # code(0), date(1) 

            lst_result = []
            for row2 in lst_sub_stock: 
                lst_result +=  list(row2) 

            lst_result_total.append([code, date] + lst_result + [next_change])     
        
        df = df.append(pd.DataFrame(lst_result_total))
        
        
    # Creating a Column List
    lst_cols = []
    for i in range(day-1, -1, -1):
        for col in df_stock.columns[2:]: 
            if i == 0: 
                lst_cols.append(f'D{i}_{col}')
            else:
                lst_cols.append(f'D-{i}_{col}')
                
    df.columns = ['Code', 'Date'] + lst_cols + ['next_change']           

    return df