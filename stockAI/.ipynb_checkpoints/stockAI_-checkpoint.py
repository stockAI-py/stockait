import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import ta
from tqdm import tqdm 
import yfinance as yf
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import pickle 
from sklearn.preprocessing import Binarizer
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

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



# 보조지표 추가
def add_index(data, index_list:list):
    
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
                    
                    'trading_value':stock_df['Close']*stock_df['Volume'],
                    'next_change':stock_df['Change'].shift(-1),
                    
                    'MFI':ta.volume.money_flow_index(high=H, low=L, close=C, volume=V, fillna=True),
                    'ADI':ta.volume.acc_dist_index(high=H, low=L, close=C, volume=V, fillna=True),
                    'OBV':ta.volume.on_balance_volume(close=C, volume=V, fillna=True),
                    'CMF':ta.volume.chaikin_money_flow(high=H, low=L, close=C, volume=V, fillna=True),
                    'FI':ta.volume.force_index(close=C, volume=V, fillna=True),
                    'EOM_EMV':ta.volume.ease_of_movement(high=H, low=L, volume=V, fillna=True),
                    'VPT':ta.volume.volume_price_trend(close=C, volume=V, fillna=True),
                    'NVI':ta.volume.negative_volume_index(close=C, volume=V, fillna=True),
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
                    
                    'SMA':ta.trend.sma_indicator(close=C, fillna=True),
                    'EMA':ta.trend.ema_indicator(close=C, fillna=True),
                    'WMA':ta.trend.wma_indicator(close=C, fillna=True),
                    'MACD':ta.trend.macd(close=C, fillna=True),
#                     'ADX':ta.trend.adx(high=H, low=L, close=C, fillna=True),
                    'VIneg':ta.trend.vortex_indicator_neg(high=H, low=L, close=C, fillna=True),
                    'VIpos':ta.trend.vortex_indicator_pos(high=H, low=L, close=C, fillna=True),
                    'TRIX':ta.trend.trix(close=C, fillna=True),
                    'MI':ta.trend.mass_index(high=H, low=L, fillna=True),
                    'CCI':ta.trend.cci(high=H, low=L, close=C, fillna=True),
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
                    'KAMA':ta.momentum.kama(close=C, fillna=True),
                    'ROC':ta.momentum.roc(close=C, fillna=True),
                    'PPO':ta.momentum.ppo(close=C, fillna=True),
                    'PVO':ta.momentum.pvo(volume=V, fillna=True)
                   }
        
        for index in index_list:
            stock_df[index] = add_form[index]
            
        result = result.append(stock_df)
        result = result.dropna()
    
    return result

# 스케일링
def scaling(data, scaler_type, window_size=None):
    
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


# 시계열 데이터 변환
def time_series(data, day=10): 
    from tqdm import tqdm

    df = pd.DataFrame()
    for code, df_stock in tqdm(data.groupby('Code')): 
        df_stock = df_stock.sort_values(by='Date')
        
        lst_nc =  df_stock['next_change'].values
        df_stock = df_stock.drop(columns=['next_change'])

        lst_stock = df_stock.values 

        lst_result_total = []
        for idx, (row, next_change) in enumerate(zip(lst_stock, lst_nc)): 
            if (idx < day-1) or (idx >= len(lst_stock)-1): # 예외 처리 
                continue

            date = row[1]

            # lst_sub_stock: D-9 ~ D0 데이터  
            lst_sub_stock = lst_stock[idx-day+1:idx+1]
            lst_sub_stock = lst_sub_stock[:, 2:] # code(0), date(1) 제거 

            lst_result = []
            for row2 in lst_sub_stock: 
                lst_result +=  list(row2) 

            lst_result_total.append([code, date] + lst_result + [next_change])     
        
        df = df.append(pd.DataFrame(lst_result_total))
        
    return df 
        
# 컬럼 리스트 생성 
    lst_cols = []
    for i in range(day-1, -1, -1):
        for col in df_stock.columns[2:]: 
            if i == 0: 
                lst_cols.append(f'D{i}_{col}')
            else:
                lst_cols.append(f'D-{i}_{col}')
                
    df.columns = ['Code', 'Date'] + lst_cols + ['next_change', 'Change']           

    return df

# Trader
class Trader: 
    def __init__(self):
        self.name = None
        
        # condition
        self.label = None ## (추가) 
        self.scaling = None ## (추가)
        
        # dataset
        self.train_code_date = None
        self.test_code_date = None
        
        self.trainX = None ## (추가)
        self.testX = None ## (추가)        
        self.trainX_scaled = None ## (추가)
        self.testX_scaled = None ## (추가)
        
        self.trainY = None ## (추가)
        self.testY = None ## (추가)        
        self.train_classification = None ## (추가)
        self.test_classification = None ## (추가)
        
        # trader
        self.buyer = None
        self.seller = None
        
# Buyer
class Buyer: 
    def __init__(self, sub_buyers):
        self.sub_buyers = sub_buyers
        pass
    
    def decision_all(self, trader_object, dtype='test', data=None, data_scaled=None): # threshold를 넘는 에측확률 + 변동성 조건 만족하는 data (최종 매수 데이터 결정)
        # 매수 시그널 리스트 
        total_amount = 1.0
        
        # change, close, trading_value 
        # conditional_buyer: df 
        # machinelearning_buyer: trader_object.testX_scaled 
        
        for sub in self.sub_buyers: # [b1, b2]
            if type(sub) == conditional_buyer: 
                if dtype == 'test': 
                    df = trader_object.testX
                
                amount = sub.decision(df) # 원본 데이터 
                
            elif type(sub) == machinelearning_buyer:
                
                if dtype == 'test':
                    if type(trader_object.testX_scaled) != 'NoneType':
                        df_machine = trader_object.testX_scaled
                    
                    else:    
                        df_machine = trader_object.testX

                amount = sub.decision(df_machine) 
                    
            total_amount *= amount
        
        total_amount = total_amount.tolist() # [0, 1, 0, 0, 1, ...]
        
        if dtype=='test':
            lst_code_date = trader_object.test_code_date

        
        # 매수 일지 작성 
        lst_buy_signal = []
        for i, row  in tqdm(df.iterrows()):
            amount = total_amount[i]
            
            lst_buy_signal.append([trader_object.name, lst_code_date[i][1], lst_code_date[i][0], '+', amount, row['D0_Close']])
        
        return pd.DataFrame(lst_buy_signal, columns = ['Trader_id', 'Date', 'Code', '+(buy)/-(sell)', 'Amount', 'Close']) 
    

    def train(self, X, y):  # 모델 학습 
        for sub in self.sub_buyers:
            if type(sub) == machinelearning_buyer:
                sub.train(X, y)
        

        
class conditional_buyer:
    def __init__(self):
        self.condition = None

    def decision(self, df):
        return self.condition(df)
    
    
class machinelearning_buyer:
    def __init__(self):
        self.algorithm = None
        self.threshold = None
        self.data_transform = None
        
    def train(self, X, y):
        self.algorithm.fit(X, y)
    
    def decision(self, df):
        lst_drop = ['Code', 'Date', 'next_change']
        for col in lst_drop:
            if col in df.columns:
                df = df.drop(columns=[col])
        if self.data_transform != None:  
            amount = (self.algorithm.predict(self.data_transform(df.values.tolist())) >= self.threshold).astype('int').reshape(1, -1)[0]
        else:     
            amount = (self.algorithm.predict_proba(df)[:, 1] >= self.threshold).astype('int')
        
        return amount 
    
# Seller
class Seller:
    def __init__(self, sub_seller):
        self.sub_seller = sub_seller
        
    def decision_all(self, trader_object, dtype='test', data=None, data_scaled=None):
        if dtype == 'test': 
            df = trader_object.testX
        
        # 매수 시그널 리스트 
        sub = self.sub_seller
        total_amount = 1
        total_amount *= sub.decision_next_day_sell(df)
        
        if dtype=='test':
            lst_code_date = trader_object.test_code_date
        
        # 매도 일지 작성 
        lst_sell_signal = []
        for i, row  in tqdm(df.iterrows()):
            amount = total_amount[i]
            
            if amount <= 0: 
                continue 
            
            lst_sell_signal.append([trader_object.name, lst_code_date[i][1], lst_code_date[i][0], '-', amount, row['D0_Close']])
        
        return pd.DataFrame(lst_sell_signal, columns = ['Trader_id', 'Date', 'Code', '+(buy)/-(sell)', 'Amount', 'Close']) 
   
      
class SubSeller:
    def __init__(self):
        pass

    def decision_next_day_sell(self, data):
        lst_amount = [1] * len(data)
        return lst_amount 

# 필요 데이터셋 생성
def save_dataset(lst_trader, train_data, test_data, scaled_train_data=None, scaled_test_data=None):
    for trader in lst_trader:
        print(f'== {trader.name} ==')
        
        trader.train_code_date = train_data[['Code', 'Date']].values
        trader.test_code_date = test_data[['Code', 'Date']].values
        print(f"== train_code_date: {trader.train_code_date.shape},  test_code_date: {trader.test_code_date.shape} 생성 ==")
                

        trader.trainX = train_data.drop(columns=["Code", "Date", "next_change"]).reset_index(drop=True)
        trader.testX = test_data.drop(columns=["Code", "Date", "next_change"]).reset_index(drop=True)

        print(f"== trainX: {trader.trainX.shape},  testX: {trader.testX.shape} 생성 ==")
        
        
        if type(scaled_train_data) != 'NoneType': 
            
            trader.trainX_scaled = scaled_train_data.drop(columns=["Code", "Date", "next_change"]).reset_index(drop=True)
            trader.testX_scaled = scaled_test_data.drop(columns=["Code", "Date", "next_change"]).reset_index(drop=True)
            
            print(f"== trainX_scaled: {trader.trainX_scaled.shape},  testX_scaled: {trader.testX_scaled.shape} 생성 ==")

            
        trader.trainY = train_data['next_change'].reset_index(drop=True)
        trader.testY = test_data['next_change'].reset_index(drop=True)
        print(f"== trainY: {trader.trainY.shape},  testY: {trader.testY.shape} 생성 ==")
        
        if 'class' in trader.label:
            threshold = float(trader.label.split('&')[1])
            trader.train_classification = (trader.trainY >= threshold).astype('int')
            trader.test_classification = (trader.testY >= threshold).astype('int')
            print(f"== trainY_classification: {trader.train_classification.shape},  testY_classification: {trader.test_classification.shape} 생성 ==")
            
        print()
        
# 모델 학습
def trader_train(lst_trader):
    for trader in lst_trader:
        b1 = trader.buyer.sub_buyers[0]
        b2 = trader.buyer.sub_buyers[1]
        
        
        if type(trader.trainX_scaled) != 'NoneType':
            if b2.data_transform != None:          
                trainX = b2.data_transform(trader.trainX_scaled.loc[b1.decision(trader.trainX)].values.tolist())
            else:     
                trainX = trader.trainX_scaled.loc[b1.decision(trader.trainX)]
        else:  
            if type(b2.data_transform) != 'NoneType':          
                trainX = b2.data_transform(trader.trainX.loc[b1.decision(trader.trainX)].values.tolist())
            else:     
                trainX = trader.trainX.loc[b1.decision(trader.trainX)]
        
        
        if type(trader.train_classification) != 'NoneType':
             trainY = trader.train_classification.loc[b1.decision(trader.trainX)]            
        else: 
             trainY = trader.trainY.loc[b1.decision(trader.trainX)]
            
    
        trader.buyer.train(trainX, trainY)
            
        print(f"== {trader.name} 모델학습 완료 ==")
        
# 모델 평가 및 threshold 설정
def get_eval_by_threshold(lst_trader):

    from sklearn.preprocessing import Binarizer
    import matplotlib.pyplot as plt
    from sklearn.metrics import f1_score, roc_curve, roc_auc_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
    import matplotlib
    import numpy as np

    matplotlib.style.use("seaborn-whitegrid")
    matplotlib.rcParams['font.family'] ='NanumSquareRound'

    
    
    thresholds = list(np.arange(0.1, 1, 0.01))
    
    
    for trader in lst_trader:
        
        b1 = trader.buyer.sub_buyers[0]
        b2 = trader.buyer.sub_buyers[1]
        
        fig = plt.figure(figsize=(12, 5))
        ax1, ax2 = fig.subplots(1, 2)

        fig.suptitle(trader.name, fontsize=20, position = (0.5, 1.0+0.03))
        
        if type(trader.testX_scaled) != 'NoneType':
            testX_filtered = trader.testX_scaled.loc[b1.decision(trader.testX)]
        else:
            testX_filtered = trader.testX.loc[b1.decision(trader.testX)]
        
        if b2.data_transform != None: 
            testX_filtered_2d = b2.data_transform(testX_filtered.values.tolist())
            pred_proba = b2.algorithm.predict(testX_filtered_2d)
        else:
            pred_proba = b2.algorithm.predict_proba(testX_filtered)[:, 1].reshape(-1,1)
        
        
        fpr, tpr, _ = roc_curve(trader.test_classification.loc[b1.decision(trader.testX)],  pred_proba)
        auc = roc_auc_score(trader.test_classification.loc[b1.decision(trader.testX)], pred_proba)

        ax1.set_title(f'auc score: {round(auc, 3)}',fontsize=15)
        ax1.plot(fpr,tpr,label="AUC="+str(auc))
        ax1.plot([0, 1], [0, 1], color='green', linestyle='--')
        ax1.set_ylabel('True Positive Rate', fontsize=13)
        ax1.set_xlabel('False Positive Rate', fontsize=13)
        ax1.legend(loc=4)


        for i in thresholds:
            binarizer = Binarizer(threshold = i).fit(pred_proba)
            pred = binarizer.transform(pred_proba)
            
            ax2.scatter(i, precision_score(trader.test_classification.loc[b1.decision(trader.testX)], pred), color='#daa933', label='정밀도') # 정밀도
            ax2.scatter(i, recall_score(trader.test_classification.loc[b1.decision(trader.testX)], pred), color='#37babc', label ='재현율') # 재현율 
            ax2.scatter(i, f1_score(trader.test_classification.loc[b1.decision(trader.testX)], pred), color='#b4d125', label='f1 score') # f1 score
            if i == 0.1:
                ax2.legend(fontsize = 10)

            ax2.axvline(0.2, color = '#c97878', linestyle='--')
            ax2.axvline(0.4, color = '#c97878', linestyle='--')
            ax2.axvline(0.6, color = '#c97878', linestyle='--')
            ax2.axvline(0.8, color = '#c97878', linestyle='--')
        
            ax2.set_title('Precision, Recall, f1 score',fontsize=15)
            ax2.set_ylabel("score", fontsize=13)
            ax2.set_xlabel("Threshhold", fontsize=13)
            
            
            
            
# 수익성 검증 histogram
def set_threshold(lst_trader, lst_threshold:list, histogram:bool=True):
    from sklearn.preprocessing import Binarizer
    import matplotlib.pyplot as plt
    from sklearn.metrics import f1_score, roc_curve, roc_auc_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
    import matplotlib
    import numpy as np
    import seaborn as sns

    matplotlib.style.use("seaborn-whitegrid")
    matplotlib.rcParams['font.family'] ='NanumSquareRound'
    
    try: 
        if len(lst_trader) != len(threshold): 
            raise(Exception("The length of the list is different."))
            print(f"lst_trader: {lst_trader}, threshold: {threshold}")
    except Exception as e: 
        print('Error:', e)
        
    for trader, threshold in zip(lst_trader, lst_threshold):
        fig = plt.figure(figsize=(16,10))
        
        for buyer in trader.buyer.sub_buyers:
            if type(buyer) == sai.conditional_buyer:
                b1 = buyer
            elif type(buyer) == sai.machinelearning_buyer: 
                b2 = buyer
        
        b2.threshold = threshold 
        
        if histogram: 
            fig = plt.figure(figsize=(16,10))
            
            testX_filtered = trader.testX_scaled.loc[b1.decision(trader.testX)]
            testY_filtered = trader.testY.loc[b1.decision(trader.testX)]

            if b2.data_transform != None:     
                testX_filtered = b2.data_transform(testX_filtered.values.tolist())
                pred_proba = b2.algorithm.predict(testX_filtered).reshape(-1,1)
        
            else:
                pred_proba = b2.algorithm.predict_proba(testX_filtered)[:, 1].reshape(-1,1)
                
              
            upper_threshold=[]
            lower_threshold=[]

            for prod, next_change in zip(pred_proba, 100*(testY_filtered)):
                if prod >= trader.buyer.sub_buyers[1].threshold:
                    upper_threshold.append(next_change)
                else:
                    lower_threshold.append(next_change)

            fig = plt.figure(figsize=(16,10))
            sns.distplot(upper_threshold, color= 'r', label='{}% 이상 예측한 값들의 다음날 종가 변화율 분포 | 평균: {}'.format(trader.buyer.sub_buyers[1].threshold * 100, round(np.mean(upper_threshold), 3)))
            sns.distplot(lower_threshold, label='{}% 미만 예측한 값들의 다음날 종가 변화율 분포 | 평균: {}'.format(trader.buyer.sub_buyers[1].threshold * 100, round(np.mean(lower_threshold), 3)))
            plt.axvline(np.mean(upper_threshold),color='r')
            plt.axvline(np.mean(lower_threshold),color='b')
            plt.legend(fontsize=15)
            plt.title(f'{trader.name}',fontsize=20, pad=20)
            plt.ylabel('Next Change', fontsize=15, labelpad=25)

            # plt.xlim([-40, 40])



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
        
        print(f'== {trader.name} 매매일지 작성 완료 ==')
    
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
        print(f"== {trader_id} 매매 완료 ==")

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