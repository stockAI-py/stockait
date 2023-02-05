import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import ta
from tqdm import tqdm 
from sklearn.metrics import f1_score, roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix



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