import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm 
from xgboost import XGBClassifier
from sklearn.preprocessing import Binarizer
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from .trader import *



def save_dataset(lst_trader, train_data, test_data, scaled_train_data=None, scaled_test_data=None):
    for trader in lst_trader:
        print(f'== {trader.name} ==')
        
        trader.train_code_date = train_data[['Code', 'Date']].values
        trader.test_code_date = test_data[['Code', 'Date']].values
        print(f"== train_code_date: {trader.train_code_date.shape},  test_code_date: {trader.test_code_date.shape} ==")
                

        trader.trainX = train_data.drop(columns=["Code", "Date", "next_change"]).reset_index(drop=True)
        trader.testX = test_data.drop(columns=["Code", "Date", "next_change"]).reset_index(drop=True)

        print(f"== trainX: {trader.trainX.shape},  testX: {trader.testX.shape} ==")
        
        
        if type(scaled_train_data) != 'NoneType': 
            
            trader.trainX_scaled = scaled_train_data.drop(columns=["Code", "Date", "next_change"]).reset_index(drop=True)
            trader.testX_scaled = scaled_test_data.drop(columns=["Code", "Date", "next_change"]).reset_index(drop=True)
            
            print(f"== trainX_scaled: {trader.trainX_scaled.shape},  testX_scaled: {trader.testX_scaled.shape} ==")

            
        trader.trainY = train_data['next_change'].reset_index(drop=True)
        trader.testY = test_data['next_change'].reset_index(drop=True)
        print(f"== trainY: {trader.trainY.shape},  testY: {trader.testY.shape} ==")
        
        if 'class' in trader.label:
            threshold = float(trader.label.split('&')[1])
            trader.train_classification = (trader.trainY >= threshold).astype('int')
            trader.test_classification = (trader.testY >= threshold).astype('int')
            print(f"== trainY_classification: {trader.train_classification.shape},  testY_classification: {trader.test_classification.shape} ==")
            
        print()

        
        
def trader_train(lst_trader):
    for trader in lst_trader:
        b1 = trader.buyer.sub_buyers[0]
        b2 = trader.buyer.sub_buyers[1]
        
        
        if type(trader.trainX_scaled) != "NoneType":
            if b2.data_transform != None:          
                trainX = b2.data_transform(trader.trainX_scaled.loc[b1.decision(trader.trainX)].values.tolist())
            else:     
                trainX = trader.trainX_scaled.loc[b1.decision(trader.trainX)]
        else:  
            if b2.data_transform != None:           
                trainX = b2.data_transform(trader.trainX.loc[b1.decision(trader.trainX)].values.tolist())
            else:     
                trainX = trader.trainX.loc[b1.decision(trader.trainX)]
        
        
        if type(trader.train_classification) != 'NoneType':
             trainY = trader.train_classification.loc[b1.decision(trader.trainX)]            
        else: 
             trainY = trader.trainY.loc[b1.decision(trader.trainX)]
            
    
        trader.buyer.train(trainX, trainY)
            
        print(f"== {trader.name} Model Fitting Completed ==")
        

        
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
            
            ax2.scatter(i, precision_score(trader.test_classification.loc[b1.decision(trader.testX)], pred), color='#daa933', label='Precision') 
            ax2.scatter(i, recall_score(trader.test_classification.loc[b1.decision(trader.testX)], pred), color='#37babc', label ='Recall')  
            ax2.scatter(i, f1_score(trader.test_classification.loc[b1.decision(trader.testX)], pred), color='#b4d125', label='f1 score') 
            if i == 0.1:
                ax2.legend(fontsize = 10)

            ax2.axvline(0.2, color = '#c97878', linestyle='--')
            ax2.axvline(0.4, color = '#c97878', linestyle='--')
            ax2.axvline(0.6, color = '#c97878', linestyle='--')
            ax2.axvline(0.8, color = '#c97878', linestyle='--')
        
            ax2.set_title('Precision, Recall, f1 score',fontsize=15)
            ax2.set_ylabel("score", fontsize=13)
            ax2.set_xlabel("Threshhold", fontsize=13)
            
            
            
            

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
            if type(buyer) == ConditionalBuyer:
                b1 = buyer
            elif type(buyer) == MachinelearningBuyer: 
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
            sns.distplot(upper_threshold, color= 'r', label="Distribution of the next_change of values predicted by more than {}% | Mean: {}".format(trader.buyer.sub_buyers[1].threshold * 100, round(np.mean(upper_threshold), 3)))
            sns.distplot(lower_threshold, label='Distribution of the next_change of values predicted by less than {}% | Mean: {}'.format(trader.buyer.sub_buyers[1].threshold * 100, round(np.mean(lower_threshold), 3)))
            plt.axvline(np.mean(upper_threshold),color='r')
            plt.axvline(np.mean(lower_threshold),color='b')
            plt.legend(fontsize=15)
            plt.title(f'{trader.name}',fontsize=20, pad=20)
            plt.ylabel('Next Change', fontsize=15, labelpad=25)


