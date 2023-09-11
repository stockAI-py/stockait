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


def save_dataset(lst_trader, train_data, valid_data, test_data, scaled_train_data=None, scaled_valid_data=None, scaled_test_data=None):
    for trader in lst_trader:
        print(f'== {trader.name} ==')
        
        trader.train_code_date = train_data[['Code', 'Date']].values
        trader.valid_code_date = valid_data[['Code', 'Date']].values
        trader.test_code_date = test_data[['Code', 'Date']].values
        print(f"== train_code_date: {trader.train_code_date.shape}, valid_code_data: {trader.valid_code_date.shape}, test_code_date: {trader.test_code_date.shape} ==")
                

        trader.trainX = train_data.drop(columns=["Code", "Date", "next_change"]).reset_index(drop=True)
        trader.validX = valid_data.drop(columns=["Code", "Date", "next_change"]).reset_index(drop=True)
        trader.testX = test_data.drop(columns=["Code", "Date", "next_change"]).reset_index(drop=True)

        print(f"== trainX: {trader.trainX.shape}, validX: {trader.validX.shape}, testX: {trader.testX.shape} ==")
        
        
        if type(scaled_train_data) != type(None):
            if trader.scaled_data:
                trader.trainX_scaled = scaled_train_data.drop(columns=["Code", "Date", "next_change"]).reset_index(drop=True)
                trader.validX_scaled = scaled_valid_data.drop(columns=["Code", "Date", "next_change"]).reset_index(drop=True)
                trader.testX_scaled = scaled_test_data.drop(columns=["Code", "Date", "next_change"]).reset_index(drop=True)

                print(f"== trainX_scaled: {trader.trainX_scaled.shape}, validX_scaled: {trader.validX_scaled.shape}, testX_scaled: {trader.testX_scaled.shape} ==")

            
        trader.trainY = train_data['next_change'].reset_index(drop=True)
        trader.validY = valid_data['next_change'].reset_index(drop=True)
        trader.testY = test_data['next_change'].reset_index(drop=True)
        print(f"== trainY: {trader.trainY.shape}, validY: {trader.validY.shape}, testY: {trader.testY.shape} ==")
        
        if 'class' in trader.label:
            threshold = float(trader.label.split('&')[1])
            trader.train_classification = (trader.trainY >= threshold).astype('int')
            trader.valid_classification = (trader.validY >= threshold).astype('int') 
            trader.test_classification = (trader.testY >= threshold).astype('int')
            print(f"== trainY_classification: {trader.train_classification.shape}, validY_classification: {trader.valid_classification.shape}, testY_classification: {trader.test_classification.shape} ==")
            
        print()

        
        
def trader_train(lst_trader, history=False):
    for trader in lst_trader:
        for sub in trader.buyer.sub_buyers: # [b1, b2]
            if type(sub) == ConditionalBuyer: 
                b1 = sub 
            if type(sub) == MachinelearningBuyer: 
                b2 = sub 
        
        validX, validY = None, None  
        
        if type(trader.trainX_scaled) != type(None):
            if b2.data_transform:          
                trainX = b2.data_transform(trader.trainX_scaled.loc[b1.decision(trader.trainX)].values.tolist())
                
                if history: 
                    validX = b2.data_transform(trader.validX_scaled.loc[b1.decision(trader.validX)].values.tolist())
                    
            else:     
                trainX = trader.trainX_scaled.loc[b1.decision(trader.trainX)]
        else:  
            if b2.data_transform:           
                trainX = b2.data_transform(trader.trainX.loc[b1.decision(trader.trainX)].values.tolist())
                
                if history: 
                    validX = b2.data_transform(trader.validX.loc[b1.decision(trader.validX)].values.tolist())
            
            else:     
                trainX = trader.trainX.loc[b1.decision(trader.trainX)]
        
        
        if type(trader.train_classification) != type(None):
            trainY = trader.train_classification.loc[b1.decision(trader.trainX)]     
            
            if (type(b2.data_transform) != type(None)) & history: 
                validY = trader.valid_classification.loc[b1.decision(trader.validX)]
            
        else: 
            trainY = trader.trainY.loc[b1.decision(trader.trainX)]
            
            if (b2.data_transform) & history: 
                validY = trader.validY.loc[b1.decision(trader.validX)]
            
    
        trader.buyer.train(trainX, trainY, validX, validY, history)
            
        print(f"== {trader.name} Model Fitting Completed ==")
        

        
def get_history_learning_curve(lst_trader): 
    dic_history = {}
    for trader in lst_trader: 
        for sub in trader.buyer.sub_buyers: # [b1, b2]
            if type(sub) == MachinelearningBuyer: 
                b2 = sub         
        
        if (b2.framework == "tensorflow") or (b2.framework == "pytorch"): 
            history = b2.history.history if b2.framework == "tensorflow" else b2.history 
            dic_history[trader.name] = history 
            
            acc = history["accuracy"]
            val_acc = history["val_accuracy"]
   
            loss = history["loss"]
            val_loss = history["val_loss"]
        
            plt.figure(figsize=(15, 7))
            plt.subplot(1, 2, 1)
            plt.title(f"{trader.name} accuracy per epoch")
            plt.plot(acc, label='Training Accuracy')
            plt.plot(val_acc, label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.ylabel('Accuracy')
            # plt.ylim([min(plt.ylim()),1])
            plt.title('Training and Validation Accuracy')

            plt.subplot(1, 2, 2)
            plt.title(f"{trader.name} loss per epoch")
            plt.plot(loss, label='Training Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.ylabel('Cross Entropy')
            plt.title('Training and Validation Loss')
            plt.xlabel('epoch')
            plt.show()        
        
    return dic_history 
        
        
        
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
        
        if type(trader.validX_scaled) != type(None):
            validX_filtered = trader.validX_scaled.loc[b1.decision(trader.validX)]
        else:
            validX_filtered = trader.validX.loc[b1.decision(trader.validX)]
        
        if b2.data_transform: 
            validX_filtered_2d = b2.data_transform(validX_filtered.values.tolist())
            
            if b2.framework == "tensorflow":
                pred_proba = b2.algorithm.predict(validX_filtered_2d)
            
            elif b2.framework == "pytorch": 
                validX_filtered_2d = torch.tensor(validX_filtered_2d, dtype=torch.float32)
                y = torch.randint(0, 2, (len(validX_filtered_2d), ), dtype=torch.float32) # temp y for dataloader  
                batch_size = b2.params["batch_size"]
                device = b2.device 

                dataset = TensorDataset(validX_filtered_2d, y)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                # model 
                model = b2.algorithm
                model.to(device) 
                
                # evaulation mode 
                model.eval() 

                pred_proba = np.array([])
                with torch.no_grad(): 
                    for batch_X, batch_y in dataloader: 
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device) 

                        outputs = model(batch_X).cpu()  
                        pred_proba_batch = np.array(outputs.reshape(1, len(batch_X))[0])
                        pred_proba = np.concatenate((pred_proba, pred_proba_batch))
                del y 
                pred_proba = pred_proba.reshape(-1, 1)  
            
        else:
            pred_proba = b2.algorithm.predict_proba(validX_filtered)[:, 1].reshape(-1, 1)
        
        
        fpr, tpr, _ = roc_curve(trader.valid_classification.loc[b1.decision(trader.validX)],  pred_proba)
        auc = roc_auc_score(trader.valid_classification.loc[b1.decision(trader.validX)], pred_proba)

        ax1.set_title(f'auc score: {round(auc, 3)}',fontsize=15)
        ax1.plot(fpr,tpr,label="AUC="+str(auc), linewidth=1, color='#f1404b')
        ax1.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax1.set_ylabel('True Positive Rate', fontsize=10)
        ax1.set_xlabel('False Positive Rate', fontsize=10)
        ax1.legend(loc=4)


        for i in thresholds:
            binarizer = Binarizer(threshold = i).fit(pred_proba)
            pred = binarizer.transform(pred_proba)
            
            ax2.scatter(i, precision_score(trader.valid_classification.loc[b1.decision(trader.validX)], pred), color='#f1404b', label='Precision', s=10) 
            ax2.scatter(i, recall_score(trader.valid_classification.loc[b1.decision(trader.validX)], pred), color='gray', label ='Recall', s=10)  
            ax2.scatter(i, f1_score(trader.valid_classification.loc[b1.decision(trader.validX)], pred), color='#a7a7a2', label='f1 score', s=10) 
            if i == 0.1:
                ax2.legend(fontsize = 10)

            ax2.axvline(0.2, color = 'lightgray', linestyle='--')
            ax2.axvline(0.4, color = 'lightgray', linestyle='--')
            ax2.axvline(0.6, color = 'lightgray', linestyle='--')
            ax2.axvline(0.8, color = 'lightgray', linestyle='--')
        
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
        if len(lst_trader) != len(lst_threshold): 
            raise(Exception("The length of the list is different."))
            print(f"lst_trader: {lst_trader}, lst_threshold: {lst_threshold}")
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
            
            if type(trader.validX_scaled) != type(None):
                validX_filtered = trader.validX_scaled.loc[b1.decision(trader.validX)]
            else:
                validX_filtered = trader.validX.loc[b1.decision(trader.validX)]
                
            validY_filtered = trader.validY.loc[b1.decision(trader.validX)]

            if b2.data_transform:     
                validX_filtered_2d = b2.data_transform(validX_filtered.values.tolist())
                
                if b2.framework == "tensorflow": 
                    pred_proba = b2.algorithm.predict(validX_filtered_2d).reshape(-1, 1)
                
                elif b2.framework == "pytorch": 
                    validX_filtered_2d = torch.tensor(validX_filtered_2d, dtype=torch.float32)
                    y = torch.randint(0, 2, (len(validX_filtered_2d), ), dtype=torch.float32) # temp y for dataloader  
                    batch_size = b2.params["batch_size"]
                    device = b2.device 

                    dataset = TensorDataset(validX_filtered_2d, y)
                    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                    # model 
                    model = b2.algorithm
                    model.to(device) 
                    
                    # evaulation mode 
                    model.eval() 
                    pred_proba = np.array([])
                    with torch.no_grad(): 
                        for batch_X, batch_y in dataloader: 
                            batch_X, batch_y = batch_X.to(device), batch_y.to(device) 

                            outputs = model(batch_X).cpu()  
                            pred_proba_batch = np.array(outputs.reshape(1, len(batch_X))[0])
                            pred_proba = np.concatenate((pred_proba, pred_proba_batch))
                    del y 
                    
                    pred_proba = pred_proba.reshape(-1, 1)  
                
            else:
                pred_proba = b2.algorithm.predict_proba(validX_filtered)[:, 1].reshape(-1, 1)
                
              
            upper_threshold=[]
            lower_threshold=[]

            for prod, next_change in zip(pred_proba, 100*(validY_filtered)):
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


