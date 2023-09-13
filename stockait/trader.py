import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import ta
from tqdm import tqdm 
from sklearn.metrics import f1_score, roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import torch 



class Trader: 
    def __init__(self):
        self.name = None
        
        # condition
        self.label = None 
        
        # dataset
        self.train_code_date = None
        self.valid_code_date = None
        self.test_code_date = None
        
        self.trainX_loader = None
        self.validX_loader = None 
        self.testX_loader = None       
        self.trainX_scaled = None
        self.validX_scaled = None 
        self.testX_scaled = None
        
        self.trainY = None 
        self.validY = None 
        self.testY = None        
        self.train_classification = None
        self.valid_classification = None 
        self.test_classification = None 
        
        self.scaled_data = True
        
        # columns
        self.columns = None
        
        # trader
        self.buyer = None
        self.seller = None
        


class Buyer: 
    def __init__(self, sub_buyers):
        self.sub_buyers = sub_buyers
        pass
    
    def decision_all(self, trader_object, dtype='test', data=None, data_scaled=None): 
        
        # Stock Purchase Signal List
        total_amount = 1.0
        
        # change, close, trading_value 
        # ConditionalBuyer: df 
        # machinelearning_buyer: trader_object.testX_scaled 
        
        for sub in self.sub_buyers: # [b1, b2]
            if type(sub) == ConditionalBuyer:

                if dtype == 'test': 
                    df = trader_object.testX
                
                amount = sub.decision(df) # 원본 데이터 
                    
            elif type(sub) == MachinelearningBuyer:
                
                if dtype == 'test':
                    if type(trader_object.testX_scaled) != type(None):
                        df_machine = trader_object.testX_scaled
                    
                    else:    
                        df_machine = trader_object.testX
                try:
                    amount = sub.decision(df_machine) 
                    
                except ValueError as e: 
                    print("An Error Occured!:", e)                    
                    
            total_amount *= amount
        
        total_amount = total_amount.tolist() # [0, 1, 0, 0, 1, ...]
        
        if dtype=='test':
            lst_code_date = trader_object.test_code_date

        
        # Write a purchase diary 
        lst_buy_signal = []
        for i, row  in tqdm(df.iterrows()):
            amount = total_amount[i]
            
            lst_buy_signal.append([trader_object.name, lst_code_date[i][1], lst_code_date[i][0], '+', amount, row['D0_Close']])
        
        return pd.DataFrame(lst_buy_signal, columns = ['Trader_id', 'Date', 'Code', '+(buy)/-(sell)', 'Amount', 'Close']) 
    

    def train(self, X, y, validX, validY, history):  # model fitting 
        for sub in self.sub_buyers:
            if type(sub) == MachinelearningBuyer:
                try:
                    sub.train(X, y, validX, validY, history)
                except ValueError as e: 
                    print("An Error Occured!:", e)

        
class ConditionalBuyer:
    def __init__(self):
        self.condition = None

    def decision(self, df):
        return self.condition(df)
    
    
class MachinelearningBuyer:
    def __init__(self):
        self.algorithm = None
        self.threshold = None
        self.data_transform = None
        self.framework = None 
        self.params = None 
        self.optim = None 
        self.device = None 
        self.history = None 
        
    def train(self, X, y, validX, validY, history):
        if self.framework == "tensorflow": 
            self.train_tensorflow(X, y, validX, validY, history) 
        
        elif self.framework == "pytorch": 
            self.train_pytorch(X, y, validX, validY, history)
        
        elif self.framework == None: 
            self.algorithm.fit(X, y)
    
        else: 
            raise ValueError('The framework is not valid. Only "tensorflow", "pytorch", and None can enter.') 
    
    
    def train_tensorflow(self, X, y, validX, validY, history): 
        if history: 
            self.params["validation_data"] = (validX, validY)
        history = self.algorithm.fit(X, y, **self.params)
        self.history = history
    
    def train_pytorch(self, X, y, validX, validY, history): 
        # hyper parameter 
        epochs, batch_size = self.params["epochs"], self.params["batch_size"]
        
        # loss function & optimizer 
        criterion, optimizer = self.optim["criterion"], self.optim["optimizer"]
        
        # device 
        device = self.device 
        
        # train datset         
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y.to_numpy(), dtype=torch.float32)  
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
              
        # model 
        model = self.algorithm
        model.to(device) 
            
        if history: 
            # valid dataset 
            validX, validY = torch.tensor(validX, dtype=torch.float32), torch.tensor(validY.to_numpy(), dtype=torch.float32)
            dataset_valid = TensorDataset(validX, validY) 
            dataloader_valid = DataLoader(dataset, batch_size=batch_size, shuffle=True) 

            # history
            dic_history = {"accuracy": [], "val_accuracy": [],
                           "loss": [], "val_loss": []}            
            
            # model fitting 
            for epoch in range(epochs): 
                # train mode 
                model.train()      
                with tqdm(dataloader, leave=False) as pbar:          
                    batch_acc, batch_loss, batch_train = 0, 0, 0  
                    pbar.set_description(f"Epoch - {epoch+1}")
                    for batch_X, batch_y in pbar: 
                        acc, loss = self.train_batch(batch_X, batch_y, device, criterion, optimizer)
                        batch_acc += acc
                        batch_loss += loss
                        batch_train += 1 
                        pbar.set_postfix({"loss": batch_loss/batch_train, "acc": batch_acc/batch_train})


                # evaluation mode 
                model.eval()
                with torch.no_grad(): 
                    with tqdm(dataloader_valid, leave=False) as pbar_valid:
                        batch_loss_val, batch_acc_val, batch_valid = 0, 0, 0   
                        pbar_valid.set_description(f"Epoch - {epoch+1}")
                        for batch_validX, batch_validY in pbar_valid:
                            acc_val, loss_val = self.valid_batch(batch_validX, batch_validY, device, criterion, optimizer)
                            batch_acc_val += acc_val 
                            batch_loss_val += loss_val                   
                            batch_valid += 1 
                            pbar_valid.set_postfix({"loss": batch_loss_val/batch_valid, "acc": batch_acc_val/batch_valid})
                        
                
                # print results
                if (epoch + 1) % 5 == 0: 
                    print(f'Epoch [{epoch+1}/{epochs}]: train loss: {batch_loss/batch_train:.4f}, train accuracy: {batch_acc/batch_train:.4f}, valid loss: {batch_loss_val/batch_valid:.4f}, valid accuracy: {batch_acc_val/batch_valid:.4f}')  
            
                # save history
                dic_history["loss"].append(batch_loss/batch_train)
                dic_history["accuracy"].append(batch_acc/batch_train)
                dic_history["val_loss"].append(batch_loss_val/batch_valid) 
                dic_history["val_accuracy"].append(batch_acc_val/batch_valid)
            
            self.history = dic_history
        
        else: 
            for epoch in range(epochs): 
                # train mode 
                model.train()      
                batch_acc, batch_loss = 0, 0 
                
                for batch_X, batch_y in tqdm(dataloader, total=len(dataloader), leave=False): 
                    acc, loss = self.train_batch(batch_X, batch_y, device, criterion, optimizer)
                    batch_acc += acc 
                    batch_loss += loss
                    
                print(f'Epoch [{epoch+1}/{epochs}], train loss: {batch_loss/batch_size:.4f}, train accuracy: {batch_acc/batch_size:.4f}')         
        
        
    def train_batch(self, batch_X, batch_y, device, criterion, optimizer):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device) 

        # forward 
        outputs = self.algorithm(batch_X).squeeze(1)
        loss = criterion(outputs, batch_y.float())

        # backward / optimize 
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step() 

        # recording history 
        predicted = (outputs > 0.5).int() 
        total = batch_y.size(0) 
        correct = (predicted == batch_y).sum().item()
        acc = correct / total 
        
        return acc, loss.item()  

    def valid_batch(self, batch_validX, batch_validY, device, criterion, optimizer): 
        # device 
        device = self.device         
        
        batch_validX, batch_validY = batch_validX.to(device), batch_validY.to(device) 

        # predict
        outputs = self.algorithm(batch_validX).squeeze(1) 
        loss_val = criterion(outputs, batch_validY.float())

        # recording history
        predicted = (outputs > 0.5).int() 
        total = batch_validY.size(0) 
        correct = (predicted == batch_validY).sum().item() 
        acc_val = correct / total 

        return acc_val, loss_val.item()  
            
    
    def decision(self, df):
        lst_drop = ['Code', 'Date', 'next_change']
        for col in lst_drop:
            if col in df.columns:
                df = df.drop(columns=[col])
        if self.framework == "tensorflow":  
            amount = self.decision_tensorflow(df)
            
        elif self.framework == "pytorch": 
            amount = self.decision_pytorch(df)

        elif self.framework == None:      
            amount = (self.algorithm.predict_proba(df)[:, 1] >= self.threshold).astype('int')
        
        else: 
            raise ValueError('The framework is not valid. Only "tensorflow", "pytorch", and None can enter.') 
        
        return amount 
    
    
    def decision_tensorflow(self, df): 
        return (self.algorithm.predict(self.data_transform(df.values.tolist())) >= self.threshold).astype('int').reshape(1, -1)[0]
    
    
    def decision_pytorch(self, df): 
        X = torch.tensor(self.data_transform(df.values.tolist()), dtype=torch.float32)
        y = torch.randint(0, 2, (len(X),), dtype=torch.float32) # temp y for dataloader  
        batch_size = self.params["batch_size"]
        device = self.device 
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # model 
        model = self.algorithm
        
        # evaulation mode 
        model.eval() 
        
        amount = np.array([])
        with torch.no_grad(): 
            for batch_X, batch_y in dataloader: 
                batch_X, batch_y = batch_X.to(device), batch_y.to(device) 
                
                outputs = model(batch_X).cpu() 
                amount_batch = np.array(outputs.reshape(1, len(batch_X.cpu()))[0] >= self.threshold).astype("int")
                amount = np.concatenate((amount, amount_batch))
                
        del y 
        
        return amount 
    
    
class Seller:
    def __init__(self, sub_seller):
        self.sub_seller = sub_seller
        
    def decision_all(self, trader_object, dtype='test', data=None, data_scaled=None):
        if dtype == 'test': 
            df = trader_object.testX
        
        # Stock Purchase Signal List 
        sub = self.sub_seller
        total_amount = 1
        total_amount *= sub.decision_next_day_sell(df)
        
        if dtype == 'test':
            lst_code_date = trader_object.test_code_date
        
        # Preparation of a sales diary
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