import pandas as pd
import numpy as np

#scaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

class FeatureExtractor():
    def __init__(self):
        pass

    
    def fit(self, X_df, y=None):
        return self
    
    
    def transform(self, X_df, y=None):
        X_df_new = X_df.copy()
        
        #add features
        X_df_new = self.add_features(X_df_new)
        
        #drop columns
        X_df_new = self.drop_columns(X_df_new)
        
        #select specific raws in a table (need y to reduce y length too)
        if y is not None:
            X_df_new, y = self.select_data(X_df_new, y)
        
        
        X_df_new = X_df_new.values
        
        if y is not None:
            return X_df_new, y
        else:
            return X_df_new
    
    
    def drop_columns(self, X_df, columns_to_drop=["name", "dateCrawled", "abtest", "dateCreated", "lastSeen", "seller"]):
        X_df_new = X_df.copy()
        X_df_new = X_df_new.drop(columns_to_drop, axis=1)
        return X_df_new
    
    
    def add_features(self, X_df):
        X_df_new = X_df.copy()
        X_df_new['name_len'] = [min(70, len(n)) for n in X_df_new['name']]
        
        return X_df_new
    
    def select_data(self, X, y):
        y_df = pd.DataFrame({'price': y}, columns = ['price'])
        data = pd.concat([X, y_df], axis=1, sort=False)
        
        data = data[(data["powerPS"].between(100, 500, inclusive=True))] 
        data = data[(data["price"].between(100, 200000, inclusive=True))]
        
        X_df_new = data.drop(["price"], axis=1)
        y = data["price"].values
        
        return X_df_new, y