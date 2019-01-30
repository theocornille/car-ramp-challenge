import pandas as pd
import numpy as np

#scaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

class FeatureExtractor():
    def __init__(self):
        pass
    
    #def __init__(self,attribute_names):
       # self.attribute_names = attribute_names
        
                
        
    def fit(self, X_df, y=None):
        return self
    
    
    def transform(self, X_df):
        X_df_new = X_df.copy()
        
        #&(X_df_new["price"].between(100, 200000, inclusive=True))
        
        X_df_new = self.drop_columns(X_df_new)
        
        
        #scaler = StandardScaler()

        #X_df_new[['yearOfRegistration', 'gearbox', 'powerPS', 'model', 'kilometer', 'monthOfRegistration']] = scaler.fit_transform(X_df_new [['yearOfRegistration', 'gearbox', 'powerPS', 'model', 'kilometer', 'monthOfRegistration']])
        
        X_df_new = X_df_new.values
        return X_df_new
    
    def drop_columns(self, X_df, columns_to_drop=["dateCrawled", "abtest", "dateCreated", "lastSeen"]):
        X_df_new = X_df.copy()
        X_df_new = X_df_new.drop(columns_to_drop, axis=1)
        return X_df_new