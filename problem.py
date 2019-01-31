
import os
import math
import random
random.seed(42)
import numpy as np
import pandas as pd
import rampwf as rw

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle

problem_title = 'Cars price'
_target_column_names = 'price'
_ignore_column_names = ["postalCode"]
#_prediction_label_names = [0, 1]

Predictions = rw.prediction_types.make_regression(
    label_names=[_target_column_names])
# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorRegressor()


score_types = [
    rw.score_types.RMSE(name='rmse'),
]


def get_cv(X, y):
    y = y[:, np.newaxis]
    group = np.array(X['name'])
    X, y, group = shuffle(X, y, group, random_state=42)
    gkf = GroupKFold(n_splits=5).split(X, y, group)
    return gkf

    # cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    # return cv.split(X, y)

def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_names].values
    X_df = data.drop([_target_column_names] + _ignore_column_names, axis=1)
    
    return X_df, y_array
    # to make "!ramp_test_submission --submission starting_kit" work (in the starting_kit.ipynb), 
    # you need to change the line above by :
    # return clean_and_transform(X_df), y_array.reshape(-1,1)

    
def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)

def clean_and_transform(X):
    data = X.copy()
    
    # Replace the NaN-Values
    data['vehicleType'].fillna(value='blank', inplace=True)
    data['gearbox'].fillna(value='blank', inplace=True)
    data['model'].fillna(value='blank', inplace=True)
    data['fuelType'].fillna(value='blank', inplace=True)
    data['notRepairedDamage'].fillna(value='blank', inplace=True)
    
    data['dateCrawled'] = pd.to_datetime(data['dateCrawled'], format="%Y-%m-%d %H:%M:%S")
    data['dateCreated'] = pd.to_datetime(data['dateCreated'], format="%Y-%m-%d %H:%M:%S")
    data['lastSeen'] = pd.to_datetime(data['lastSeen'], format="%Y-%m-%d %H:%M:%S")
    
    for col in data:
        if data[col].dtype == "object":
            data[col] = data[col].astype('category')

    # Assign codes to categorical attributes instead of strings        
    cat_columns = data.select_dtypes(['category']).columns
    
    cat_columns = cat_columns[1:] #'name' column is not transformed
    
    data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)
    
    return data

