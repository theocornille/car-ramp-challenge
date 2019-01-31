import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, Lasso, LassoCV, LassoLarsCV


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = DecisionTreeRegressor(random_state=42)
        #self.reg = LinearRegression()

    def fit(self, X, y):
        self.reg.fit(X, y)


    def predict(self, X):
        return self.reg.predict(X)[:, np.newaxis]  #(:, 1)