from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class CustomImputer(BaseEstimator, TransformerMixin):
    """
    Custom imputer to fill missing values:
    - Numeric columns: fill with median
    - Categorical columns: fill with mode
    """

    def __init__(self):
        self.fill_values_ = {}

    def fit(self, X, y=None):
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.fill_values_[col] = X[col].median()
            else:
                self.fill_values_[col] = X[col].mode()[0] if not X[col].mode().empty else "Missing"
        return self

    def transform(self, X):
        X = X.copy()
        for col, value in self.fill_values_.items():
            X[col] = X[col].fillna(value)
        return X
