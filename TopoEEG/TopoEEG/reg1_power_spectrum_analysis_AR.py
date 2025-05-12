import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm


# Reg1. Power spectrum analysis, AR
class ARTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 psd_calculator,
                 domain, 
                 order=3):

        self.psd_calculator = psd_calculator
        self.domain = domain

        self.order = order
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        n_samples, n_channels, n_timestamps = X.shape
        n_freqs = self.psd_calculator.features_n
        X_ar = np.zeros((n_samples, n_channels * self.order))
        
        for i in tqdm(range(n_samples), desc="Computing features (Reg1. Power spectrum analysis, AR)"):
            for j in range(n_channels):
                y = self.psd_calculator.PSD(X[i, j, :], self.domain) 
    
                if len(y) <= self.order:
                    raise ValueError(
                        f"Time-series too short for AR({self.order}) modeling. "
                        f"Need at least {self.order + 1} samples."
                    )
                
                Y = y[self.order:]
                X_lag = np.zeros((n_freqs - self.order, self.order))
                
                for k in range(self.order):
                    X_lag[:, k] = y[self.order - (k + 1) : n_freqs - (k + 1)]
                
                XTX = np.dot(X_lag.T, X_lag)
                XTY = np.dot(X_lag.T, Y)
                beta = np.linalg.pinv(XTX) @ XTY 
                
                start_idx = j * self.order
                end_idx = start_idx + self.order
                X_ar[i, start_idx:end_idx] = beta
        
        return X_ar