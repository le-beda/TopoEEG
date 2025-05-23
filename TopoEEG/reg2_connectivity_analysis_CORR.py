import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
from itertools import combinations


# Reg2. Connectivity analysis, CORR
class CorrelationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 psd_calculator,
                 domain):  
        
        self.psd_calculator = psd_calculator
        self.domain = domain
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        n_samples, n_channels, n_timestamps = X.shape
        n_pairs = (n_channels * (n_channels - 1)) // 2  # Number of unique channel pairs
        
        X_corr = np.zeros((n_samples, n_pairs)) # Initialize output array
        
        for i in tqdm(range(n_samples), desc="Computing features (Reg2. Connectivity analysis, CORR)"):
            corr_matrix = np.zeros((n_channels, n_channels))
            
            for j, k in combinations(range(n_channels), 2):
                # corr_val = self._compute_correlation( self.psd_calculator.PSD(X[i, j, :], self.domain) , self.psd_calculator.PSD(X[i, k, :], self.domain) )
                corr_val = self._compute_correlation( X[i, j, :] , X[i, k, :] )
                corr_matrix[j, k] = corr_val
                corr_matrix[k, j] = corr_val  # Symmetric
            
            np.fill_diagonal(corr_matrix, 1.0)  # Auto-correlation = 1
            
            triu_indices = np.triu_indices(n_channels, k=1)
            X_corr[i, :] = corr_matrix[triu_indices]

                
        return X_corr

    def _compute_correlation(self, x, y):
        return np.corrcoef(x, y)[0, 1]
        # return np.cov(x, y)[0, 1]