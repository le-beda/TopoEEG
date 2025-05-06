import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from gtda.time_series import  SlidingWindow
from gtda.diagrams import Scaler
from gtda.homology import VietorisRipsPersistence
from gtda.pipeline import Pipeline
from gtda.time_series import TakensEmbedding
    
# Topo1. Persistent Homology
class FilteringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_long_living):
        self.n_long_living = n_long_living

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.reshape([-1] + list(X.shape))
        sorted_ids = np.argsort(-(X[:, :, :, 1]- X[:, :, :, 0]), axis=2).reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        X = np.take_along_axis(X, sorted_ids, axis=2)[:, :, :self.n_long_living, :2]

        return X.reshape(X.shape[0], -1)

class PersistentHomologyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 one_time_series_len,
                 psd_calculator,
                 domain,
                 n_long_living=30
                 ):
        
        self.psd_calculator = psd_calculator
        self.domain = domain

        window_size = one_time_series_len
        window_stride = one_time_series_len
        SW = SlidingWindow(size=window_size, stride=window_stride)

        embedding_time_delay_nonperiodic = 15
        embedding_dimension_nonperiodic = 6
        stride=5

        TE = TakensEmbedding(time_delay=embedding_time_delay_nonperiodic, dimension=embedding_dimension_nonperiodic, stride=stride)


        homology_dimensions = [1]
        VRP = VietorisRipsPersistence(homology_dimensions=homology_dimensions)

        diagramScaler = Scaler()

        self.n_long_living = n_long_living
        diagramFiltering = FilteringTransformer(n_long_living=self.n_long_living)

        multiple_series_steps = [('window', SW),
                ('embedding', TE),
                ('diagrams', VRP),
                ('scaler', diagramScaler),
                ('filtering', diagramFiltering),
                ]
        
        self.pipeline = Pipeline(multiple_series_steps)

    def fit(self, X, y=None):
        self.pipeline.fit(np.vstack( [np.concatenate( [self.psd_calculator.PSD(freqs, self.domain) for freqs in record]) for record in tqdm(X, desc="Fitting transformer (Topo1. Persistent Homology)")] ),
                          y)
        return self

    def transform(self, X):
        n_samples, n_channels, n_timesteps = X.shape

        X_topology = np.zeros((n_samples, n_channels * self.n_long_living * 2))
        for i in tqdm(range(n_samples), desc="Computing features (Topo1. Persistent Homology)"):
            X_topology[i, :] = self.pipeline.transform(np.concatenate( [self.psd_calculator.PSD(freqs, self.domain) for freqs in X[i]] ))

        return X_topology
