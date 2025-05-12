import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin


# Topo2. Network Topology 
class NetworkTopologyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 freq_bands={
                    'delta' : (0.9, 4.),
                    'theta' : (4. , 8.),
                    'alpha' : (8. , 14.),
                    'beta'  : (14., 25.),
                    'gamma' : (25., 40.),
                    'all'   : (0.5, 40.)
                }, 
                 threshold=0.5, 
                 sfreq=512):
        
        self.freq_bands = freq_bands
        self.threshold = threshold
        self.sfreq = sfreq
        
    def fit(self, X, y=None):
        self.n_bands_ = len(self.freq_bands)
        self.band_names_ = list(self.freq_bands.keys())
        return self
    
    def transform(self, X):
        n_samples, n_channels, n_timesteps = X.shape
        
        n_features = 0
        if self.include_degree:
            n_features += n_channels
        if self.include_global_efficiency:
            n_features += 1
        if self.include_clustering:
            n_features += n_channels
        if self.include_transitivity:
            n_features += 1
            
        X_topology = np.zeros((n_samples, self.n_bands_ * n_features))
        
        for i in tqdm(range(n_samples), desc="Computing features (Topo2. Network Topology)"):
            spectra = X[i, :, :]  # (n_channels, n_timesteps)
            
            connectivity_matrices = []
            for band, (fmin, fmax) in self.freq_bands.items():
                freqs = np.linspace(0, self.sfreq/2, n_timesteps)
                band_idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
                
                if len(band_idx) == 0:
                    print(f"No frequencies found in {band} band")
                    continue
                
                conn = self._compute_plv(spectra[:, band_idx])
                # conn = self._compute_coherence(spectra[:, band_idx])
                # conn = self._compute_imaginary_coherence(spectra[:, band_idx])
                
                connectivity_matrices.append(conn)
            
            connectivity_matrices = np.array(connectivity_matrices) 
            
            band_features = []
            for b in range(self.n_bands_):
                W = connectivity_matrices[b, :, :]  
                if self.threshold is not None:
                    W = (W > self.threshold).astype(float)
                
                curr_features = []
                
                # 1. Node degrees
                degrees = np.sum(W, axis=1)
                curr_features.extend(degrees)
                
                # 2. Global efficiency
                D = self._compute_shortest_paths(W)
                with np.errstate(divide='ignore', invalid='ignore'):
                    inv_D = 1 / D
                    inv_D[D == 0] = 0
                efficiency = np.sum(inv_D) / (n_channels * (n_channels - 1))
                curr_features.append(efficiency)
                
                # 3. Clustering coefficients
                clustering = np.zeros(n_channels)
                for j in range(n_channels):
                    neighbors = np.where(W[j, :] > 0)[0]
                    k = len(neighbors)
                    if k < 2:
                        clustering[j] = 0
                        continue
                        
                    subgraph = W[neighbors, :][:, neighbors]
                    triangles = np.sum(subgraph) / 2
                    clustering[j] = (2 * triangles) / (k * (k - 1))
                curr_features.extend(clustering)
                
                # 4. Transitivity
                triangles = 0
                triplets = 0
                for j in range(n_channels):
                    neighbors = np.where(W[j, :] > 0)[0]
                    k = len(neighbors)
                    if k < 2:
                        continue
                        
                    subgraph = W[neighbors, :][:, neighbors]
                    triangles += np.sum(subgraph) / 2
                    triplets += k * (k - 1) / 2
                
                transitivity = triangles / triplets if triplets > 0 else 0
                curr_features.append(transitivity)
                
                band_features.extend(curr_features)
            
            X_topology[i, :] = band_features
        
        return X_topology
    
    def _compute_plv(self, spectra):
        n_channels = spectra.shape[0]
        plv_matrix = np.zeros((n_channels, n_channels))
        
        phases = np.angle(spectra)
        
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                phase_diff = phases[i, :] - phases[j, :]
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                plv_matrix[i, j] = plv
                plv_matrix[j, i] = plv
        
        np.fill_diagonal(plv_matrix, 1)
        return plv_matrix
    
    def _compute_coherence(self, spectra):
        n_channels = spectra.shape[0]
        coh_matrix = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(i, n_channels):
                if i == j:
                    coh_matrix[i, j] = 1
                else:
                    cross_power = np.mean(spectra[i, :] * np.conj(spectra[j, :]))
                    power_i = np.mean(np.abs(spectra[i, :])**2)
                    power_j = np.mean(np.abs(spectra[j, :])**2)
                    coh = np.abs(cross_power) / np.sqrt(power_i * power_j)
                    coh_matrix[i, j] = coh
                    coh_matrix[j, i] = coh
        
        return coh_matrix
    
    def _compute_imaginary_coherence(self, spectra):
        n_channels = spectra.shape[0]
        icoh_matrix = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(i, n_channels):
                if i == j:
                    icoh_matrix[i, j] = 0
                else:
                    cross_power = np.mean(spectra[i, :] * np.conj(spectra[j, :]))
                    power_i = np.mean(np.abs(spectra[i, :])**2)
                    power_j = np.mean(np.abs(spectra[j, :])**2)
                    coh = cross_power / np.sqrt(power_i * power_j)
                    icoh_matrix[i, j] = np.abs(np.imag(coh))
                    icoh_matrix[j, i] = np.abs(np.imag(coh))
        
        return icoh_matrix
    
    def _compute_shortest_paths(self, W):
        n_nodes = W.shape[0]
        D = np.full((n_nodes, n_nodes), np.inf)
        np.fill_diagonal(D, 0)
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if W[i, j] > 0:
                    D[i, j] = 1 / W[i, j] 
        
        # Floyd-Warshall algorithm
        for k in range(n_nodes):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if D[i, j] > D[i, k] + D[k, j]:
                        D[i, j] = D[i, k] + D[k, j]
        
        return D