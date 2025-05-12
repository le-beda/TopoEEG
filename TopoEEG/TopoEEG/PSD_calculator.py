import mne
import numpy as np


class PSD_calculator:
    def __init__(self,
                 freq_sample,
                 frequency_domains,
                 features_n=800):
        
        self.features_n = features_n
        self.frequency_domains = frequency_domains

        psd, freqs = mne.time_frequency.psd_array_welch(freq_sample,
                                                        512/7,
                                                        fmin=frequency_domains['all'][0],
                                                        fmax=frequency_domains['all'][1],
                                                        n_per_seg=256,
                                                        n_fft=2**15,
                                                        )

        self.frequency_domains_ids_topo = {}
        for domain in frequency_domains:
            ids = np.arange(len(freqs))[(freqs >= frequency_domains[domain][0]) & (freqs <= frequency_domains[domain][1])]
            if len(ids) < features_n:
                print(f'{domain} PSD length < {features_n} ({len(ids)})')
            self.frequency_domains_ids_topo[domain] = np.linspace(start=ids[0], stop=ids[-1], num=features_n, dtype=int)

    def PSD(self, freq_sample, domain):
        psds, freqs = mne.time_frequency.psd_array_welch(freq_sample,
                                                        512/7,
                                                        fmin=self.frequency_domains['all'][0],
                                                        fmax=self.frequency_domains['all'][1],
                                                        n_per_seg=256,
                                                        n_fft=2**15,
                                                        verbose=False
                                                        )
        return psds[self.frequency_domains_ids_topo[domain]]
