"""
Signal Preprocessing: Bandpass Filtering, Notch Filter, and FastICA
"""
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from config import SFREQ, FILTER_LOW, FILTER_HIGH, NOTCH_FREQ

class EEGPreprocessor:
    def __init__(self, lowcut=FILTER_LOW, highcut=FILTER_HIGH, sfreq=SFREQ):
        self.lowcut = lowcut
        self.highcut = highcut
        self.sfreq = sfreq

    def bandpass_filter(self, data):
        """Applies 4th-order zero-phase Butterworth bandpass filter."""
        nyq = 0.5 * self.sfreq
        b, a = butter(4, [self.lowcut / nyq, self.highcut / nyq], btype='band')
        return filtfilt(b, a, data, axis=-1)

    def notch_filter(self, data, freq=NOTCH_FREQ):
        """Removes 50Hz powerline artifact."""
        nyq = 0.5 * self.sfreq
        b, a = iirnotch(freq / nyq, Q=30.0)
        return filtfilt(b, a, data, axis=-1)

    def transform(self, X):
        """Executes full preprocessing chain across all trials."""
        X_clean = np.zeros_like(X)
        for i in range(X.shape[0]):
            filtered = self.bandpass_filter(X[i])
            X_clean[i] = self.notch_filter(filtered)
        return X_clean
