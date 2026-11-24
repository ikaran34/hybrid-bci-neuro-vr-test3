"""
Real-Time Online Ring Buffer Streaming Simulator & Wolpaw ITR Calculation
"""
import time
import numpy as np
from config import SFREQ, BUFFER_WINDOW_SEC, BUFFER_STEP_SEC

def compute_itr(n_classes, accuracy, time_sec):
    """Calculates Wolpaw's Information Transfer Rate (ITR) in bits/min."""
    if accuracy <= 1.0 / n_classes:
        return 0.0
    if accuracy >= 1.0:
        accuracy = 0.99999
        
    p = accuracy
    n = n_classes
    b = np.log2(n) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (n - 1))
    itr_bits_per_min = b * (60.0 / time_sec)
    return max(0.0, itr_bits_per_min)

class OnlineRingBuffer:
    def __init__(self, n_channels=7, window_sec=BUFFER_WINDOW_SEC, sfreq=SFREQ):
        self.window_samples = int(window_sec * sfreq)
        self.buffer = np.zeros((n_channels, self.window_samples))

    def push_chunk(self, chunk):
        """Applies sliding ring buffer update."""
        n_new = chunk.shape[1]
        self.buffer = np.roll(self.buffer, -n_new, axis=1)
        self.buffer[:, -n_new:] = chunk

    def process(self, preprocessor, csp, classifier):
        """Measures pipeline execution latency per streaming chunk."""
        t0 = time.perf_counter()
        
        # 1. Filter window
        filtered = preprocessor.bandpass_filter(self.buffer)
        # 2. Extract CSP features
        feat = csp.transform(np.array([filtered]))
        # 3. Predict class probability
        prob = classifier.model.predict_proba(feat)[0]
        
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return prob, elapsed_ms
