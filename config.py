"""
Central Configuration for BCI Motor Imagery Pipeline
"""
import numpy as np

# Data Acquisition Specs
SFREQ = 250.0  # Sampling Frequency (Hz)
CHANNELS = ['C3', 'Cz', 'C4', 'Fz', 'Pz', 'O1', 'O2']
N_CHANNELS = len(CHANNELS)

# Motor Imagery Paradigm
CLASSES = {0: 'Left Hand Imagery', 1: 'Right Hand Imagery'}
EPOCH_TMIN = 0.0  # Seconds
EPOCH_TMAX = 4.0  # Seconds
N_SAMPLES_PER_EPOCH = int((EPOCH_TMAX - EPOCH_TMIN) * SFREQ)

# Frequency Band Filtering (Sensorimotor Rhythms)
FILTER_LOW = 8.0   # Mu rhythm lower bound (Hz)
FILTER_HIGH = 30.0 # Beta rhythm upper bound (Hz)
NOTCH_FREQ = 50.0  # Powerline noise cutoff (Hz)

# CSP Parameters
CSP_M_COMPONENTS = 4  # Number of spatial filters (2 pairs)

# Online Buffer Simulation
BUFFER_WINDOW_SEC = 0.500  # 500 ms sliding window
BUFFER_STEP_SEC = 0.0625   # 62.5 ms step size (16 Hz decision rate)
MAX_ALLOWABLE_LATENCY_MS = 15.0
