"""
Synthetic BCI Competition IV 2a Style Motor Imagery Dataset Generator
Simulates motor cortex ERD/ERS physics and EOG ocular artifacts.
"""
import numpy as np
from config import SFREQ, CHANNELS, N_SAMPLES_PER_EPOCH

class SyntheticBCIDataset:
    def __init__(self, n_trials_per_class=20, random_seed=42):
        np.random.seed(random_seed)
        self.n_trials = n_trials_per_class * 2
        self.labels = np.array([0] * n_trials_per_class + [1] * n_trials_per_class)
        np.random.shuffle(self.labels)

    def generate(self):
        """Generates multi-channel EEG trials [N_trials, N_channels, N_samples]."""
        X = []
        t = np.linspace(0, 4.0, N_SAMPLES_PER_EPOCH)
        
        for label in self.labels:
            # Baseline background noise
            trial = np.random.normal(0, 1.2, (len(CHANNELS), len(t)))
            
            # Add Mu (10Hz) & Beta (20Hz) background SMR activity
            mu_wave = np.sin(2 * np.pi * 10 * t)
            beta_wave = 0.5 * np.sin(2 * np.pi * 20 * t)
            
            # Physics ERD Simulation:
            # Class 0 (Left Hand): Suppression over C4 (Right Hemisphere)
            # Class 1 (Right Hand): Suppression over C3 (Left Hemisphere)
            c3_idx, c4_idx = CHANNELS.index('C3'), CHANNELS.index('C4')
            
            c3_amp = 0.3 if label == 1 else 1.2  # ERD on C3 during Right Hand
            c4_amp = 0.3 if label == 0 else 1.2  # ERD on C4 during Left Hand
            
            trial[c3_idx] += c3_amp * (mu_wave + beta_wave)
            trial[c4_idx] += c4_amp * (mu_wave + beta_wave)
            
            # Add ocular blink artifacts to Fz
            fz_idx = CHANNELS.index('Fz')
            blink_times = np.where(np.random.rand(len(t)) < 0.005)[0]
            for bt in blink_times:
                trial[fz_idx, bt:min(bt+30, len(t))] += 8.0 * np.hanning(min(30, len(t)-bt))
                
            X.append(trial)
            
        return np.array(X), self.labels
