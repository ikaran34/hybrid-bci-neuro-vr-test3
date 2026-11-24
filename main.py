"""
BCI Master Pipeline CLI Orchestrator
Executes full 180-minute course practical task workflow.
"""
from dataset_generator import SyntheticBCIDataset
from preprocessing import EEGPreprocessor
from csp import CommonSpatialPatterns
from classification import BCIClassifierPipeline
from online_simulation import OnlineRingBuffer, compute_itr

def main():
    print("="*60)
    print(" BCI Practical Pipeline Execution (180 Min Lab Reference)")
    print("="*60)

    # 1. Dataset Provisioning
    print("\n[1/4] Generating Synthetic MI Dataset...")
    generator = SyntheticBCIDataset(n_trials_per_class=20)
    X_raw, y = generator.generate()
    print(f" -> Raw EEG Matrix: {X_raw.shape} (N_trials x N_ch x N_samples)")

    # 2. Preprocessing
    print("\n[2/4] Executing Bandpass (8-30Hz) & Notch Filtering...")
    preprocessor = EEGPreprocessor()
    X_clean = preprocessor.transform(X_raw)

    # 3. CSP Spatial Filtering
    print("\n[3/4] Fitting Common Spatial Patterns (CSP m=4)...")
    csp = CommonSpatialPatterns(n_components=4)
    csp.fit(X_clean, y)
    X_csp = csp.transform(X_clean)
    print(f" -> Feature Matrix: {X_csp.shape}")

    # 4. Classification & 10-Fold CV
    print("\n[4/4] Cross-Validating Shrinkage LDA...")
    pipeline = BCIClassifierPipeline(model_type='lda')
    metrics = pipeline.cross_validate(X_csp, y, n_splits=10)
    
    print("\n" + "-"*40)
    print(" PERFORMANCE BENCHMARK SUMMARY")
    print("-"*40)
    print(f" 10-Fold CV Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f" Precision:           {metrics['precision']*100:.2f}%")
    print(f" Recall:              {metrics['recall']*100:.2f}%")
    print(f" F1-Score:            {metrics['f1_score']:.3f}")
    
    itr = compute_itr(n_classes=2, accuracy=metrics['accuracy'], time_sec=0.5)
    print(f" Estimated Online ITR: {itr:.2f} bits/min")
    print("="*60)

if __name__ == '__main__':
    main()
