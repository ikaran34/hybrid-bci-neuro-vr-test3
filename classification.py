"""
Classifier Training (Shrinkage LDA & SVM) and Stratified 10-Fold Cross-Validation
"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class BCIClassifierPipeline:
    def __init__(self, model_type='lda', C=1.0):
        if model_type == 'lda':
            self.model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        elif model_type == 'svm-rbf':
            self.model = SVC(kernel='rbf', C=C, probability=True)
        else:
            self.model = SVC(kernel='linear', C=C, probability=True)

    def cross_validate(self, X_features, y, n_splits=10):
        """Performs 10-fold Stratified CV and computes evaluation metrics."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        accs, precs, recs, f1s = [], [], [], []
        
        for train_idx, test_idx in skf.split(X_features, y):
            X_tr, X_te = X_features[train_idx], X_features[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            
            self.model.fit(X_tr, y_tr)
            preds = self.model.predict(X_te)
            
            accs.append(accuracy_score(y_te, preds))
            precs.append(precision_score(y_te, preds))
            recs.append(recall_score(y_te, preds))
            f1s.append(f1_score(y_te, preds))
            
        return {
            'accuracy': np.mean(accs),
            'precision': np.mean(precs),
            'recall': np.mean(recs),
            'f1_score': np.mean(f1s)
        }
