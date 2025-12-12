import os
import json
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from features import FeatureExtractor, extract_features_batch
from evaluate import calculate_metrics


class XGBoostBaseline:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42
    ):
        
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric="logloss"
        )
        
        self.feature_extractor = FeatureExtractor(use_tree_sitter=False)
        self.feature_names = [
            "line_count", "char_count", "avg_line_length", "max_line_length",
            "indent_depth_avg", "indent_depth_max", "num_tokens", "unique_tokens",
            "token_entropy", "num_identifiers", "avg_identifier_length",
            "identifier_entropy", "snake_case_ratio", "camel_case_ratio",
            "comment_ratio"
        ]
    
    def extract_features(
        self,
        codes: List[str],
        languages: Optional[List[str]] = None
    ) -> np.ndarray:
        return extract_features_batch(codes, languages, show_progress=True)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        if eval_set:
            self.model.fit(
                X, y,
                eval_set=[eval_set],
                verbose=True
            )
        else:
            self.model.fit(X, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))
    
    def save(self, path: str):
        joblib.dump(self.model, path)
    
    def load(self, path: str):
        self.model = joblib.load(path)
        return self


def train_xgboost_baseline(
    train_codes: List[str],
    train_labels: List[int],
    val_codes: Optional[List[str]] = None,
    val_labels: Optional[List[int]] = None,
    train_languages: Optional[List[str]] = None,
    val_languages: Optional[List[str]] = None,
    output_dir: str = "./outputs/xgboost_baseline"
) -> Tuple[XGBoostBaseline, Dict[str, float]]:
    
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    model = XGBoostBaseline()
    
    X_train = model.extract_features(train_codes, train_languages)
    y_train = np.array(train_labels)
    
    X_val, y_val = None, None
    if val_codes and val_labels:
        X_val = model.extract_features(val_codes, val_languages)
        y_val = np.array(val_labels)
    
    if X_val is not None:
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
    else:
        model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    if X_val is not None:
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        metrics = calculate_metrics(y_val, y_pred, y_proba)
    else:
        y_pred = model.predict(X_train)
        y_proba = model.predict_proba(X_train)[:, 1]
        metrics = calculate_metrics(y_train, y_pred, y_proba)
    
    importance = model.get_feature_importance()
    sorted_importance = sorted(importance.items(), key=lambda x: -x[1])
    
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Macro F1: {metrics['macro_f1']:.4f}")
    print(f"   ROC-AUC:  {metrics.get('roc_auc', 0):.4f}")
    
    model.save(os.path.join(output_dir, "xgboost_model.pkl"))
    
    results = {
        "metrics": metrics,
        "feature_importance": importance,
        "training_time_seconds": training_time
    }
    
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    
    return model, metrics


def quick_baseline_experiment(max_samples: int = 10000):
    from data_loader import load_semeval_data
    from config import DataConfig
    
    config = DataConfig()
    config.max_samples_train = max_samples
    config.max_samples_val = max_samples // 5
    
    train_samples, _ = load_semeval_data(config, "train")
    val_samples, _ = load_semeval_data(config, "validation")
    
    train_codes = [s.code for s in train_samples]
    train_labels = [s.label for s in train_samples]
    train_langs = [s.language for s in train_samples]
    
    val_codes = [s.code for s in val_samples]
    val_labels = [s.label for s in val_samples]
    val_langs = [s.language for s in val_samples]
    
    model, metrics = train_xgboost_baseline(
        train_codes, train_labels,
        val_codes, val_labels,
        train_langs, val_langs
    )
    
    return model, metrics


if __name__ == "__main__":
    quick_baseline_experiment(50000)
