import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import os
import logging
import time
from sklearn.model_selection import cross_val_score
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def train_model(X_train: np.ndarray, y_train: np.ndarray, 
                 X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    
    """
    Farklı modelleri eğit ve takip et
    
    Args:
        X_train: Eğitim özellikleri
        y_train: Eğitim hedefleri
        X_test: Test özellikleri
        y_test: Test hedefleri
        experiment_name: MLflow deney adı
        
    Returns:
        Dict: Eğitilen modeller ve metrikleri
    """


    logger.info("Logistic Regression eğitiliyor...")
    start_time = time.time()
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    # Tahminler
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
            
    # Metrikler
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)

    metrics = {
        'model_type': 'Logistic Regression',
        'accuracy': float(accuracy),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test)
    }

    # Model ve metrik kaydet
    os.makedirs('models', exist_ok=True)
    model_path = f'models/logistic_regression_model.pkl'
    joblib.dump(model, model_path)

    with open('models/metrics_lg.json', 'w') as f:
        json.dump(metrics, f, indent=2)
            
    logger.info(f"Logistic Regression tamamlandı - Accuracy: {accuracy:.4f}")

    # 2. Random Forest

    logger.info("Random Forest eğitiliyor...")
            
    start_time = time.time()
            
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
            
    training_time = time.time() - start_time
            
    # Tahminler
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
            
    # Metrikler
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)

    # Metrikleri kaydet
    metrics = {
        'model_type': 'Random Forest',
        'accuracy': float(accuracy),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test)
    }

    # Model ve metrik kaydet
    os.makedirs('models', exist_ok=True)
    model_path = f'models/random_forest_model.pkl'
    joblib.dump(model, model_path)

    with open('models/metrics_rf.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Random Forest tamamlandı - Accuracy: {accuracy:.4f}")

    return model,metrics

if __name__ == "__main__":
    data_dir: str = "data/processed"
    X_train = np.load(f"{data_dir}/X_train.npy")
    y_train = np.load(f"{data_dir}/y_train.npy")
    X_test = np.load(f"{data_dir}/X_test.npy")
    y_test = np.load(f"{data_dir}/y_test.npy")

    train_model(X_train, y_train, X_test, y_test)

        