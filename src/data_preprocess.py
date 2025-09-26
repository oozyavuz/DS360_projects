import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import os
from typing import Tuple

logger = logging.getLogger(__name__)
input_path=r"C:\Users\Ozbert\vscode\data\raw\iris.csv"
output_dir=r"C:\Users\Ozbert\vscode\data\processed"
df = pd.read_csv(input_path)

def preprocess_data(input_path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Veriyi ön işleme
    
    Args:
        data: Ham veri
  
        
    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    
    """

    df = pd.read_csv(input_path)
    
    # Kopyasını al
    df_clean = df.copy()

    try:
        logger.info("Veri ön işleme başlatılıyor...")
        
        # Özellikler ve hedef değişkeni ayır
        feature_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        X = df_clean[feature_columns].values
        y = df_clean['target'].values
        
        # Veriyi train/test olarak böl
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Standardizasyon
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info(f"Veri ön işleme tamamlandı:")
        logger.info(f"  - Train seti: {X_train_scaled.shape}")
        logger.info(f"  - Test seti: {X_test_scaled.shape}")
        logger.info(f"  - Sınıf dağılımı: {np.bincount(y_train)}")

    except Exception as e:
        logger.error(f"Veri ön işleme hatası: {e}")
        raise   

    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Numpy array'leri kaydet
        np.save(f"{output_dir}/X_train.npy", X_train_scaled)
        np.save(f"{output_dir}/X_test.npy", X_test_scaled)
        np.save(f"{output_dir}/y_train.npy", y_train)
        np.save(f"{output_dir}/y_test.npy", y_test)
        
        logger.info(f"İşlenmiş veri kaydedildi: {output_dir}")
        
    except Exception as e:
        logger.error(f"Veri kaydetme hatası: {e}")
        raise
        
    

def get_feature_names() -> list:
    """Özellik isimlerini döndür"""
    return ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

def get_target_names() -> list:
    """Hedef sınıf isimlerini döndür"""
    return ['setosa', 'versicolor', 'virginica']

def analyze_data(input_path) -> dict:
    """
    Veri analizi yap
    
    Args:
        data: Veri seti
        
    Returns:
        dict: Analiz sonuçları
    """
    data = pd.read_csv(input_path)

    try:
        analysis = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'numeric_summary': data.describe().to_dict()
        }
        
        logger.info("Veri analizi tamamlandı")
        return analysis
        
    except Exception as e:
        logger.error(f"Veri analizi hatası: {e}")
        raise


def load_processed_data(data_dir: str = "data/processed") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    İşlenmiş veriyi yükle
    
    Args:
        data_dir: Veri dizini
        
    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    try:
        X_train = np.load(f"{data_dir}/X_train.npy")
        X_test = np.load(f"{data_dir}/X_test.npy")
        y_train = np.load(f"{data_dir}/y_train.npy")
        y_test = np.load(f"{data_dir}/y_test.npy")
        
        logger.info(f"İşlenmiş veri yüklendi: {data_dir}")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"İşlenmiş veri yükleme hatası: {e}")
        raise 

if __name__ == "__main__":
    analyze_data(input_path)
    preprocess_data(input_path)
    
