import pandas as pd
import seaborn as sns
import os
from sklearn.datasets import load_iris

def download_iris_data():
    """iris veri setini indir"""
    
    # Veri dizinlerini oluştur
    os.makedirs('data/raw', exist_ok=True)
    
    # veri setini yükle
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    
    # Ham veriyi kaydet
    df.to_csv('data/raw/iris.csv', index=False)
    
    print("✅ İris veri seti indirildi: data/raw/titanic.csv")
    print(f"Veri boyutu: {df.shape}")
    print(f"Kolonlar: {list(df.columns)}")
    print(f"Eksik değerler:\n{df.isnull().sum()}")
    
    return df

if __name__ == "__main__":
    download_iris_data()