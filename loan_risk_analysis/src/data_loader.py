import pandas as pd
import kagglehub

def download_loan_data():
    """
    Kaggle'dan loan dataset'ini indirir
    """
    # Download latest version
    path = kagglehub.dataset_download("zhijinzhai/loandata")
    print("Path to dataset files:", path)
    return path

def load_data():
    """
    Loan verisini yükler
    """
    import os
    
    # Data klasörünü oluştur
    os.makedirs('../data', exist_ok=True)
    
    try:
        # Önce yerel dosyayı kontrol et
        df = pd.read_csv('../data/loan_data.csv')
        print("Yerel veri yüklendi!")
    except:
        print("Kaggle'dan veri indiriliyor...")
        dataset_path = download_loan_data()
        
        # Dataset dosyalarını kontrol et
        files = os.listdir(dataset_path)
        print("İndirilen dosyalar:", files)
        
        # İlk CSV dosyasını yükle
        csv_files = [f for f in files if f.endswith('.csv')]
        if csv_files:
            df = pd.read_csv(os.path.join(dataset_path, csv_files[0]))
            # Yerel kopyasını kaydet
            df.to_csv('../data/loan_data.csv', index=False)
            print("Veri başarıyla yüklendi ve kaydedildi!")
        else:
            raise Exception("CSV dosyası bulunamadı!")
    
    return df

if __name__ == "__main__":
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(df.head())