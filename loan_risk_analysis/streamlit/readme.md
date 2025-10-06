Loan Risk Analysis Project
Bu proje, kredi başvurularının risk analizini yapan bir makine öğrenmesi projesidir. SMOTE, undersampling ve class weights teknikleri kullanılarak dengesiz veri problemi çözülmüş, Logistic Regression ve XGBoost modelleri ile karşılaştırma yapılmıştır.

📁 Proje Yapısı
loan_risk_analysis/
├── data/                    # Veri dosyaları
│   ├── loan_data.csv       # Ana veri seti
│   └── processed/          # İşlenmiş veriler
├── notebooks/               # Jupyter notebook'lar
│   └── eda.ipynb        # Keşifsel veri analizi
├── src/                     # Python source kodları
│   ├── data_loader.py      # Veri yükleme
│   ├── data_preprocess.py    # Veri ön işleme
│   ├── eda.py              # Keşifsel veri analizi
│   └── train.py            # Model eğitimi
├── streamlit/          # Streamlit uygulaması
│   └── app.py              # Ana dashboard
├── artifacts/              # Eğitilmiş modeller ve metadata
│   ├── model_*.pkl         # Trained models
│   ├── preprocessor_*.pkl  # Preprocessing pipelines
│   └── feature_schema*.json # Feature schemas
├── models/                 # Model çıktıları (opsiyonel)
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore dosyası
└── README.md              # Bu dosya