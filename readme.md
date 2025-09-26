## IRIS Dataset DVC ile tahminleme projesi

## 📁 Proje Yapısı

```
titanic-mlops/
├── src/                        # Kaynak kodlar
│   ├── data_preprocess.py      # Veri inceleme ve önişlem
│   ├── download_data.py        # Veri indirme
│   ├── train_model.py          # Model eğitimi
├── data/                       # Veri dosyaları
│   ├── raw/                    # Ham veri
│   └── processed/              # İşlenmiş veriler
├── models/                     # Eğitilmiş modeller
│   ├── %_model.pkl             # Model özellikleri
│   └── metrics_%.json          # Model metrikleri
├── .dvc/                       # DVC yapılandırması
├── dvc.yaml                    # DVC pipeline
├── dvc.lock                    # DVC lock dosyası
├── .dvcignore                  # DVC ignore dosyası
├── .dockerignore               # Docker ignore dosyası
├── requirements.txt            # Python bağımlılıkları
└── README.md                   # Proje dokümantasyonu
```

## 🚀 Hızlı Başlangıç

### 1. Projeyi Klonla ve Kurulum Yap

```bash
git clone <repo-url>
cd titanic-mlops
pip install -r requirements.txt
```

### 2. DVC Pipeline Çalıştır

```bash
# Veri hazırlama + model eğitimi
dvc repro

# Sonuçları kontrol et
dvc metrics show
```
```
ornek cikti:
Ozbert@LAPTOP-TGD5GNUQ MINGW64 ~/vscode (main)
$ dvc metrics show
Path                     accuracy    model_type           n_test_samples    n_train_samples
models\metrics_rf.json   0.9         Random Forest        30                120
models\metrics_lg.json   0.93333     Logistic Regression  30                120
models\metrics_svm.json  0.96667     SVM                  30                120
(.venv) 


```
