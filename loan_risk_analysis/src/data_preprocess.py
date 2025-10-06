"""
Preprocessing + Split
- Hedef kodlama: PAIDOFF=1, diğerleri=0
- Leakage drop: paid_off_time, past_due_days, Loan_ID
- Tarih → timestamp
- OneHot + Impute
- Split (stratify)
- Opsiyon: SMOTE / UnderSampling (train aşamasında çağrılacak)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os
# Bu python dosyasının bulunduğu klasör
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# data klasörüne git
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "loan_data.csv")

TARGET_COL = "loan_status"

LEAKAGE_COLS = ["paid_off_time", "past_due_days"]
DROP_COLS = ["Loan_ID", TARGET_COL] + LEAKAGE_COLS

def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Tarihler
    for c in ["effective_date","due_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

#effective_date ve due_date'i datetime'a çeviriyoruz.
#errors="coerce" dersen:Tarihe çevrilemeyen (örneğin "abc" veya boş değer) hücreler NaT (Not a Time) olur
    
    # Türev: planlanan vade (gün)
    if {"effective_date","due_date"}.issubset(df.columns):
        df["planned_term_days"] = (df["due_date"] - df["effective_date"]).dt.days

    # Tarihleri timestamp'a çevir (saniye)
    for c in ["effective_date","due_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").astype("int64") // 10**9

    # Ek örnek: principal/terms (bölme hatasına karşı)
    if "Principal" in df.columns and "terms" in df.columns:
        terms = df["terms"].replace({0: np.nan})
        df["principal_per_term"] = df["Principal"] / terms
    #kredi tutarı(Principal) ile “ödeme sayısı” (terms) arasındaki ilişkiyi kullanarak yeni bir özellik çıkarıyor.
    #terms 0 olabilir, bölme hatasına karşı np.nan yapıyor.
    return df

def load_data() -> pd.DataFrame:
    p = Path(DATA_PATH)
    if not p.exists():
        raise FileNotFoundError(f"CSV bulunamadı: {p}")
    return pd.read_csv(p)

def make_xy(df: pd.DataFrame):
    y = df[TARGET_COL].replace({
        "PAIDOFF": 1,
        "COLLECTION": 0,
        "COLLECTION_PAIDOFF": 0
    }).astype(int)

    X = df.drop(columns=DROP_COLS, errors="ignore").copy()
    X = _feature_engineering(X)

    # Kolon grupları
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric, num_cols),  #num_cols yani numerik kolonlara numeric pipeline'ını uygula
        ("cat", categorical, cat_cols)   #car_cols yani kategoric değişkenlere categorical pipeline'ını uygula 
    ])

    return X, y, preprocessor

def get_splits(test_size=0.2, random_state=42):
    df = load_data()
    X, y, pre = make_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, pre

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, pre = get_splits()
    print("✅ Split hazır:", X_train.shape, X_test.shape)
    print("y dağılımı (train):", y_train.value_counts(normalize=True).round(3).to_dict())
