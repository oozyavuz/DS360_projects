"""
train.py
A) class_weight="balanced" (LR) + XGB(scale_pos_weight)
B) Random undersampling (LR + XGB)
C) SMOTE (LR + XGB)
Artifact'lar artifacts/ klasörüne kaydedilir.
"""

import os
import joblib
import numpy as np
import json
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
#modelleme ve örnekleme(resampling) araçları
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

# Paketli import (src klasörü ile - sorunsuz)
from data_preprocess import get_splits

current_dir = os.path.dirname(os.path.abspath(__file__))
# src klasörünün bir üstüne çık (project_root)
project_root = os.path.dirname(current_dir)
ARTIFACTS_DIR = os.path.join(project_root, "artifacts")



#artifacts/ yoksa oluştur; varsa devam et (hata verme).
def ensure_dir():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

#Pozitif/negatif örnek sayısını sayar.
#pos_w = neg/pos hesaplar. Bu değer, XGBoost’ta scale_pos_weight olarak kullanılır 
# (azınlık sınıfı dengelemek için).
def class_ratio(y):
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    pos_w = (neg / pos) if pos > 0 else 1.0 #azınlığa kaç kat önem vereceğine karar veriyor örN: pozitif:200 neg:400 2kat önemse xgbostta
    return pos, neg, pos_w

#her modelin başarısını değerlendirir ve yazdırır
def eval_and_print(title, y_test, y_pred, y_proba):
    print(f"\n=== {title} ===")
    print(classification_report(y_test, y_pred, digits=3))
    try:
        print("ROC AUC:", round(roc_auc_score(y_test, y_proba), 3))
    except Exception:
        pass

# Senaryo 1: Resampling yok
def scenario_no_resampling(save_tag="_cw"):
    """A) Resampling yok → LR(class_weight) ve XGB(scale_pos_weight)"""
    ensure_dir()
    X_train, X_test, y_train, y_test, pre = get_splits()

#Önemli: Eğitimde kullanılan kolon isim/sırası kaydedilir. Üretimde “X has N features…” hatalarını önler.
    with open(f"{ARTIFACTS_DIR}/feature_schema.json", "w") as f:
        json.dump({"columns": list(X_train.columns)}, f)



    # Logistic Regression (class_weight)
    #Pipeline: önce pre (ölçekleme/one-hot vs), sonra LR.
    #class_weight="balanced": sınıfları otomatik ağırlıklandırır.
    lr = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_proba = lr.predict_proba(X_test)[:, 1]
    eval_and_print("LR (class_weight)", y_test, y_pred, y_proba)
    joblib.dump(lr, f"{ARTIFACTS_DIR}/model_lr{save_tag}.pkl")
    joblib.dump(pre, f"{ARTIFACTS_DIR}/preprocessor{save_tag}.pkl")

    # XGBoost (scale_pos_weight = neg/pos)
    #Azınlık dengesini neg/pos ile çıkar.
    _, _, pos_w = class_ratio(y_train)
    xgb = Pipeline([
        ("pre", pre),
        ("clf", XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=pos_w #dengesizliği dengeler
        ))
    ])
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)[:, 1]
    eval_and_print("XGB (scale_pos_weight)", y_test, y_pred, y_proba)
    joblib.dump(xgb, f"{ARTIFACTS_DIR}/model_xgb{save_tag}.pkl") #Veriyi dokunmadan, modeli ağırlıklarla adil davranmaya zorluyoruz.


#Resampling var (sadece TRAIN’e!)Bu senaryonun kolon şeması da ayrıca kaydedilir (etiket: under veya smote).
def scenario_with_sampler(sampler, tag):
    """B/C) Resampling: sadece TRAIN'e uygula (leakage yok)"""
    ensure_dir()
    X_train, X_test, y_train, y_test, pre = get_splits()
        # ... scenario_with_sampler() içinde de aynı şekilde:
    with open(f"{ARTIFACTS_DIR}/feature_schema_{tag}.json", "w") as f:
        json.dump({"columns": list(X_train.columns)}, f)

    # TRAIN'i preprocessor ile fit-transform et
    #Çok kritik kural:

#pre sadece train’de fit edilir (testi hiç görmez).
#Resampling de sadece train’e yapılır.
#Böylece leakage yok.
    Xt_train = pre.fit_transform(X_train) #transform edilmiş xtrain
    Xt_res, y_res = sampler.fit_resample(Xt_train, y_train)

    # Logistic Regression:LR, resampled train ile eğitilir.
    lr = LogisticRegression(max_iter=1000)
    lr.fit(Xt_res, y_res)

    # Test → sadece transform fit yok, sonuçlar yazdırılır
    Xt_test = pre.transform(X_test)
    y_pred = lr.predict(Xt_test)
    y_proba = lr.predict_proba(Xt_test)[:, 1]
    eval_and_print(f"LR ({sampler.__class__.__name__})", y_test, y_pred, y_proba)

    # XGB de resampled train ile eğitilir (bu sefer scale_pos_weight yok; zaten veri dengelenmiş).
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42
    )
    xgb.fit(Xt_res, y_res)
    y_pred = xgb.predict(Xt_test)
    y_proba = xgb.predict_proba(Xt_test)[:, 1]
    eval_and_print(f"XGB ({sampler.__class__.__name__})", y_test, y_pred, y_proba)

    # Artefact’ları kaydet
    joblib.dump(pre, f"{ARTIFACTS_DIR}/preprocessor_{tag}.pkl")
    joblib.dump(lr,  f"{ARTIFACTS_DIR}/model_lr_{tag}.pkl")
    joblib.dump(xgb, f"{ARTIFACTS_DIR}/model_xgb_{tag}.pkl")


def main():
    # A) class_weight (LR) + XGB(scale_pos_weight)
    scenario_no_resampling(save_tag="_cw")

    # B) Random undersampling
    scenario_with_sampler(RandomUnderSampler(random_state=42), tag="under")

    # C) SMOTE
    scenario_with_sampler(SMOTE(random_state=42), tag="smote")


if __name__ == "__main__":
    main()
