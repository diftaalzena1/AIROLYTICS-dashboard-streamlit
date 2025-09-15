# utils/helpers.py
#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# ==============================
# Base path supaya path fleksibel
# ==============================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = DATA_DIR / "model_hybrid.pkl"
DF_LATIH_PATH = DATA_DIR / "df_latih.csv"
DF_HOT_PATH = DATA_DIR / "df_hot.csv"
GEOJSON_PATH = DATA_DIR / "provinces_idn.geojson"   # 🔹 ubah ke .geojson

# ==============================
# 1. Load Model
# ==============================
loaded = joblib.load(MODEL_PATH)
rf_model = loaded["rf_model"]
lr_model = loaded["lr_model"]
X_min = loaded["X_min"]
X_max = loaded["X_max"]

# ==============================
# 2. Load CSV
# ==============================
df_latih = pd.read_csv(DF_LATIH_PATH)
df_hot = pd.read_csv(DF_HOT_PATH)

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Bersihkan nama kolom: hapus spasi dan karakter aneh"""
    df.columns = df.columns.str.strip().str.replace('\ufeff','', regex=False)\
                            .str.replace(r'\s+', '_', regex=True)
    return df

# Terapkan clean_column_names
df_latih = clean_column_names(df_latih)
df_hot = clean_column_names(df_hot)

# ==============================
# 3. Load GeoJSON Provinsi
# ==============================
def load_geojson(path: Path = GEOJSON_PATH):
    """Load file GeoJSON provinsi"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

geojson_prov = load_geojson()  # otomatis load default

# ==============================
# 4. Klasifikasi IKU
# ==============================
def iku_category(value: float) -> str:
    if value >= 90:
        return "Sangat Baik"
    elif value >= 70:
        return "Baik"
    elif value >= 50:
        return "Sedang"
    elif value >= 25:
        return "Kurang"
    else:
        return "Sangat Kurang"

color_map_category = {
    "Sangat Baik": "darkgreen",
    "Baik": "green",
    "Sedang": "yellow",
    "Kurang": "orange",
    "Sangat Kurang": "red"
}

# ==============================
# 5. Weighted Hybrid Prediction
# ==============================
def weighted_hybrid_predict(row: pd.Series, threshold_minor=0.1, threshold_major=0.3) -> float:
    """Prediksi IKU dengan hybrid RF + LR"""
    diff_lower = (X_min - row).clip(lower=0)
    diff_upper = (row - X_max).clip(lower=0)
    extremity = (diff_lower + diff_upper).sum() / (X_max - X_min).sum()
    row_df = row.to_frame().T

    if extremity < threshold_minor:
        return rf_model.predict(row_df)[0]
    elif extremity < threshold_major:
        return 0.7*rf_model.predict(row_df)[0] + 0.3*lr_model.predict(row_df)[0]
    else:
        return lr_model.predict(row_df)[0]

# ==============================
# 6. Load CSV modular
# ==============================
def load_predictions(path: Path = DF_HOT_PATH) -> pd.DataFrame:
    """Load CSV prediksi. Default: df_hot.csv"""
    return pd.read_csv(path)

# ==============================
# 7. Bersihkan kolom numerik
# ==============================
numeric_cols = ["IKTL_(%)","Karhutla_(ha)","Kendaraan_Bermotor",
                "Rumah_Tangga_Listrik_PLN_(%)","Indeks_Kualitas_Udara_(%)"]

for col in numeric_cols:
    df_latih[col] = pd.to_numeric(
        df_latih[col].astype(str).str.replace('.', '', regex=False)
                            .str.replace(',', '.', regex=False)
    )
    df_hot[col] = pd.to_numeric(
        df_hot[col].astype(str).str.replace('.', '', regex=False)
                           .str.replace(',', '.', regex=False)
    )

# ==============================
# 8. Fitur untuk prediksi
# ==============================
# daftar nama kolom fitur input
features = ["IKTL_(%)", "Karhutla_(ha)", "Kendaraan_Bermotor", "Rumah_Tangga_Listrik_PLN_(%)"]

# nama lengkap / deskriptif untuk ditampilkan di plot
features_fullname = {
    "IKTL_(%)": "Persentase Penduduk Memiliki AKTL",
    "Karhutla_(ha)": "Luas Karhutla (ha)",
    "Kendaraan_Bermotor": "Jumlah Kendaraan Bermotor",
    "Rumah_Tangga_Listrik_PLN_(%)": "Persentase Rumah Tangga Memiliki Listrik PLN"
}

# ==============================
# 9. Data Augmentasi Deterministik
# ==============================
def augment_data_extreme(
    X: pd.DataFrame, 
    y: pd.Series, 
    model=None, 
    noise_level=0.05,
    extrapol_frac=0.15,
    extreme_frac=0.3,
    n_samples: int = 500,
    random_seed: int = 42
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Augmentasi deterministik: noise, extrapolation, extreme extrapolation.
    Digunakan untuk menambah variasi data sebelum training Random Forest.
    
    Args:
        X : fitur input
        y : target asli
        model : model sementara untuk prediksi y_aug (opsional)
        noise_level : proporsi noise
        extrapol_frac : proporsi extrapolasi normal
        extreme_frac : proporsi extrapolasi ekstrem
        n_samples : jumlah sample akhir yang diambil
        random_seed : seed untuk deterministik

    Returns:
        X_aug (DataFrame), y_aug (ndarray)
    """
    rng = np.random.default_rng(random_seed)
    X_aug_list, y_aug_list = [], []

    # -------------------------
    # 1. Noise injection
    # -------------------------
    noise = X * (1 + rng.uniform(-noise_level, noise_level, X.shape))
    y_noise = model.predict(noise) if model else y.values
    X_aug_list.append(noise)
    y_aug_list.append(y_noise)

    # -------------------------
    # 2. Extrapolasi biasa
    # -------------------------
    X_min_vals, X_max_vals = X.min().values, X.max().values
    X_range = (X_max_vals - X_min_vals).reshape(1, -1)
    X_extra = X.values + rng.uniform(-extrapol_frac, extrapol_frac, X.shape) * X_range
    y_extra = model.predict(X_extra) if model else y.values
    X_aug_list.append(X_extra)
    y_aug_list.append(y_extra)

    # -------------------------
    # 3. Extrapolasi ekstrem
    # -------------------------
    X_extreme = X.values + rng.uniform(-extreme_frac, extreme_frac, X.shape) * X_range
    y_extreme = model.predict(X_extreme) if model else y.values
    X_aug_list.append(X_extreme)
    y_aug_list.append(y_extreme)

    # -------------------------
    # 4. Gabungkan semua
    # -------------------------
    X_combined = np.vstack(X_aug_list)
    y_combined = np.concatenate(y_aug_list)

    # Subset deterministik
    if n_samples < len(y_combined):
        idx = rng.choice(len(y_combined), n_samples, replace=False)
        X_combined, y_combined = X_combined[idx], y_combined[idx]

    X_aug_df = pd.DataFrame(X_combined, columns=X.columns)
    return X_aug_df, y_combined


