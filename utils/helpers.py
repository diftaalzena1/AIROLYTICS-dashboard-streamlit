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
MODEL_PATH = DATA_DIR / "rf_augmentasi.pkl"   
DF_LATIH_PATH = DATA_DIR / "df_latih.csv"
DF_HOT_PATH = DATA_DIR / "df_hot.csv"
GEOJSON_PATH = DATA_DIR / "provinces_idn.geojson"   

# ==============================
# 1. Load Model RF Augmentasi
# ==============================
rf_model = joblib.load(MODEL_PATH)

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

df_latih = clean_column_names(df_latih)
df_hot = clean_column_names(df_hot)

# ==============================
# 3. Load GeoJSON Provinsi
# ==============================
def load_geojson(path: Path = GEOJSON_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

geojson_prov = load_geojson()

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
# 5. Prediksi RF Augmentasi
# ==============================
def rf_predict(row: pd.Series) -> float:
    """Prediksi IKU menggunakan RF augmentasi"""
    row_df = row.to_frame().T
    return rf_model.predict(row_df)[0]

# ==============================
# 6. Load CSV modular
# ==============================
def load_predictions(path: Path = DF_HOT_PATH) -> pd.DataFrame:
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
features = ["IKTL_(%)", "Karhutla_(ha)", "Kendaraan_Bermotor", "Rumah_Tangga_Listrik_PLN_(%)"]

features_fullname = {
    "IKTL_(%)": "Persentase Penduduk Memiliki AKTL",
    "Karhutla_(ha)": "Luas Karhutla (ha)",
    "Kendaraan_Bermotor": "Jumlah Kendaraan Bermotor",
    "Rumah_Tangga_Listrik_PLN_(%)": "Persentase Rumah Tangga Memiliki Listrik PLN"
}
