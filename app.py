#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from modules.sidebar import show_sidebar
from tabs import tab1, tab2, tab3

# ==============================
# Config halaman
# ==============================
st.set_page_config(
    page_title="AIROLYTICS: Model Random Forest dengan Augmentasi Data untuk Prediksi & Analisis Kualitas Udara Indonesia",
    layout="wide"
)

# ==============================
# Tambah CSS custom
# ==============================
st.markdown("""
<style>
/* Ubah background header bawaan Streamlit */
[data-testid="stHeader"] {
    background: linear-gradient(to right, #E2CEB1, #FDFCE8);
}

/* Kurangi jarak kiri agar lebih dekat ke sidebar */
.block-container {
    padding-left: 3rem !important;
    padding-right: 3rem !important;
}

/* Styling untuk tab navigasi */
button[role="tab"] {
    margin-right: 1.5rem;
    font-size: 18px;
    font-weight: 1000;  /* tebal */
}
</style>
""", unsafe_allow_html=True)

# --- Tampilkan sidebar di semua tab ---
show_sidebar()

# ==============================
# Tabs di atas halaman
# ==============================
tabs = st.tabs([
    "Tentang Proyek",
    "Demo Prediksi Interaktif",
    "Evaluasi & Visualisasi"
])

# ----- Tab 1 -----
with tabs[0]:
    tab1.show_tab()

# ----- Tab 2 -----
with tabs[1]:
    tab2.show_tab()

# ----- Tab 3 -----
with tabs[2]:
    tab3.show_tab()
