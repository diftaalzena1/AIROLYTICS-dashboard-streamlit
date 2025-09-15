#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from modules.sidebar import show_sidebar
from tabs import tab1, tab2, tab3

# ==============================
# Config halaman
# ==============================
st.set_page_config(page_title="AIROLYTICS: Hybrid Machine Learning untuk Prediksi & Analisis Kualitas Udara Indonesia", layout="wide")

# ==============================
# Tambah CSS custom
# ==============================
st.markdown("""
<style>
/* Kurangi jarak kiri agar lebih dekat ke sidebar */
.block-container {  /* Container utama */
    padding-left: 3rem !important;
    padding-right: 3rem !important;
}

/* Jarak antar tab atas halaman */
button[role="tab"] {
    margin-right: 1.5rem;  /* sesuaikan sesuai selera */
    font-size: 18px;       /* ubah ukuran font tab */
    font-weight: 10000;    /* lebih tebal */
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
    "Evaluasi & Visualisasi"])

# ----- Tab 1 -----
with tabs[0]:
    tab1.show_tab()

# ----- Tab 2 -----
with tabs[1]:
    tab2.show_tab()

# ----- Tab 3 -----
with tabs[2]:
    tab3.show_tab()
