#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from rapidfuzz import process
import joblib
import json
import os
from utils.helpers import df_hot, df_latih, features, features_fullname, iku_category, geojson_prov, color_map_category

def show_tab():
    # ---------------------------------
    # CSS Global
    # ---------------------------------
    st.markdown("""
    <style>
    /* Metrics abu-abu muda */
    div[data-testid="stMetric"] {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 0 5px rgba(0,0,0,0.05);
    }

    /* Subheader gradasi coklat */
    .gradient-subheader-tab3 {
        font-size: 22px !important;
        font-weight: 600;
        background: linear-gradient(to right, #734128, #A47551);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
        line-height: 1.1;
        margin-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---------------------------------
    # Load Model & Metrics
    # ---------------------------------
    saved = joblib.load("data/model_hybrid.pkl")
    rf_model = saved["rf_model"]
    lr_model = saved["lr_model"]
    X_min = saved["X_min"]
    X_max = saved["X_max"]

    metrics_file = "data/eval_metrics.json"
    if not os.path.exists(metrics_file):
        st.error("File eval_metrics.json tidak ditemukan! Pastikan sudah disimpan saat training.")
        return
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    train_metrics = metrics.get("train_test", {})
    hot_metrics = metrics.get("hot_test", {})

    # Bersihkan kolom numerik
    for col in features:
        df_hot[col] = pd.to_numeric(df_hot[col], errors='coerce').fillna(0)
        df_latih[col] = pd.to_numeric(df_latih[col], errors='coerce').fillna(0)

    # Weighted Hybrid Predict Function
    def weighted_hybrid_predict_tab(row, threshold_minor=0.1, threshold_major=0.3):
        diff_lower = (X_min - row).clip(lower=0)
        diff_upper = (row - X_max).clip(lower=0)
        extremity = (diff_lower + diff_upper).sum() / (X_max - X_min).sum()
        row_df = row.to_frame().T
        if extremity < threshold_minor:
            return rf_model.predict(row_df)[0]
        elif extremity < threshold_major:
            return 0.7 * rf_model.predict(row_df)[0] + 0.3 * lr_model.predict(row_df)[0]
        else:
            return lr_model.predict(row_df)[0]

    # ---------------------------------
    # Evaluasi Metrics
    # ---------------------------------
    st.markdown('<div class="gradient-subheader-tab3">Evaluasi Metrics Hasil Training</div>', unsafe_allow_html=True)
    st.caption("Nilai diambil langsung dari hasil evaluasi training agar konsisten.")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² Train", f"{train_metrics.get('r2_train', 0):.4f}")
    col2.metric("R² Test", f"{train_metrics.get('r2_test', 0):.4f}")
    col3.metric("RMSE Test", f"{train_metrics.get('rmse_test', 0):.2f}")
    col4.metric("MAPE Test", f"{train_metrics.get('mape_test', 0):.2f}%")

    st.markdown('<div class="gradient-subheader-tab3">Evaluasi Metrics Hot Test</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² Hot", f"{hot_metrics.get('r2_hot', 0):.4f}")
    col2.metric("RMSE Hot", f"{hot_metrics.get('rmse_hot', 0):.2f}")
    col3.metric("MAE Hot", f"{hot_metrics.get('mae_hot', 0):.2f}")
    col4.metric("MAPE Hot", f"{hot_metrics.get('mape_hot', 0):.2f}%")

    # Prediksi Hot Test
    df_hot["Prediksi_IKU"] = df_hot[features].apply(weighted_hybrid_predict_tab, axis=1)
    df_hot["Kategori"] = df_hot["Prediksi_IKU"].apply(iku_category)

    # ---------------------------------
    # Peta Choropleth
    # ---------------------------------
    st.markdown('<div class="gradient-subheader-tab3">Peta Choropleth Prediksi Indeks Kualitas Udara (2022)</div>', unsafe_allow_html=True)

    # Narasi ringkas sebelum peta
    st.caption("Peta ini menampilkan sebaran prediksi kualitas udara per provinsi. Semakin gelap warnanya, semakin baik nilai IKU yang diprediksi. " \
    "Peta juga dapat digeser, di-zoom, dan kursor bisa diarahkan ke provinsi tertentu untuk melihat nilai detail.")

    # Tambahkan jarak pakai <br>
    # st.markdown("<br>", unsafe_allow_html=True)

    m = folium.Map(location=[-2.5, 118], zoom_start=5)
    geojson_prov_list = [f['properties']['Propinsi'].strip().upper() for f in geojson_prov['features']]

    def map_provinsi_auto(prov_name, choices, score_cutoff=80):
        result = process.extractOne(prov_name.strip().upper(), choices, score_cutoff=score_cutoff)
        return result[0] if result else prov_name.strip().upper()

    df_hot['Provinsi_geo'] = df_hot['Provinsi'].apply(lambda x: map_provinsi_auto(x, geojson_prov_list))
    prov_to_pred = df_hot.set_index('Provinsi_geo')['Prediksi_IKU'].to_dict()

    for feature in geojson_prov['features']:
        prov = feature['properties']['Propinsi'].strip().upper()
        pred_val = prov_to_pred.get(prov, None)
        feature['properties']['Prediksi_IKU'] = f"{pred_val:.2f}" if pred_val is not None else "N/A"

    folium.Choropleth(
        geo_data=geojson_prov,
        name='choropleth',
        data=df_hot,
        columns=['Provinsi_geo', 'Prediksi_IKU'],
        key_on='feature.properties.Propinsi',
        fill_color='YlOrBr',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Prediksi IKU (%), Semakin Gelap Semakin Baik',
    ).add_to(m)

    folium.GeoJson(
        geojson_prov,
        tooltip=folium.GeoJsonTooltip(
            fields=['Propinsi', 'Prediksi_IKU'],
            aliases=['Provinsi:', 'Prediksi IKU:'],
            localize=True
        )
    ).add_to(m)
    st_folium(m, width=None, height=500)

    st.markdown("""
    <div style="margin-top:10px; padding:10px; background-color:#8B5E3C; 
                color:white; border-radius:10px; font-size:14px;">
        🌍 <b>Wilayah timur (Papua & Maluku)</b> → tampil paling gelap, menandakan kualitas udara relatif lebih baik.
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:10px; padding:10px; background-color:#8B5E3C; 
                color:white; border-radius:10px; font-size:14px;">
        🏭 <b>Wilayah barat (Jawa & sebagian Sumatra)</b> → cenderung lebih terang, konsisten dengan dampak urbanisasi dan aktivitas industri yang padat.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:10px; padding:10px; background-color:#A47551; 
                color:white; border-radius:10px; font-size:14px;">
        🌳 <b>Wilayah tengah (Kalimantan & sebagian Sulawesi)</b> → relatif baik, dengan dominasi warna gelap tanpa area terang mencolok.  
    </div>
    """, unsafe_allow_html=True)

    # Insight utama setelah peta
    st.markdown("""
    <div style="margin-top:20px; padding:12px; background-color:#f0f0f0; 
                color:#333; border-radius:10px; font-size:14px; 
                box-shadow: 0 0 5px rgba(0,0,0,0.05);">
        ✨ <b>Insight utama:</b> terdapat perbedaan spasial yang jelas — 
        <b>timur unggul</b>, <b>barat tertekan urbanisasi</b>, sementara 
        <b>tengah berada pada posisi menengah</b>. Temuan ini penting untuk merancang strategi kebijakan yang lebih terarah per wilayah.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True) 

    # ---------------------------------
    # Top/Bottom 5 Provinsi
    # ---------------------------------
    st.markdown('<div class="gradient-subheader-tab3">Top 5 & Bottom 5 Provinsi Berdasarkan Prediksi IKU (2022)</div>', unsafe_allow_html=True)
    
    # Narasi ringkas sebelum grafik top/bottom
    st.caption("Grafik berikut memperlihatkan 5 provinsi teratas dan terbawah dalam prediksi kualitas udara. Perbandingan ini membantu melihat gap wilayah timur vs barat Indonesia.")

    df_sorted = df_hot.sort_values(by="Prediksi_IKU", ascending=False)
    top5 = df_sorted.head(5)
    bottom5 = df_sorted.tail(5).sort_values(by="Prediksi_IKU", ascending=True)
    col_top, col_bottom = st.columns(2)

    # Top 5
    with col_top:
        colors_top = ["#804000", "#A67845", "#C29A6C", "#D9B88C", "#E2CEB1"]
        fig_top = go.Figure()
        for i, row in enumerate(top5.itertuples()):
            fig_top.add_trace(go.Bar(
                x=[row.Prediksi_IKU],
                y=[row.Provinsi],
                orientation='h',
                marker_color=colors_top[i],
                text=f"{row.Prediksi_IKU:.2f}",
                textposition='outside',
                name=row.Provinsi
            ))
        fig_top.update_layout(yaxis={'categoryorder': 'total ascending', 'autorange': "reversed"},
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False, margin=dict(t=20, l=10, r=10, b=10), height=400)
        st.plotly_chart(fig_top, use_container_width=True)
    st.markdown("""
    <div style="margin-top:15px; padding:10px; background-color:#6B4226; 
                color:white; border-radius:10px; font-size:14px;">
        🏅 <b>Papua Barat & Papua</b> menempati posisi tertinggi (≥94%), diikuti Kalimantan Utara, Gorontalo, dan Maluku Utara (>91%) → menunjukkan bahwa daerah-daerah dengan tekanan urbanisasi rendah masih mampu menjaga kualitas udara.
    """, unsafe_allow_html=True)

    # Bottom 5
    with col_bottom:
        colors_bottom = ["#E2CEB1", "#D9B88C", "#C29A6C", "#A67845", "#804000"]
        fig_bottom = go.Figure()
        for i, row in enumerate(bottom5.itertuples()):
            fig_bottom.add_trace(go.Bar(
                x=[row.Prediksi_IKU],
                y=[row.Provinsi],
                orientation='h',
                marker_color=colors_bottom[i],
                text=f"{row.Prediksi_IKU:.2f}",
                textposition='outside',
                name=row.Provinsi
            ))
        fig_bottom.update_layout(yaxis={'categoryorder': 'total ascending'},
                                 plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False, margin=dict(t=20, l=10, r=10, b=10), height=400)
        st.plotly_chart(fig_bottom, use_container_width=True)
    st.markdown("""
    <div style="margin-top:15px; padding:10px; background-color:#8B5E3C; 
                color:white; border-radius:10px; font-size:14px;">
        ⚠️ <b>DKI Jakarta, Banten, dan tiga provinsi besar di Jawa</b> justru ada di posisi terbawah (<84%) → memperlihatkan tantangan serius akibat konsentrasi penduduk dan aktivitas ekonomi intensif.
    </div>
    """, unsafe_allow_html=True)

    # Insight utama setelah ttop/bottom
    st.markdown("""
    <div style="margin-top:20px; padding:12px; background-color:#f0f0f0; 
                color:#333; border-radius:10px; font-size:14px; 
                box-shadow: 0 0 5px rgba(0,0,0,0.05);">
        ✨ <b>Insight utama:</b> Grafik ini menyoroti <b>gap kualitas udara antarprovinsi</b>, yakni wilayah timur & sebagian tengah unggul dalam peringkat tertinggi, sementara provinsi padat di Jawa konsisten tertekan di posisi terbawah. Perbandingan ini menekankan pentingnya kebijakan diferensial, tidak hanya berbasis wilayah besar (timur–barat–tengah), tetapi juga berbasis <b>provinsi prioritas</b>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    # ---------------------------------
    # Analisis Faktor vs IKU (Scatter)
    # ---------------------------------
    st.markdown('<div class="gradient-subheader-tab3">Analisis Faktor terhadap Prediksi IKU (2022)</div>', unsafe_allow_html=True)
    
    # Narasi ringkas sebelum scatter
    st.caption("Scatter plot berikut memperlihatkan hubungan tiap faktor (lahan, karhutla, kendaraan, listrik) dengan kualitas udara. Pola tren membantu mengidentifikasi faktor dominan.")

    features_fullname = {
        "IKTL_(%)": "Indeks Kualitas Tutupan Lahan (%)",
        "Karhutla_(ha)": "Luas Kebakaran Hutan dan Lahan (ha)",
        "Kendaraan_Bermotor": "Jumlah Kendaraan Bermotor (unit)",
        "Rumah_Tangga_Listrik_PLN_(%)": "Persentase Rumah Tangga Listrik PLN (%)"
    }
    color_map_scatter = {
        "Sangat Baik": "#804000",
        "Baik": "#A67845",
        "Sedang": "#A1887F",
        "Kurang": "#D7CCC8",
        "Sangat Kurang": "#E2CEB5"
    }
    faktor_list = list(features_fullname.keys())
    for i in range(0, len(faktor_list), 2):
        col1, col2 = st.columns(2)
        with col1:
            faktor = faktor_list[i]
            st.markdown(f"**{features_fullname[faktor]} vs IKU Prediksi**")
            fig1 = px.scatter(df_hot, x=faktor, y="Prediksi_IKU", color="Kategori",
                              color_discrete_map=color_map_scatter, hover_name="Provinsi",
                              labels={faktor: features_fullname[faktor], "Prediksi_IKU": "IKU Prediksi"},
                              size_max=15)
            fig1.update_traces(marker=dict(size=14, line=dict(width=0.5, color='black')))
            fig1.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=400)
            st.plotly_chart(fig1, use_container_width=True)
            
        if i + 1 < len(faktor_list):
            faktor2 = faktor_list[i+1]
            with col2:
                st.markdown(f"**{features_fullname[faktor2]} vs IKU Prediksi**")
                fig2 = px.scatter(df_hot, x=faktor2, y="Prediksi_IKU", color="Kategori",
                                  color_discrete_map=color_map_scatter, hover_name="Provinsi",
                                  labels={faktor2: features_fullname[faktor2], "Prediksi_IKU": "IKU Prediksi"},
                                  size_max=15)
                fig2.update_traces(marker=dict(size=14, line=dict(width=0.5, color='black')))
                fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=400)
                st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div style="margin-top:15px; padding:10px; background-color:#6B4226; 
                color:white; border-radius:10px; font-size:14px;">
        🌳 <b>Indeks Kualitas Tutupan Lahan (%) →</b> pola positif: provinsi dengan tutupan lahan tinggi cenderung memiliki kualitas udara lebih baik. Vegetasi berperan penting sebagai penyerap polusi.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:10px; padding:10px; background-color:#8B5E3C; 
                color:white; border-radius:10px; font-size:14px;">
        🔥 <b>Luas Kebakaran Hutan dan Lahan (ha) →</b> meski tidak linier, titik ekstrem dengan area terbakar luas terlihat menekan kualitas udara. Karhutla menjadi faktor episodik dengan dampak signifikan saat terjadi.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:10px; padding:10px; background-color:#A47551; 
                color:white; border-radius:10px; font-size:14px;">
        🚗 <b>Jumlah Kendaraan Bermotor (unit) →</b> tren negatif jelas: semakin banyak kendaraan, kualitas udara menurun. Konsisten dengan fakta bahwa transportasi adalah sumber emisi terbesar di kota besar.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:10px; padding:10px; background-color:#C19A6B; 
                color:white; border-radius:10px; font-size:14px;">
        ⚡ <b>Persentase Rumah Tangga Listrik PLN (%) →</b> mayoritas provinsi sudah di atas 90%. Variabel ini lebih merefleksikan infrastruktur energi dan pola konsumsi, yang secara tidak langsung berkaitan dengan emisi.
    </div>
    """, unsafe_allow_html=True)

        # Insight utama setelah scatter plot
    st.markdown("""
    <div style="margin-top:20px; padding:12px; background-color:#f0f0f0; 
                color:#333; border-radius:10px; font-size:14px; 
                box-shadow: 0 0 5px rgba(0,0,0,0.05);">
        ✨ <b>Insight utama:</b> Scatter plot menegaskan bahwa <b>kendaraan bermotor</b> dan <b>tutupan lahan</b> paling berpengaruh terhadap kualitas udara. 
                <b>Karhutla</b> berdampak besar hanya pada kasus ekstrem, sementara akses listrik tidak signifikan. 
                Fokus utama kebijakan sebaiknya pada <b>pengendalian emisi transportasi</b> dan <b>perlindungan ekosistem hijau</b>.
    </div>
    """, unsafe_allow_html=True) 

    st.markdown("<br>", unsafe_allow_html=True) 

    # ---------------------------------
    # Feature Importance
    # ---------------------------------
    st.markdown('<div class="gradient-subheader-tab3">Feature Importance dengan Random Forest (2022)</div>', unsafe_allow_html=True)
    
    # Narasi ringkas sebelum feature importance
    st.caption("Grafik feature importance menunjukkan kontribusi relatif tiap faktor. Semakin besar skor, semakin kuat pengaruhnya terhadap prediksi IKU.")

    importances = rf_model.feature_importances_
    feat_imp = pd.Series(importances, index=list(features_fullname.values())).sort_values(ascending=True)
    fig = go.Figure(go.Bar(
        x=feat_imp.values,
        y=feat_imp.index,
        orientation='h',
        marker=dict(color=feat_imp.values, colorscale=[[0, '#E2CEB1'], [1, "#804000"]], showscale=False)
    ))
    fig.update_layout(xaxis_title="Importance Score", yaxis_title="Fitur",
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=20, l=80, r=20, b=5), height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div style="margin-top:15px; padding:10px; background-color:#6B4226; 
                color:white; border-radius:10px; font-size:14px;">
        🚗 <b>Jumlah Kendaraan Bermotor</b> menempati posisi paling dominan. 
                Ini konsisten dengan tren scatter plot dan top/bottom yang menunjukkan kota padat (Jakarta & Jawa) tertekan polusi transportasi.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:10px; padding:10px; background-color:#8B5E3C; 
                color:white; border-radius:10px; font-size:14px;">
        🌳 <b>Indeks Kualitas Tutupan Lahan</b> juga memiliki skor penting, 
                mengonfirmasi peran vegetasi sebagai penyangga kualitas udara di wilayah dengan ekosistem masih terjaga (Papua & Maluku).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:10px; padding:10px; background-color:#A47551; 
                color:white; border-radius:10px; font-size:14px;">
        🔥 <b>Karhutla</b> dan <b>akses listrik PLN</b> muncul dengan pengaruh lebih kecil, tetapi tetap signifikan pada kejadian ekstrem. 
                Ini melengkapi insight scatter plot yang menilai karhutla sebagai faktor “kejutan” bukan tren harian.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:10px; padding:10px; background-color:#A47551; 
                color:white; border-radius:10px; font-size:14px;">
        ⚡ <b>Persentase Rumah Tangga Listrik PLN (%)</b> memberikan kontribusi relatif rendah. Sejalan dengan analisis sebelumnya, 
                variabel ini lebih merefleksikan pembangunan infrastruktur dasar ketimbang faktor langsung pencemar udara.
    </div>
    """, unsafe_allow_html=True)

         # Insight utama setelah feature importance
    st.markdown("""
    <div style="margin-top:20px; padding:12px; background-color:#f0f0f0; 
                color:#333; border-radius:10px; font-size:14px; 
                box-shadow: 0 0 5px rgba(0,0,0,0.05);">
        ✨ <b>Insight utama:</b> Feature importance menegaskan kendaraan bermotor dan tutupan lahan sebagai dua faktor paling krusial, selaras dengan temuan peta dan scatter plot. 
                Sementara karhutla dan akses listrik tetap relevan sebagai konteks tambahan. Hal ini memperkuat dasar kebijakan: 
                <b>prioritas pengendalian emisi transportasi + perlindungan ekosistem hijau.</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True) 

    # ---------------------------------
    # Heatmap Korelasi Fitur
    # ---------------------------------
    st.markdown('<div class="gradient-subheader-tab3">Heatmap Korelasi Fitur (2022)</div>', unsafe_allow_html=True)
    
    # Narasi ringkas sebelum heatmap
    st.caption("Heatmap korelasi digunakan sebagai validasi statistik: apakah hubungan antar variabel sejalan dengan tren scatter plot dan feature importance.")

    features_fullname["Prediksi_IKU"] = "Prediksi Indeks Kualitas Udara"
    corr = df_hot[list(features_fullname.keys())].corr().round(2)   # dibulatkan 2 desimal
    colorscale = [[0.0, "#804000"], [0.5, "#f0f0f0"], [1.0, "#E2CEB1"]]
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale=colorscale)
    col_names = list(features_fullname.keys())
    display_names = [features_fullname[c] for c in col_names]
    fig_corr.update_xaxes(ticktext=display_names, tickvals=list(range(len(display_names))))
    fig_corr.update_yaxes(ticktext=display_names, tickvals=list(range(len(display_names))))
    fig_corr.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=25, l=80, r=20, b=5), height=500) # atur tinggi grafik)
    st.plotly_chart(fig_corr, use_container_width=True)
                
    st.markdown("""
    <div style="margin-top:15px; padding:10px; background-color:#6B4226; 
                color:white; border-radius:10px; font-size:14px;">
        🌳 <b>Indeks Kualitas Tutupan Lahan ↔ IKU (r=+0.74)</b> → validasi kuat bahwa tutupan lahan berperan positif menjaga kualitas udara.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:10px; padding:10px; background-color:#8B5E3C; 
                color:white; border-radius:10px; font-size:14px;">
        🚗 <b>Jumlah Kendaraan Bermotor IKU (r=-0.76)</b> → konsisten menjadi faktor tekanan terbesar terhadap kualitas udara, menguatkan hasil feature importance.
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:10px; padding:10px; background-color:#A47551; 
                color:white; border-radius:10px; font-size:14px;">
        ⚡ <b>Rumah Tangga Listrik PLN IKU (r=-0.42)</b> → korelasi moderat, merepresentasikan dinamika konsumsi energi rumah tangga yang ikut memengaruhi kualitas udara.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:10px; padding:10px; background-color:#C19A6B; 
                color:white; border-radius:10px; font-size:14px;">
        🔥 <b>Luas Karhutla ↔ IKU (r=+0.10)</b> → hubungan lemah secara statistik, wajar karena kejadian karhutla bersifat insidental/episodik dalam data tahunan, tetapi tetap relevan sebagai faktor risiko.
    </div>
    """, unsafe_allow_html=True)

     # Insight utama setelah heatmap
    st.markdown("""
    <div style="margin-top:20px; padding:12px; background-color:#f0f0f0; 
                color:#333; border-radius:10px; font-size:14px; 
                box-shadow: 0 0 5px rgba(0,0,0,0.05);">
        ✨ <b>Insight utama:</b> Heatmap menegaskan konsistensi hubungan antar faktor. 
    <b>Kendaraan</b> dan <b>tutupan lahan</b> muncul paling kuat dan stabil, 
    sedangkan <b>listrik</b> dan <b>karhutla</b> memberi konteks tambahan yang melengkapi analisis. 
    Tidak ada variabel yang terabaikan; masing-masing berkontribusi sesuai karakteristik datanya.
    </div>
    """, unsafe_allow_html=True)