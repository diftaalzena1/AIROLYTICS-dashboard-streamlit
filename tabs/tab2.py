import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.helpers import weighted_hybrid_predict, iku_category

def show_tab():
    # ---------------------------------
    # CSS Styling Global (Gradient Subheader, Box, Caption)
    # ---------------------------------
    st.markdown("""
    <style>
    .gradient-subheader {
        font-size: 26px !important;
        font-weight: 600;
        background: linear-gradient(to right, #734128, #A47551);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        color: transparent;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    .prediction-box {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .input-caption {
        font-size: 0.8rem;
        color: #555;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Judul Tab
    st.markdown('<div class="gradient-subheader">Demo Prediksi Interaktif</div>', unsafe_allow_html=True)

    # Load df_hot.csv & bersihkan data
    try:
        df_hot = pd.read_csv("data/df_hot.csv")
    except FileNotFoundError:
        st.error("❌ File df_hot.csv tidak ditemukan!")
        return

    # Bersihkan angka
    num_cols_float = ["IKTL_(%)", "Rumah_Tangga_Listrik_PLN_(%)", "Indeks_Kualitas_Udara_(%)"]
    for col in num_cols_float:
        df_hot[col] = df_hot[col].astype(str).str.replace(",", ".").astype(float)

    df_hot["Kendaraan_Bermotor"] = df_hot["Kendaraan_Bermotor"].astype(str).str.replace(".", "").astype(float)
    df_hot["Karhutla_(ha)"] = pd.to_numeric(df_hot["Karhutla_(ha)"], errors="coerce")

    # Input Ranges
    input_ranges = {
        "IKTL_(%)": (0.00, 100.00),
        "Karhutla_(ha)": (0.00, 336798.00),
        "Kendaraan_Bermotor": (142862, 22774562),
        "Rumah_Tangga_Listrik_PLN_(%)": (43.14, 100.00)
    }

    st.markdown(
        "Masukkan nilai fitur secara manual. Range data latih ditampilkan di bawah. Nilai di luar range tetap bisa diprediksi, tetapi mungkin kurang akurat."
    )

    # ---------------------------------
    # Input User
    # ---------------------------------
    IKTL = st.number_input("Indeks Kualitas Tutupan Lahan (%)", value=50.0, step=0.1)
    st.markdown(f"<div class='input-caption'>Range data latih: {input_ranges['IKTL_(%)'][0]} – {input_ranges['IKTL_(%)'][1]}</div>", unsafe_allow_html=True)

    Karhutla = st.number_input("Luas Kebakaran Hutan dan Lahan (ha)", value=0.0, step=1.0)
    st.markdown(f"<div class='input-caption'>Range data latih: {input_ranges['Karhutla_(ha)'][0]} – {input_ranges['Karhutla_(ha)'][1]}</div>", unsafe_allow_html=True)

    Kendaraan = st.number_input("Jumlah Kendaraan Bermotor (unit)", value=142862, step=1)
    st.markdown(f"<div class='input-caption'>Range data latih: {input_ranges['Kendaraan_Bermotor'][0]} – {input_ranges['Kendaraan_Bermotor'][1]}</div>", unsafe_allow_html=True)

    Listrik = st.number_input("Persentase Rumah Tangga Listrik PLN (%)", value=50.0, step=0.1)
    st.markdown(f"<div class='input-caption'>Range data latih: {input_ranges['Rumah_Tangga_Listrik_PLN_(%)'][0]} – {input_ranges['Rumah_Tangga_Listrik_PLN_(%)'][1]}</div>", unsafe_allow_html=True)

    # Tombol Prediksi
    if st.button("Prediksi IKU"):
        X_input = pd.Series({
            "IKTL_(%)": IKTL,
            "Karhutla_(ha)": Karhutla,
            "Kendaraan_Bermotor": Kendaraan,
            "Rumah_Tangga_Listrik_PLN_(%)": Listrik
        })
        pred = weighted_hybrid_predict(X_input)
        kategori = iku_category(pred)

        color_map_demo = {
            "Sangat Baik": "#4E342E",
            "Baik": "#6D4C41",
            "Sedang": "#A1887F",
            "Kurang": "#D7CCC8",
            "Sangat Kurang": "#E2CEB1"
        }

        st.markdown(
            f"<h3 style='background-color:{color_map_demo[kategori]};"
            f"color:white;padding:10px;border-radius:5px;'>Prediksi IKU: {pred:.2f}% ({kategori})</h3>",
            unsafe_allow_html=True
        )

        col_labels = {
            "IKTL_(%)": "Indeks Kualitas Tutupan Lahan (%)",
            "Karhutla_(ha)": "Luas Kebakaran Hutan dan Lahan (ha)",
            "Kendaraan_Bermotor": "Jumlah Kendaraan Bermotor (unit)",
            "Rumah_Tangga_Listrik_PLN_(%)": "Persentase Rumah Tangga Listrik PLN (%)"
        }

        # Warning jika input di luar range
        for val, (col, (min_val, max_val)) in zip([IKTL, Karhutla, Kendaraan, Listrik], input_ranges.items()):
            if val < min_val or val > max_val:
                label = col_labels.get(col, col)
                st.warning(f"⚠️ {label} di luar range data latih ({min_val} – {max_val})")

    # ---------------------------------
    # Prediksi vs Data Aktual 2022
    # ---------------------------------
    st.markdown('<div class="gradient-subheader">Prediksi vs Data Aktual 2022 (Hot Test)</div>', unsafe_allow_html=True)

    provinsi_hot = st.selectbox("Pilih Provinsi (2022)", df_hot["Provinsi"])
    df_row = df_hot[df_hot["Provinsi"] == provinsi_hot].iloc[0]
    X_input_hot = df_row[["IKTL_(%)", "Karhutla_(ha)", "Kendaraan_Bermotor", "Rumah_Tangga_Listrik_PLN_(%)"]]
    pred_hot = weighted_hybrid_predict(X_input_hot)
    kategori_hot = iku_category(pred_hot)
    aktual_hot = df_row["Indeks_Kualitas_Udara_(%)"]
    error_hot = abs(pred_hot - aktual_hot)

    st.markdown(
        f"<div class='prediction-box'>"
        f"<h3>{provinsi_hot}</h3>"
        f"<p><b>Prediksi:</b> {pred_hot:.2f}% ({kategori_hot})</p>"
        f"<p><b>Aktual:</b> {aktual_hot:.2f}%</p>"
        f"<p><b>Error Absolut:</b> {error_hot:.2f}%</p>"
        f"</div>",
        unsafe_allow_html=True
    )

    # ---------------------------------
    # Grafik Line: Aktual vs Prediksi + Error Semua Provinsi
    # ---------------------------------
    st.markdown('<div class="gradient-subheader">Prediksi vs Data Aktual 2022 Semua Provinsi (Hot Test)</div>', unsafe_allow_html=True)

    preds, errors = [], []
    for _, row in df_hot.iterrows():
        X_input_hot = row[["IKTL_(%)", "Karhutla_(ha)", "Kendaraan_Bermotor", "Rumah_Tangga_Listrik_PLN_(%)"]]
        pred_val = weighted_hybrid_predict(X_input_hot)
        preds.append(pred_val)
        errors.append(row["Indeks_Kualitas_Udara_(%)"] - pred_val)

    # Buat subplot dengan secondary axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Aktual & Prediksi (y-axis kiri)
    fig.add_trace(
        go.Scatter(
            x=df_hot["Provinsi"], y=df_hot["Indeks_Kualitas_Udara_(%)"],
            mode="lines+markers", name="Aktual", line=dict(color="#734128")
        ), secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=df_hot["Provinsi"], y=preds,
            mode="lines+markers", name="Prediksi", line=dict(color="#A47551", dash="dash")
        ), secondary_y=False
    )

    # Error (y-axis kanan)
    fig.add_trace(
        go.Scatter(
            x=df_hot["Provinsi"], y=errors,
            mode="lines+markers", name="Error (Aktual - Prediksi)",
            line=dict(color="red", dash="dot")
        ), secondary_y=True
    )

    # Layout
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, tickangle=45),
        legend=dict(orientation="h", yanchor="top", y=-0.7, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=40, b=80)
    )

    # Label sumbu
    fig.update_yaxes(title_text="IKU (%)", secondary_y=False)
    fig.update_yaxes(title_text="Error (%)", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)
