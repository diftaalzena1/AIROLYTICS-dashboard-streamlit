import streamlit as st 

def show_tab():
    # ---------------------------------
    # CSS Styling (Background, Tabel, Gradient Title & Subheader)
    # ---------------------------------
    st.markdown("""
    <style>
    /* Background halaman */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to right, #E2CEB1, #FDFCE8);
        padding: 10px;
    }

    /* Tabel transparan */
    table {
        width: 100%;
        border-collapse: collapse;
        background-color: transparent;
    }
    th, td {
        border: 1px solid #ccc;
        padding: 10px;
        text-align: left;
        background-color: transparent;
        color: black;
    }
    th {
        font-weight: bold;
        text-align: center;
        color: black;
        background-color: rgba(115, 65, 40, 0.05); /* aksen tipis */
    }

    /* Hover row effect */
    tbody tr:hover {
        background-color: rgba(255,255,255,0.2);
    }

    /* Teks sumber */
    .iku-sumber {
        font-size: 0.85rem;
        font-style: italic;
        color: #333333;
        margin-top: 5px;
        margin-bottom: 20px;
    }

    /* Gradient Title */
    .gradient-title {
        font-size: 42px !important;
        font-weight: 700;
        background: linear-gradient(to right, #734128, #A47551);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        line-height: 1.1;
        margin-bottom: 10px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1); /* efek elegan */
    }

    /* Tagline */
    .tagline {
        font-size: 18px;
        font-style: italic;
        text-align: center;
        color: #555;
        margin-bottom: 25px;
    }

    /* Gradient Subheader */
    .gradient-subheader {
        font-size: 22px !important;
        font-weight: 600;
        background: linear-gradient(to right, #734128, #A47551);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
        line-height: 1.1;
        margin-bottom: 10px;
    }

    /* Responsive untuk mobile */
    @media (max-width: 768px) {
        table, th, td {
            font-size: 0.75rem;  /* lebih kecil agar muat layar */
        }
        .gradient-title {
            font-size: 32px !important;
        }
        .tagline {
            font-size: 16px;
        }
        .gradient-subheader {
            font-size: 16px !important;
        }
        p {
            font-size: 16px !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # ---------------------------------
    # Judul & Deskripsi
    # ---------------------------------
    st.markdown("""
    <div style="padding-top:5px; padding-bottom:15px; text-align:center;">
        <h1 class="gradient-title">AIROLYTICS: Dashboard Prediksi Indeks Kualitas Udara di Indonesia</h1> 
        <div class="tagline">"Smart Insights for Cleaner Air"</div>
        <p style="font-size:18px; color:#333; text-align:center;">
            <b style="color:#734128;">Prediksi kualitas udara</b> berbasis <b style="color:#734128;">Random Forest dengan augmentasi data.</b>  
            Mendukung <i>keputusan berbasis data</i> untuk <b style="color:#734128;">lingkungan & kesehatan masyarakat</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------------
    # Tujuan Proyek
    # ---------------------------------
    st.markdown('<div style="padding-top:5px;"><span class="gradient-subheader">Tujuan Utama Proyek</span></div>', unsafe_allow_html=True)
    st.markdown("""
    - Memproyeksikan Indeks Kualitas Udara (IKU) tiap provinsi berdasarkan data historis.  
    - Mengidentifikasi faktor utama penurunan kualitas udara untuk perumusan kebijakan.  
    - Menyediakan informasi pendukung bagi pemerintah & masyarakat dalam meningkatkan kesadaran lingkungan.
    """)

    # ---------------------------------
    # Klasifikasi IKU (HTML Table)
    # ---------------------------------
    st.markdown('<div style="padding-top:5px;"><span class="gradient-subheader">Klasifikasi Indeks Kualitas Udara (IKU)</span></div>', unsafe_allow_html=True)
    st.markdown("""
    <table>
        <thead>
            <tr>
                <th>Kategori</th>
                <th>Rentang IKU (%)</th>
                <th>Keterangan</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>🟢 Sangat Baik</td>
                <td>90 – 100</td>
                <td>Udara sangat bersih, aman untuk semua orang</td>
            </tr>
            <tr>
                <td>🙂 Baik</td>
                <td>70 – 89.99</td>
                <td>Sehat, aman untuk sebagian besar populasi</td>
            </tr>
            <tr>
                <td>😐 Sedang</td>
                <td>50 – 69.99</td>
                <td>Kualitas mulai menurun, kelompok sensitif perlu waspada</td>
            </tr>
            <tr>
                <td>⚠️ Kurang</td>
                <td>25 – 49.99</td>
                <td>Risiko kesehatan meningkat, perlu pengendalian polusi</td>
            </tr>
            <tr>
                <td>❌ Sangat Kurang</td>
                <td>0 – 24.99</td>
                <td>Kualitas buruk sekali, berisiko tinggi bagi semua orang</td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<div class="iku-sumber">Sumber: Portal Satu Data Indonesia</div>', unsafe_allow_html=True)

    # ---------------------------------
    # Call to Action / Penutup
    # ---------------------------------
    st.markdown("""
    <div style="margin-top:15px; padding:15px; background-color:#6B4226; color:white; border-radius:10px; text-align:center; font-size:18px;">
        🌍 Mari gunakan data untuk membangun lingkungan yang lebih sehat. Setelah itu, jelajahi tab berikutnya untuk melakukan demo prediksi dan melihat visualisasi interaktif. Dengan cara ini, kita dapat lebih memahami kondisi udara sehingga bersama-sama kita bisa mewujudkan udara yang lebih bersih dan berkelanjutan. 💡
    </div>
    """, unsafe_allow_html=True)
