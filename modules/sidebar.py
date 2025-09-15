import streamlit as st 

def show_sidebar():
    # ==============================
    # CSS Sidebar
    # ==============================
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        color: white !important;
        background: linear-gradient(to right, #734128, #A47551);
        padding: 0px !important; 
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    .sidebar-header {
        text-align: left;
        color: #FFDDC1;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .logo-box {
        width: 100%;
        padding: 20px 10px;
        text-align: center;
    }

    /* ===== Core Fix: rapatkan semua komponen ===== */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.7rem !important;
        padding: 0rem !important;
        margin: 0rem !important;
    }
    [data-testid="stSidebar"] h3 {
        margin-top: 0.4rem !important;
        margin-bottom: 0.3rem !important;
    }
    [data-testid="stSidebar"] p {
        margin-block-start: 0.2rem !important;
        margin-block-end: 0.2rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stAlert"] {
        padding: 0.4rem !important;
        margin: 0.3rem 0rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ==============================
    # Sidebar content
    # ==============================
    with st.sidebar:
        st.markdown('<div class="logo-box">', unsafe_allow_html=True)
        st.image("assets/logo.png", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### 🎯 Fokus SDGs:")
        cols = st.columns(3)
        with cols[0]: st.image("assets/SDG_3.png", use_container_width=True)
        with cols[1]: st.image("assets/SDG_11.png", use_container_width=True)
        with cols[2]: st.image("assets/SDG_13.png", use_container_width=True)

        st.markdown("### 📌 Sumber Data:")
        st.markdown("""
- **BPS** – Data Statistik Resmi  
- **BPS** – Publikasi Lingkungan Hidup Indonesia  
- **KLHK** – Direktorat Pengendalian Karhutla
""")

        st.markdown("### ℹ️ Fakta:")
        st.info("Menurut WHO, polusi udara menyebabkan **7 juta kematian global / tahun** ☠️")

        st.markdown("### 👥 Tim")
        st.markdown("- Friza Nur Fatmala (23083010051)")
        st.markdown("- Difta Alzena Sakhi (23083010061)")

        st.markdown("### 🏫  Instansi:")
        st.markdown("Universitas Pembangunan Nasional Veteran Jawa Timur")
