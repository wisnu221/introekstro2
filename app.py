# app.py

import streamlit as st
import pickle
import pandas as pd
import time

# ---------------------------------
# Konfigurasi Halaman dan Judul
# ---------------------------------
st.set_page_config(
    page_title="Prediktor Kepribadian 🧠",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fungsi untuk memuat aset (model, encoder, dll.)
@st.cache_resource
def load_model_assets():
    """Memuat semua aset yang diperlukan untuk prediksi."""
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('le.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('columns.pkl', 'rb') as f:
        columns = pickle.load(f)
    return model, le, columns

# Muat aset
try:
    model, le, columns = load_model_assets()
except FileNotFoundError:
    st.error("File model tidak ditemukan. Pastikan Anda sudah menjalankan `train_model.py` atau mengunduh file .pkl dari Colab.")
    st.stop()


# ---------------------------------
# Sidebar
# ---------------------------------
with st.sidebar:
    st.image("https://em-content.zobj.net/source/apple/354/brain_1f9e0.png", width=100)
    st.title("Tentang Aplikasi")
    st.info(
        "Aplikasi ini menggunakan model Machine Learning untuk memprediksi "
        "apakah seseorang cenderung **Introvert** atau **Extrovert**."
        "\n\n"
        "Data yang digunakan untuk melatih model ini adalah dataset survei "
        "kebiasaan sosial."
    )
    st.warning("**Disclaimer:** Hasil prediksi ini hanyalah untuk tujuan edukasi dan hiburan, bukan diagnosis psikologis.")


# ---------------------------------
# Konten Utama
