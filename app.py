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
# @st.cache_resource memastikan ini hanya dijalankan sekali
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
    st.error("File model tidak ditemukan. Pastikan Anda sudah menjalankan `train_model.py` terlebih dahulu.")
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
# ---------------------------------
st.title("🔮 Prediktor Kepribadian: Introvert atau Extrovert?")
st.write(
    "Jawab beberapa pertanyaan di bawah ini untuk melihat prediksi kepribadian Anda. "
    "Geser slider dan pilih opsi yang paling sesuai dengan diri Anda."
)
st.divider()


# ---------------------------------
# Form Input
# ---------------------------------
# Menggunakan st.form untuk mengelompokkan input dan tombol
with st.form(key='prediction_form'):
    st.subheader("📝 Silakan isi survei singkat ini:")
    
    # Membuat layout dengan 2 kolom
    col1, col2 = st.columns(2, gap="large")

    with col1:
        # Input numerik dengan slider
        time_spent_alone = st.slider(
            '⏰ Waktu yang dihabiskan sendirian (jam/hari)', 
            min_value=0.0, max_value=10.0, value=5.0, step=0.5,
            help="Rata-rata berapa jam Anda habiskan sendirian setiap hari?"
        )
        social_event_attendance = st.slider(
            '🎉 Kehadiran di acara sosial (per bulan)', 
            min_value=0.0, max_value=10.0, value=4.0, step=1.0,
            help="Seberapa sering Anda menghadiri acara sosial seperti pesta atau pertemuan besar?"
        )
        friends_circle_size = st.slider(
            '👥 Ukuran lingkaran pertemanan', 
            min_value=0.0, max_value=20.0, value=8.0, step=1.0,
            help="Berapa banyak teman dekat yang Anda miliki?"
        )

    with col2:
        # Input kategori dengan selectbox/radio
        stage_fear = st.radio(
            '🎤 Apakah Anda takut panggung?', 
            ('Tidak', 'Ya'), horizontal=True,
            help="Apakah Anda merasa cemas saat harus berbicara di depan umum?"
        )
        drained_after_socializing = st.radio(
            '🔋 Apakah Anda merasa lelah setelah bersosialisasi?', 
            ('Tidak', 'Ya'), horizontal=True,
            help="Apakah Anda butuh waktu menyendiri untuk 'mengisi ulang energi' setelah berinteraksi sosial?"
        )
        post_frequency = st.slider(
            '📱 Frekuensi posting di media sosial (per minggu)', 
            min_value=0.0, max_value=10.0, value=3.0, step=1.0,
            help="Seberapa sering Anda membuat postingan baru di media sosial?"
        )
        # Menambahkan input yang sebelumnya tidak ada di UI (Going_outside)
        going_outside = st.slider(
            '🚶 Frekuensi pergi keluar rumah (per minggu)', 
            min_value=0.0, max_value=7.0, value=3.0, step=1.0,
            help="Seberapa sering Anda pergi keluar rumah untuk aktivitas non-wajib?"
        )

    # Tombol submit di dalam form
    st.divider
