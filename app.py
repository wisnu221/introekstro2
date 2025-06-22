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
# INI ADALAH BAGIAN KUNCI PERBAIKANNYA
with st.form(key='prediction_form'): # <-- Form dimulai di sini
    st.subheader("📝 Silakan isi survei singkat ini:")
    
    col1, col2 = st.columns(2, gap="large")

    with col1:
        time_spent_alone = st.slider('⏰ Waktu yang dihabiskan sendirian (jam/hari)', 0.0, 10.0, 5.0, 0.5)
        social_event_attendance = st.slider('🎉 Kehadiran di acara sosial (per bulan)', 0.0, 10.0, 4.0, 1.0)
        friends_circle_size = st.slider('👥 Ukuran lingkaran pertemanan', 0.0, 20.0, 8.0, 1.0)

    with col2:
        stage_fear = st.radio('🎤 Apakah Anda takut panggung?', ('Tidak', 'Ya'), horizontal=True)
        drained_after_socializing = st.radio('🔋 Apakah Anda merasa lelah setelah bersosialisasi?', ('Tidak', 'Ya'), horizontal=True)
        post_frequency = st.slider('📱 Frekuensi posting di media sosial (per minggu)', 0.0, 10.0, 3.0, 1.0)
        going_outside = st.slider('🚶 Frekuensi pergi keluar rumah (per minggu)', 0.0, 7.0, 3.0, 1.0)

    st.divider()
    
    # Tombol submit WAJIB berada di dalam blok 'with st.form'
    submit_button = st.form_submit_button(
        label='✨ Prediksi Kepribadian Saya!', 
        use_container_width=True
    )
# <-- Form berakhir di sini


# ---------------------------------
# Logika Prediksi dan Tampilan Hasil
# ---------------------------------
if submit_button:
    # Logika ini hanya akan berjalan SETELAH tombol di dalam form ditekan
    with st.spinner('Menganalisis jawaban Anda...'):
        time.sleep(1)

        stage_fear_num = 1 if stage_fear == 'Ya' else 0
        drained_after_socializing_num = 1 if drained_after_socializing == 'Ya' else 0

        data = {
            'Time_spent_Alone': time_spent_alone,
            'Stage_fear': stage_fear_num,
            'Social_event_attendance': social_event_attendance,
            'Going_outside': going_outside,
            'Drained_after_socializing': drained_after_socializing_num,
            'Friends_circle_size': friends_circle_size,
            'Post_frequency': post_frequency
        }
        
        input_df = pd.DataFrame(data, index=[0])[columns]
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        prediction_label = le.inverse_transform(prediction)[0]
        
    st.subheader("🎉 Hasil Analisis Kepribadian Anda:", anchor=False)

    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        if prediction_label == 'Introvert':
            st.image("https://em-content.zobj.net/source/apple/354/person-in-lotus-position_1f9d8.png", use_column_width=True)
        else:
            st.image("https://em-content.zobj.net/source/apple/354/partying-face_1f973.png", use_column_width=True)
    with res_col2:
        confidence = prediction_proba[0][prediction[0]]
        st.metric(label="Anda Cenderung Seorang...", value=prediction_label, delta=f"{confidence:.0%} Keyakinan")
        if prediction_label == 'Introvert':
            st.write("Anda kemungkinan besar **mendapatkan energi dari waktu menyendiri** dan lebih suka interaksi yang mendalam dengan beberapa teman dekat.")
        else:
            st.write("Anda kemungkinan besar **mendapatkan energi dari interaksi sosial**, menikmati bertemu orang baru, dan merasa bersemangat di lingkungan yang aktif.")

    with st.expander("Lihat detail data yang Anda masukkan"):
        st.table(input_df)
