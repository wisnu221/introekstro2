# app.py

import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Fungsi untuk memuat model dan objek lainnya
def load_model_assets():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('le.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('columns.pkl', 'rb') as f:
        columns = pickle.load(f)
    return model, le, columns

# Muat aset saat aplikasi dimulai
model, le, columns = load_model_assets()

# Judul dan deskripsi aplikasi
st.set_page_config(page_title="Prediktor Kepribadian", page_icon="🧠")
st.title("🧠 Aplikasi Prediksi Kepribadian")
st.write(
    "Aplikasi ini menggunakan Machine Learning untuk memprediksi kepribadian seseorang "
    "(Introvert atau Extrovert) berdasarkan beberapa kebiasaan sosial. "
    "Silakan isi parameter di bawah ini."
)

st.divider()

# Membuat form input di sidebar
st.sidebar.header("Masukkan Data Anda:")

def user_input_features():
    # Slider dan selectbox untuk input pengguna
    time_spent_alone = st.sidebar.slider('Waktu yang dihabiskan sendirian (jam/hari)', 0.0, 10.0, 5.0)
    stage_fear = st.sidebar.selectbox('Apakah Anda takut panggung?', ('Tidak', 'Ya'))
    social_event_attendance = st.sidebar.slider('Kehadiran di acara sosial (per bulan)', 0.0, 10.0, 5.0)
    going_outside = st.sidebar.slider('Frekuensi pergi keluar (per minggu)', 0.0, 7.0, 3.0)
    drained_after_socializing = st.sidebar.selectbox('Apakah Anda merasa lelah setelah bersosialisasi?', ('Tidak', 'Ya'))
    friends_circle_size = st.sidebar.slider('Ukuran lingkaran pertemanan', 0.0, 20.0, 8.0)
    post_frequency = st.sidebar.slider('Frekuensi posting di media sosial (per minggu)', 0.0, 10.0, 4.0)
    
    # Mapping input kategori ke numerik
    stage_fear_num = 1 if stage_fear == 'Ya' else 0
    drained_after_socializing_num = 1 if drained_after_socializing == 'Ya' else 0

    # Membuat dataframe dari input
    data = {
        'Time_spent_Alone': time_spent_alone,
        'Stage_fear': stage_fear_num,
        'Social_event_attendance': social_event_attendance,
        'Going_outside': going_outside,
        'Drained_after_socializing': drained_after_socializing_num,
        'Friends_circle_size': friends_circle_size,
        'Post_frequency': post_frequency
    }
    
    # Pastikan urutan kolom sesuai dengan saat training
    features = pd.DataFrame(data, index=[0])[columns]
    return features

input_df = user_input_features()

# Menampilkan data input yang dimasukkan pengguna di halaman utama
st.header("Data yang Anda Masukkan:")
st.table(input_df)


# Tombol untuk melakukan prediksi
if st.sidebar.button('Prediksi Kepribadian Saya'):
    # Melakukan prediksi
    prediction = model.predict(input_df)
    
    # Mengubah hasil prediksi kembali ke label asli (Introvert/Extrovert)
    prediction_label = le.inverse_transform(prediction)
    
    # Menampilkan hasil
    st.divider()
    st.subheader("🎉 Hasil Prediksi:")
    
    if prediction_label[0] == 'Introvert':
        st.success(f"Anda cenderung seorang **{prediction_label[0]}**.")
        st.write("Anda mungkin lebih suka menghabiskan waktu sendirian atau dalam kelompok kecil, dan merasa lebih berenergi saat menyendiri.")
    else:
        st.info(f"Anda cenderung seorang **{prediction_label[0]}**.")
        st.write("Anda kemungkinan besar menikmati interaksi sosial yang aktif, bertemu orang baru, dan merasa bersemangat di tengah keramaian.")
