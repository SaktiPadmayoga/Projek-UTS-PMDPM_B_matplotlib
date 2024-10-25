import streamlit as st
import pandas as pd
import pickle
import os
from streamlit_option_menu import option_menu

import numpy as np

# Navigasi sidebar
with st.sidebar:
    selected = option_menu('Prediksi Harga Properti',
                           ['Klasifikasi', 'Regresi'], 
                           default_index=0)

# Fungsi untuk memuat model
def load_model():
    with open('BestModel_CLF_gscv_SVM_percentile_matplotlib.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

gscv_SVM_percentile_model = load_model()

def load_model1():
    with open('BestModel_REG_GSCV_RF_matplotlib.pkl', 'rb') as file:
        model1 = pickle.load(file)
    return model1

GSCV_RF_model = load_model1()

# Muat model

# Halaman Klasifikasi
if selected == 'Klasifikasi':
    st.title('Klasifikasi')
    
    # Inputan file dataset CSV
    file = st.file_uploader("Masukkan File", type=["csv", "txt"])
    
    # Input data properti
    squaremeters = st.number_input("Masukkan luas tanah dalam meter persegi", min_value=0)
    numberofrooms = st.number_input("Masukkan jumlah kamar", min_value=0)
    
    # Input untuk kategori yang terpisah
    hasyard_yes = st.selectbox("Memiliki halaman (Ya)", [1, 0])
    hasyard_no = 1 - hasyard_yes
    
    haspool_yes = st.selectbox("Memiliki kolam renang (Ya)", [1, 0])
    haspool_no = 1 - haspool_yes
    
    floors = st.number_input("Masukkan jumlah lantai", min_value=0)
    citycode = st.number_input("Masukkan kode lokasi", min_value=0)
    citypartrange = st.number_input("Masukkan eksklusivitas kawasan", min_value=0)
    numprevowners = st.number_input("Masukkan jumlah pemilik sebelumnya", min_value=0)
    made = st.number_input("Masukkan tahun pembuatan", min_value=0)
    
    isnewbuilt_new = st.selectbox("Bangunan baru (Ya)", [1, 0])
    isnewbuilt_old = 1 - isnewbuilt_new
    
    hasstormprotector_yes = st.selectbox("Memiliki pelindung badai (Ya)", [1, 0])
    hasstormprotector_no = 1 - hasstormprotector_yes
    
    basement = st.number_input("Masukkan luas basement", min_value=0)
    attic = st.number_input("Masukkan luas loteng", min_value=0)
    garage = st.number_input("Masukkan luas garase", min_value=0)
    
    hasstorageroom_yes = st.selectbox("Memiliki gudang (Ya)", [1, 0])
    hasstorageroom_no = 1 - hasstorageroom_yes+1  # Inversi dari hasstorageroom_yes
    
    hasguestroom = st.number_input("Masukkan jumlah ruang tamu", min_value=0)

    # Siapkan data input
    input_data = np.array([[
        squaremeters,
        numberofrooms,
        hasyard_yes,
        hasyard_no,
        haspool_yes,
        haspool_no,
        floors,
        citycode,
        citypartrange,
        numprevowners,
        made,
        isnewbuilt_new,
        isnewbuilt_old,
        hasstormprotector_yes,
        hasstormprotector_no,
        basement,
        attic,
        garage,
        hasstorageroom_yes,
        hasstorageroom_no,
        hasguestroom
    ]])
    
    # Tombol untuk prediksi
    hitung = st.button("Prediksi")
    
    if hitung:
        # Debug info sebelum prediksi
        st.write("Data yang akan diprediksi:", input_data)
        
        # Gunakan model untuk prediksi
        rf_model_prediction = gscv_SVM_percentile_model.predict(input_data)

        # Tampilkan hasil dengan format yang lebih baik
        kategori = rf_model_prediction[0]   
        
        st.write("predik:", kategori)    

        # Tampilkan hasil dengan warna dan format yang lebih baik
        if kategori == "Basic":
            st.success(f"🏠 Properti termasuk kategori Basic")
        elif kategori == "Middle":
            st.warning(f"🏠 Properti termasuk kategori Middle")
        else:
            st.error(f"🏠 Properti termasuk kategori Luxury")



if selected == 'Regresi':
    st.title('Regresi')
    
    # Inputan file dataset CSV
    file = st.file_uploader("Masukkan File", type=["csv", "txt"])
    
    # Input data properti
    squaremeters = st.number_input("Masukkan luas tanah dalam meter persegi", min_value=0)
    numberofrooms = st.number_input("Masukkan jumlah kamar", min_value=0)
    
    # Perbaikan untuk variabel kategorikal
    hasyard_yes = st.selectbox("Memiliki halaman", [0, 1])
    hasyard_no = 1 - hasyard_yes
    
    haspool_yes = st.selectbox("Memiliki kolam renang", [0, 1])
    haspool_no = 1 - haspool_yes
    
    floors = st.number_input("Masukkan jumlah lantai", min_value=0)
    citycode = st.number_input("Masukkan kode lokasi", min_value=0)
    citypartrange = st.number_input("Masukkan eksklusivitas kawasan", min_value=0)
    numprevowners = st.number_input("Masukkan jumlah pemilik sebelumnya", min_value=0)
    made = st.number_input("Masukkan tahun pembuatan", min_value=0)
    
    isnewbuilt_new = st.selectbox("Bangunan baru", [0, 1])
    isnewbuilt_old = 1 - isnewbuilt_new
    
    hasstormprotector_yes = st.selectbox("Memiliki pelindung badai", [0, 1])
    hasstormprotector_no = 1 - hasstormprotector_yes
    
    basement = st.number_input("Masukkan luas basement", min_value=0)
    attic = st.number_input("Masukkan luas loteng", min_value=0)
    garage = st.number_input("Masukkan luas garase", min_value=0)
    
    hasstorageroom_yes = st.selectbox("Memiliki gudang", [0, 1])
    hasstorageroom_no = 1 - hasstorageroom_yes  
    
    hasguestroom = st.number_input("Masukkan jumlah ruang tamu", min_value=0)

    # Siapkan data input
    input_data = np.array([[
        squaremeters,
        numberofrooms,
        hasyard_yes,
        hasyard_no,
        haspool_yes,
        haspool_no,
        floors,
        citycode,
        citypartrange,
        numprevowners,
        made,
        isnewbuilt_new,
        isnewbuilt_old,
        hasstormprotector_yes,
        hasstormprotector_no,
        basement,
        attic,
        garage,
        hasstorageroom_yes,
        hasstorageroom_no,
        hasguestroom
    ]])
    
    # Tombol untuk prediksi
    hitung = st.button("Prediksi")
    
    if hitung:
        try:
            # Gunakan model untuk prediksi
            rf_model_prediction = GSCV_RF_model.predict(input_data)
            # Format hasil prediksi dengan 1 angka di belakang koma
            formatted_prediction = "{:,.1f}".format(rf_model_prediction[0])
            st.success(f"Harga properti yang diprediksi: Rp {formatted_prediction}")
        except Exception as e:
            st.error(f"Terjadi kesalahan dalam prediksi: {str(e)}")
