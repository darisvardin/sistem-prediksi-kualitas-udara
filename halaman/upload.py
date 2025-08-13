# pages/upload.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
import os
import joblib
import io

# Folder untuk menyimpan file
UPLOAD_DIR = "upload"
FILE_DIR = "file"
DATA_FILE = os.path.join(UPLOAD_DIR, "persistent_data.csv")
SCALER_FILE = os.path.join(FILE_DIR, "scaler.pkl")

def normalize_data(df):
    """Melakukan normalisasi data menggunakan MinMaxScaler."""
    scaler = MinMaxScaler()
    numeric_cols = ['CO (ppm)', 'PM10 (µg/m3)', 'NO2 (ppb)', 'Suhu (°C)', 'Kelembaban (%)', 'Kecepatan Angin (m/s)']
    df_normalized = df.copy()
    
    # Memastikan hanya kolom numerik yang dinormalisasi
    df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df_normalized, scaler

def show():
    st.title("📤 Upload dan Normalisasi Data")
    
    st.markdown("""
    <div class="card">
        <h4>Panduan Upload Data</h4>
        <p>Unggah data pemantauan kualitas udara dalam format CSV dengan kolom berikut:</p>
        <ul>
            <li><code>CO (ppm)</code></li>
            <li><code>PM10 (µg/m3)</code></li>
            <li><code>NO2 (ppb)</code></li>
            <li><code>Suhu (°C)</code></li>
            <li><code>Kelembaban (%)</code></li>
            <li><code>Kecepatan Angin (m/s)</code></li>
            <li><code>Kategori Kualitas Udara</code></li>
        </ul>
        <p>Sistem akan melakukan normalisasi data (skala 0-1) untuk mempersiapkan analisis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Inisialisasi session state jika belum ada
    if 'normalized_data' not in st.session_state:
        st.session_state.normalized_data = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False

    # Logika baru: Muat data dari file jika ada dan session state kosong
    if st.session_state.normalized_data is None and os.path.exists(DATA_FILE):
        st.info("✅ Data ditemukan di server. Memuat data secara otomatis...")
        try:
            df_loaded = pd.read_csv(DATA_FILE)
            st.session_state.normalized_data = df_loaded
            
            # Latih ulang scaler untuk data yang dimuat
            X_loaded = df_loaded.drop('Kategori Kualitas Udara', axis=1, errors='ignore')
            scaler = MinMaxScaler()
            scaler.fit(X_loaded)
            st.session_state.scaler = scaler

            st.success("🎉 Data berhasil dimuat dari file! Anda bisa melanjutkan ke halaman lain.")
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memuat data dari file: {e}")
            st.session_state.normalized_data = None
    
    if st.session_state.normalized_data is None:
        uploaded_file = st.file_uploader("Pilih file CSV", type="csv", key="uploader")
        
        if uploaded_file is not None:
            try:
                # --- LOGIKA PERBAIKAN: Membaca file langsung dari buffer tanpa menyimpan file asli ---
                df = pd.read_csv(io.BytesIO(uploaded_file.getbuffer()))

                required_cols = ['CO (ppm)', 'PM10 (µg/m3)', 'NO2 (ppb)', 'Suhu (°C)', 
                                 'Kelembaban (%)', 'Kecepatan Angin (m/s)', 'Kategori Kualitas Udara']
                
                if all(col in df.columns for col in required_cols):
                    with st.spinner('🔄 Sedang memproses dan menormalisasi data...'):
                        time.sleep(1) 
                        df_normalized, scaler = normalize_data(df)
                        
                        st.session_state.normalized_data = df_normalized
                        st.session_state.scaler = scaler
                        st.session_state.model_trained = False
                        
                        # Simpan data yang dinormalisasi ke file persistent
                        if not os.path.exists(UPLOAD_DIR):
                            os.makedirs(UPLOAD_DIR)
                        df_normalized.to_csv(DATA_FILE, index=False)
                        
                        # Simpan scaler ke file
                        if not os.path.exists(FILE_DIR):
                            os.makedirs(FILE_DIR)
                        joblib.dump(scaler, SCALER_FILE)
                        st.success("✅ Scaler telah disimpan ke file!")
                        
                    st.success(f"🎉 Data berhasil diunggah dan disimpan! **{len(df_normalized)} baris** data siap dianalisis.")
                    st.rerun() 

                else:
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    st.error(f"""
                        ❌ File CSV tidak memiliki kolom yang sesuai.
                        Kolom yang harus diunggah: `{', '.join(required_cols)}`
                        Kolom yang hilang dari file Anda: `{', '.join(missing_cols)}`
                    """)
                    
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {str(e)}")
    
    if st.session_state.normalized_data is not None:
        st.markdown("---")
        
        with st.expander("🔍 Lihat Data yang Sudah Diunggah", expanded=False):
            st.dataframe(st.session_state.normalized_data.head(10))
            st.info(f"Menampilkan 10 dari {len(st.session_state.normalized_data)} baris data")
        
        col1, col2 = st.columns(2)
        with col1:
            csv = st.session_state.normalized_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Unduh Data Ternormalisasi",
                data=csv,
                file_name='data_normalisasi.csv',
                mime='text/csv',
                use_container_width=True
            )
        with col2:
            if st.button("🗑️ Hapus Semua Data", use_container_width=True):
                # Hapus file data dan reset session state
                if os.path.exists(DATA_FILE):
                    os.remove(DATA_FILE)
                if os.path.exists(SCALER_FILE):
                    os.remove(SCALER_FILE)
                for key in ['normalized_data', 'scaler', 'model', 'label_encoder', 'model_trained']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("Data dan model berhasil dihapus!")
                st.rerun()

        st.subheader("📈 Statistik Data")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Distribusi Kategori Kualitas Udara**")
            category_counts = st.session_state.normalized_data['Kategori Kualitas Udara'].value_counts()
            st.bar_chart(category_counts)
        
        with col2:
            st.markdown("**Statistik Deskriptif**")
            st.dataframe(st.session_state.normalized_data.describe().T)
        
        st.markdown(f"""
        <div class="primary-box">
            <h4>Informasi Data:</h4>
            <ul>
                <li><strong>Jumlah sampel:</strong> {len(st.session_state.normalized_data)}</li>
                <li><strong>Jumlah fitur:</strong> {len(st.session_state.normalized_data.columns)-1}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.info("ℹ️ Silakan unggah file CSV untuk memulai proses normalisasi data.")
