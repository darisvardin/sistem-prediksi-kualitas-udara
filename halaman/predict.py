# pages/predict.py
import streamlit as st
import pandas as pd
import joblib
import os
import time

# Nama file tempat model dan scaler disimpan
MODEL_FILE = "file/c45_model.pkl"
SCALER_FILE = "file/scaler.pkl"

def get_form_values():
    """
    Fungsi untuk mendapatkan nilai input formulir dari session state.
    Jika belum ada, gunakan nilai default.
    """
    # Mengambil nilai dari session_state, atau nilai default jika belum ada
    co = st.session_state.get('last_co', 0.5)
    pm10 = st.session_state.get('last_pm10', 50.0)
    no2 = st.session_state.get('last_no2', 20.0)
    suhu = st.session_state.get('last_suhu', 25.0)
    kelembaban = st.session_state.get('last_kelembaban', 60.0)
    kecepatan_angin = st.session_state.get('last_kecepatan_angin', 2.0)
    return co, pm10, no2, suhu, kelembaban, kecepatan_angin

def show():
    st.title("🔮 Prediksi Kualitas Udara")
    st.markdown("""
    <div class="card">
        <h4>Simulasi Prediksi Kualitas Udara</h4>
        <p>Masukkan parameter lingkungan untuk memprediksi kategori kualitas udara saat ini.</p>
        <p>Sistem menggunakan model Decision Tree yang telah dilatih sebelumnya.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- LOGIKA PERBAIKAN: Muat model dan scaler jika belum ada di session state ---
    model = None
    scaler = None
    
    # Nama-nama fitur yang benar sesuai data pelatihan Anda
    feature_names = ['CO (ppm)', 'PM10 (µg/m3)', 'NO2 (ppb)', 'Suhu (°C)', 'Kelembaban (%)', 'Kecepatan Angin (m/s)']
    # Nama-nama kelas hasil prediksi
    class_names = ['Baik', 'Sedang', 'Tidak Sehat', 'Sangat Tidak Sehat', 'Berbahaya']
    
    # Prioritas 1: Coba muat dari session state (paling cepat)
    if 'model' in st.session_state and 'scaler' in st.session_state:
        model = st.session_state.model
        scaler = st.session_state.scaler
        st.session_state.feature_names = feature_names # Memastikan nama fitur konsisten
        st.session_state.class_names = class_names
        
    # Prioritas 2: Jika tidak ada di session state, coba muat dari file (fallback)
    elif os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            scaler = joblib.load(SCALER_FILE)
            st.session_state.model = model
            st.session_state.scaler = scaler
            # Simpan nama kelas dan fitur untuk konsistensi
            st.session_state.class_names = class_names
            st.session_state.feature_names = feature_names
            st.success("✅ Model dan Scaler berhasil dimuat dari file. Anda bisa melakukan prediksi.")
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memuat model dari file: {e}")
            return
    else:
        st.warning("⚠️ Model prediksi belum tersedia. Silakan latih model terlebih dahulu di halaman 'Penerapan Algoritma C4.5'.")
        return
    # --- AKHIR LOGIKA PERBAIKAN ---
    
    # Ambil nilai awal untuk form
    co_val, pm10_val, no2_val, suhu_val, kelembaban_val, kecepatan_angin_val = get_form_values()
    
    # Form input
    with st.form("prediction_form"):
        st.subheader("🌡️ Parameter Lingkungan")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Polutan Udara**")
            co = st.number_input("CO (ppm)", min_value=0.0, max_value=10.0, value=co_val, step=0.5, 
                                 help="Karbon Monoksida - Gas beracun dari pembakaran tidak sempurna")
            pm10 = st.number_input("PM10 (µg/m3)", min_value=0.0, max_value=200.0, value=pm10_val, step=5.0, 
                                   help="Partikel debu halus - Partikel udara berdiameter ≤10 mikrometer")
            no2 = st.number_input("NO2 (ppb)", min_value=0.0, max_value=100.0, value=no2_val, step=5.0, 
                                 help="Nitrogen Dioksida - Gas beracun dari kendaraan bermotor dan industri")
            
        with col2:
            st.markdown("**Kondisi Cuaca**")
            suhu = st.number_input("Suhu (°C)", min_value=0.0, max_value=50.0, value=suhu_val, step=0.5, 
                                   help="Temperatur udara - Suhu lingkungan saat ini")
            kelembaban = st.number_input("Kelembaban (%)", min_value=0.0, max_value=100.0, value=kelembaban_val, step=1.0, 
                                         help="Tingkat kelembaban udara - Persentase uap air di udara")
            kecepatan_angin = st.number_input("Kecepatan Angin (m/s)", min_value=0.0, max_value=20.0, value=kecepatan_angin_val, step=0.1, 
                                             help="Kecepatan angin - Memengaruhi penyebaran polutan")
        
        submitted = st.form_submit_button("🔍 Prediksi Kualitas Udara", use_container_width=True)
        
    if submitted and model is not None and scaler is not None:
        # --- LOGIKA PERBAIKAN: Menyimpan nilai input ke session_state ---
        st.session_state['last_co'] = co
        st.session_state['last_pm10'] = pm10
        st.session_state['last_no2'] = no2
        st.session_state['last_suhu'] = suhu
        st.session_state['last_kelembaban'] = kelembaban
        st.session_state['last_kecepatan_angin'] = kecepatan_angin
        
        # --- LOGIKA PERBAIKAN: Periksa apakah semua nilai input adalah nol ---
        input_values = [co, pm10, no2, suhu, kelembaban, kecepatan_angin]
        if all(v == 0 for v in input_values):
            st.error("Semua nilai tidak boleh 0. Silakan masukkan parameter yang valid untuk prediksi.")
            return # Menghentikan eksekusi jika semua nilai adalah nol
        
        # Buat dataframe input dengan nama kolom yang benar
        input_data = pd.DataFrame([[co, pm10, no2, suhu, kelembaban, kecepatan_angin]], columns=feature_names) 
        
        # Normalize input menggunakan scaler yang sudah dimuat
        input_normalized = scaler.transform(input_data)
        
        # Make prediction
        prediction_index = model.predict(input_normalized)[0]
        prediction_label = class_names[prediction_index]
        
        # Show prediction result
        st.subheader("📊 Hasil Prediksi")
        
        # Display with appropriate styling
        color_map = {
            "Baik": "good-air", "Sedang": "moderate-air", "Tidak Sehat": "bad-air", 
            "Sangat Tidak Sehat": "critical-air", "Berbahaya": "dangerous-air"
        }
        css_class = color_map.get(prediction_label, '')

        st.markdown(f"""
        <div class="prediction-result {css_class}">
            <h2>Kualitas Udara: {prediction_label.upper()}</h2>
            <p>{ 'Kondisi udara sehat untuk semua aktivitas' if prediction_label == 'Baik' else 
                         'Kelompok sensitif mungkin mengalami efek kesehatan' if prediction_label == 'Sedang' else
                         'Semua kelompok mungkin mengalami efek kesehatan' if prediction_label == 'Tidak Sehat' else
                         'Peringatan kesehatan darurat untuk seluruh populasi' if prediction_label == 'Sangat Tidak Sehat' else
                         'Kondisi udara sangat berbahaya, evakuasi disarankan' if prediction_label == 'Berbahaya' else '' }
            </p>
        </div>
        """, unsafe_allow_html=True)

        recommendation_map = {
            "Baik": "success-box", "Sedang": "warning-box", "Tidak Sehat": "danger-box", 
            "Sangat Tidak Sehat": "danger-box", "Berbahaya": "critical-box"
        }
        rec_class = recommendation_map.get(prediction_label, '')
        
        recommendations = {
            "Baik": "<li>Aktivitas luar ruangan dapat dilakukan dengan aman</li><li>Tidak ada pembatasan untuk kelompok sensitif</li><li>Pertahankan kualitas udara dengan mengurangi emisi</li><li>Nikmati udara segar dan aktivitas luar ruangan</li>",
            "Sedang": "<li>Kelompok sensitif (anak, lansia, penderita ISPA) batasi aktivitas luar</li><li>Masyarakat umum masih dapat beraktivitas normal</li><li>Pantau perkembangan kualitas udara secara berkala</li><li>Hindari area dengan polusi tinggi seperti jalan raya padat</li>",
            "Tidak Sehat": "<li>Hindari aktivitas fisik berat di luar ruangan</li><li>Kelompok sensitif tetap di dalam ruangan</li><li>Gunakan masker N95 jika harus keluar ruangan</li><li>Tutup jendela dan pintu untuk mengurangi paparan polusi</li><li>Cari sumber polusi dan lakukan upaya pengurangan</li>",
            "Sangat Tidak Sehat": "<li>Hindari semua aktivitas di luar ruangan</li><li>Tetap di dalam ruangan dengan ventilasi tertutup</li><li>Gunakan air purifier jika memungkinkan</li><li>Pantau informasi kualitas udara terbaru</li><li>Laporkan ke pihak berwenang untuk tindakan darurat</li>",
            "Berbahaya": "<li>Jangan keluar ruangan dalam keadaan apapun</li><li>Gunakan alat bantu pernapasan jika tersedia</li><li>Segera cari tempat dengan udara bersih</li><li>Laporkan ke pihak berwenang untuk tindakan darurat</li><li>Evakuasi ke area dengan kualitas udara lebih baik</li>"
        }
        rec_title = "✅ Rekomendasi:" if prediction_label == "Baik" else "ℹ️ Rekomendasi:" if prediction_label == "Sedang" else "❗ Rekomendasi:" if prediction_label == "Tidak Sehat" else "🚨 Rekomendasi:" if prediction_label == "Sangat Tidak Sehat" else "🔥 Rekomendasi Darurat:"

        st.markdown(f"""
        <div class="recommendation-box {rec_class}">
            <h4>{rec_title}</h4>
            <ul>
                {recommendations.get(prediction_label, '')}
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Show input parameters
        st.subheader("⚙️ Parameter Input yang Digunakan")
        st.dataframe(input_data.style.format("{:.2f}"))
        
        st.info("""
        **Keterangan:**
        - CO: Karbon Monoksida (ppm)
        - PM10: Partikulat halus (µg/m³)
        - NO2: Nitrogen Dioksida (ppb)
        - Suhu dalam derajat Celsius (°C)
        - Kelembaban dalam persentase (%)
        - Kecepatan angin dalam meter per detik (m/s)
        """)
