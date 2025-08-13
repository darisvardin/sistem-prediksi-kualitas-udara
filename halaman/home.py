# pages/home.py
import streamlit as st

def show():
    st.markdown("""<h1 style='text-align: center;'>Sistem Prediksi Kategori Kualitas Udara</h1>""", unsafe_allow_html=True)
    st.markdown("""
    <div class="header-gradient">
        <h2 style="color:white; text-align:center">Prediksi Kategori Kualitas Udara dengan Algoritma C4.5</h2>
        <p style="color:white; text-align:center; font-size:1.2rem">
            Sistem ini membantu memprediksi kategori kualitas udara berdasarkan parameter lingkungan 
            menggunakan algoritma Decision Tree C4.5
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("📋 Tentang Sistem")
    st.markdown("""
    Sistem Prediksi Kategori Kualitas Udara adalah sistem berbasis web yang dapat:
    - Memprediksi kualitas udara (Baik, Sedang, Tidak Sehat, Sangat Tidak Sehat, Berbahaya)
    - Menganalisis pengaruh parameter lingkungan terhadap kualitas udara
    - Memberikan rekomendasi berdasarkan hasil prediksi
    - Memvisualisasikan proses pengambilan keputusan dengan pohon keputusan
    """)
    
    st.subheader("✨ Fitur Utama")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card feature-card">
            <h4>📤 Upload Data</h4>
            <p>Unggah dan normalisasi data pemantauan kualitas udara</p>
            <img src="https://cdn-icons-png.flaticon.com/512/1239/1239718.png" width="100">
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card feature-card">
            <h4>🌳 Analisis Algoritma</h4>
            <p>Penerapan algoritma C4.5 untuk membangun model prediksi</p>
            <img src="https://cdn-icons-png.flaticon.com/512/2103/2103793.png" width="100">
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card feature-card">
            <h4>🔮 Prediksi Real-time</h4>
            <p>Prediksi kualitas udara berdasarkan parameter input</p>
            <img src="https://cdn-icons-png.flaticon.com/512/1995/1995450.png" width="100">
        </div>
        """, unsafe_allow_html=True)
    
    st.subheader("🔧 Cara Kerja Sistem")
    st.markdown("""
    Sistem bekerja melalui 3 tahap utama:
    1. **Upload Data**: Mengunggah data pemantauan lingkungan dalam format CSV
    2. **Pemrosesan Data**: Sistem akan melakukan normalisasi data (skala 0-1)
    3. **Analisis Algoritma**: Penerapan algoritma C4.5 untuk membangun model
    4. **Prediksi**: Memasukkan parameter lingkungan untuk mendapatkan prediksi kualitas udara
    """)
    
    # Diagram alur sistem dengan HTML/CSS
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="card flow-step">
            <h4>Upload Data</h4>
            <p>Mengunggah data pemantauan lingkungan</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card flow-step">
            <h4>Pemrosesan Data</h4>
            <p>Normalisasi data (skala 0-1)</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card flow-step">
            <h4>Analisis Algoritma</h4>
            <p>Penerapan algoritma C4.5 untuk membangun model</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="card flow-step">
            <h4>Prediksi</h4>
            <p>Prediksi kualitas udara berdasarkan parameter input</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.subheader("💡 Manfaat Penggunaan")
    st.markdown("""
    - Memantau kualitas udara secara real-time
    - Mengidentifikasi faktor dominan yang mempengaruhi polusi udara
    - Memberikan peringatan dini kualitas udara buruk
    - Mendukung pengambilan keputusan untuk kebijakan lingkungan
    - Memvisualisasikan hubungan parameter lingkungan dengan kualitas udara
    """)
    
    st.markdown("---")
    st.info("""
    **Panduan Memulai:**
    1. Pilih **Upload Data** di menu navigasi untuk mengunggah data pemantauan
    2. Lanjutkan ke **Penerapan Algoritma C4.5** untuk membangun model prediksi
    3. Gunakan **Prediksi Kualitas Udara** untuk simulasi real-time
    """)