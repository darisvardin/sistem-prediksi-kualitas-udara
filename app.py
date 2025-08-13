# app.py
import streamlit as st
import halaman.home
import halaman.upload
import halaman.c45_model
import halaman.predict

# Load CSS
def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Session state initialization
def init_session_state():
    # Inisialisasi semua state yang diperlukan
    if 'page' not in st.session_state:
        st.session_state.page = "Beranda"
    if 'normalized_data' not in st.session_state:
        st.session_state.normalized_data = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'label_encoder' not in st.session_state:
        st.session_state.label_encoder = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = None
    if 'class_names' not in st.session_state:
        st.session_state.class_names = None

# Function to change page
def change_page(page_name):
    st.session_state.page = page_name

# Main application
def main():
    st.set_page_config(
        page_title="Sistem Prediksi Kategori Kualitas Udara",
        layout="wide",
        page_icon="☁️",
        initial_sidebar_state="expanded"
    )
    
    load_css()
    init_session_state()
    
    # Sidebar navigation with custom styling
    st.sidebar.markdown("""
    <div class="sidebar-header">
        <h1>Sistem Prediksi Kategori Kualitas Udara</h1>
        <p>Prediksi Kualitas Udara Menggunakan Algoritma C4.5</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation buttons with icons
    menu_options = {
        "Beranda": "🏠",
        "Upload Data": "📤",
        "Penerapan Algoritma C4.5": "🌳",
        "Prediksi Kualitas Udara": "🔮"
    }
    
    # Create a button for each page
    for page, icon in menu_options.items():
        if st.sidebar.button(
            f"{icon} {page}", 
            use_container_width=True,
            key=f"nav_{page}",
            help=f"Halaman {page}",
        ):
            change_page(page)
    
    st.sidebar.markdown("---")
    st.sidebar.info(""" 
    Sistem Prediksi Kategori Kualitas Udara
    © 2025
    """)
    
    # Page routing
    if st.session_state.page == "Beranda":
        halaman.home.show()
    elif st.session_state.page == "Upload Data":
        halaman.upload.show()
    elif st.session_state.page == "Penerapan Algoritma C4.5":
        halaman.c45_model.show()
    elif st.session_state.page == "Prediksi Kualitas Udara":
        halaman.predict.show()

if __name__ == "__main__":
    main()