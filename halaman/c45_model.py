# pages/c45_model.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree
import base64
from io import BytesIO
import matplotlib as mpl
import seaborn as sns
import joblib
import os
import time

# Nama file tempat model, scaler, dan metadata akan disimpan
MODEL_SAVE_FILE = "file/model_and_scaler_data.pkl"

def get_tree_image(model, feature_names, class_names):
    """
    Fungsi untuk membuat visualisasi pohon keputusan dengan warna kustom
    dan mengembalikannya dalam format base64.
    """
    color_map = {
        'Baik': '#8bc34a',
        'Sedang': '#ffb300',
        'Tidak Sehat': '#e53935',
        'Sangat Tidak Sehat': '#7b1fa2',
        'Berbahaya': '#212121'
    }

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 12))

    tree.plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=10,
        proportion=True,
        node_ids=True,
        impurity=False,
        label='root',
        ax=ax
    )
    
    all_objects = ax.get_children()
    node_values = model.tree_.value.squeeze().astype(int)

    node_counter = 0
    for obj in all_objects:
        if isinstance(obj, mpl.patches.FancyBboxPatch):
            if node_counter < len(node_values):
                class_index = node_values[node_counter].argmax()
                class_name = class_names[class_index]
                color = color_map.get(class_name, '#cccccc')
                
                obj.set_facecolor(color)
                node_counter += 1
            
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return img_base64

def explain_tree_visual(class_names):
    """Fungsi untuk menampilkan penjelasan visual pohon keputusan."""
    st.markdown("#### 📖 Cara Membaca Pohon Keputusan (Seperti Flowchart!)")
    
    st.markdown("""
    Pohon keputusan ini adalah "otak" dari model prediksi kita. Anda bisa membacanya seperti sebuah diagram alur sederhana:
    1.  **Mulai dari kotak paling atas.** Ini adalah pertanyaan pertama.
    2.  **Jawab pertanyaannya.** Contoh: `CO (ppm) ≤ 0.25`.
    3.  **Ikuti panah.** Jika jawabannya **"Ya" (True)**, ikuti panah ke kiri. Jika **"Tidak" (False)**, ikuti panah ke kanan.
    4.  **Lanjutkan sampai kotak terakhir.** Kotak ini tidak ada panah lagi dan di sanalah hasil prediksi Anda!
    """)
    
    st.markdown("#### 🎨 Arti Warna Kotak:")
    
    cols = st.columns(2)
    col1_items = [('Baik', 'Hijau', '4caf50'), ('Sedang', 'Oranye', 'ff9800'), ('Tidak Sehat', 'Merah', 'f44336')]
    col2_items = [('Sangat Tidak Sehat', 'Ungu', 'ba68c8'), ('Berbahaya', 'Hitam', '212121')]
    
    with cols[0]:
        for name, color_name, color_hex in col1_items:
            st.markdown(f"""
            <div style="background-color:#{color_hex}1a; padding:10px; border-radius:5px; margin-bottom:10px; border-left: 4px solid #{color_hex};">
                <p style="margin:0; color:#333;">🟢 {name}: <b>{color_name}</b></p>
            </div>
            """, unsafe_allow_html=True)
    with cols[1]:
        for name, color_name, color_hex in col2_items:
            st.markdown(f"""
            <div style="background-color:#{color_hex}1a; padding:10px; border-radius:5px; margin-bottom:10px; border-left: 4px solid #{color_hex};">
                <p style="margin:0; color:#333;">🟢 {name}: <b>{color_name}</b></p>
            </div>
            """, unsafe_allow_html=True)

def get_rules_list(model, feature_names, class_names):
    """Mengambil semua aturan dari pohon dan mengembalikannya dalam format yang mudah dibaca."""
    tree_ = model.tree_
    
    feature_name = [
        feature_names[i] if i != -2 else "Tidak ada"
        for i in tree_.feature
    ]
    
    rules = []
    
    def get_rule_paths(node_idx, path):
        if tree_.feature[node_idx] == -2:
            class_index = tree_.value[node_idx].argmax()
            class_name = class_names[class_index]
            rules.append((path, class_name))
        else:
            left_child = tree_.children_left[node_idx]
            threshold = tree_.threshold[node_idx]
            new_path_left = path + [f"{feature_name[node_idx]} ≤ {threshold:.2f}"]
            get_rule_paths(left_child, new_path_left)

            right_child = tree_.children_right[node_idx]
            threshold = tree_.threshold[node_idx]
            new_path_right = path + [f"{feature_name[node_idx]} > {threshold:.2f}"]
            get_rule_paths(right_child, new_path_right)
    
    get_rule_paths(0, [])
    return rules

def display_attractive_rules(rules):
    """Menampilkan aturan model dalam format alur yang menarik dengan card."""
    color_map = {
        'Baik': {'bg': '#e8f5e9', 'border': '#4caf50', 'text': '#1b5e20'},
        'Sedang': {'bg': '#fff3e0', 'border': '#ff9800', 'text': '#e65100'},
        'Tidak Sehat': {'bg': '#ffebee', 'border': '#f44336', 'text': '#b71c1c'},
        'Sangat Tidak Sehat': {'bg': '#f3e5f5', 'border': '#ba68c8', 'text': '#4a148c'},
        'Berbahaya': {'bg': '#e0e0e0', 'border': '#212121', 'text': '#212121'}
    }
    
    for i, (path, result) in enumerate(rules[:10]):
        colors = color_map.get(result, {'bg': '#f0f2f6', 'border': '#ccc', 'text': '#333'})
        
        with st.expander(
            f"Alur Keputusan #{i+1} - Prediksi: **{result}**"
        ):
            st.markdown(
                f'<div style="background-color: {colors["bg"]}; border-left: 5px solid {colors["border"]}; padding: 10px; border-radius: 5px; color: {colors["text"]};">'
                "Berikut langkah-langkah yang dilalui model untuk mencapai prediksi ini:",
                unsafe_allow_html=True
            )
            for step_num, step_rule in enumerate(path):
                if ' ≤ ' in step_rule:
                    feature_name, threshold_value = step_rule.split(' ≤ ', 1)
                    operator = 'kurang dari atau sama dengan'
                else:
                    feature_name, threshold_value = step_rule.split(' > ', 1)
                    operator = 'lebih dari'
                
                st.markdown(f"➡️ **Langkah {step_num+1}:** Jika nilai **{feature_name}** {operator} {threshold_value}")
            
            st.markdown(f"""
            <div style="margin-top: 15px; background-color: {colors['border']}1a; border-left: 4px solid {colors['border']}; padding: 10px; border-radius: 5px;">
                <p style="margin:0; font-weight:bold; color: {colors['text']};">
                    ✅ Hasil Akhir: <span style="font-weight:bolder; color: {colors['text']};">{result}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
    if len(rules) > 10:
        st.info(f"Masih ada {len(rules) - 10} alur keputusan lainnya. Model ini sangat detail!")

def show():
    st.title("🌳 Penerapan Algoritma C4.5")
    st.markdown("""
    <div class="card">
        <h4>Pemodelan Decision Tree untuk Prediksi Kualitas Udara</h4>
        <p>Algoritma C4.5 akan membangun sebuah "pohon keputusan" berdasarkan data yang Anda unggah. Pohon ini yang nantinya akan digunakan untuk memprediksi kualitas udara.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- PENTING: Periksa ketersediaan data sebelum memulai ---
    if 'normalized_data' not in st.session_state or st.session_state.normalized_data is None:
        st.warning("⚠️ Data yang dinormalisasi tidak ditemukan. Silakan unggah dan normalisasi data terlebih dahulu di halaman **'Upload Data'**.")
        return
    if 'scaler' not in st.session_state or st.session_state.scaler is None:
        st.warning("⚠️ Objek scaler tidak ditemukan. Pastikan Anda telah menormalisasi data di halaman **'Upload Data'**.")
        return
    
    # Ambil data dari session state
    df_normalized = st.session_state.normalized_data
    
    # Memisahkan fitur (X) dan target (y)
    try:
        X = df_normalized.drop('Kategori Kualitas Udara', axis=1)
        y = df_normalized['Kategori Kualitas Udara']
    except KeyError:
        st.error("❌ Kolom 'Kategori Kualitas Udara' tidak ditemukan dalam data.")
        return

    st.success("✅ Data berhasil dimuat dari halaman Upload Data.")

    with st.expander("🔍 Pratinjau Data yang Digunakan", expanded=False):
        st.dataframe(df_normalized.head(10))
        st.info(f"Menampilkan 10 dari {len(df_normalized)} baris data")
    
    st.markdown("---")
    st.subheader("⚙️ Konfigurasi Model & Pelatihan")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        test_size = st.slider("Ukuran Data Uji (%)", 10, 50, 20, 5) / 100
        max_depth = st.slider("Kedalaman Maksimum Pohon", 1, 20, 7, 1)

    if st.button("🚀 Latih dan Evaluasi Model C4.5", use_container_width=True):
        with st.spinner('⏳ Sedang melatih dan mengevaluasi model...'):
            time.sleep(1) 
            
            # Mengubah label target menjadi numerik
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            # Membagi data menjadi data latih dan data uji
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
            )

            # Inisialisasi dan latih model Decision Tree
            model = DecisionTreeClassifier(
                criterion='entropy', 
                max_depth=max_depth,
                random_state=42
            )
            model.fit(X_train, y_train)

            # Melakukan prediksi pada data uji
            y_pred = model.predict(X_test)
            
            # Simpan SEMUA objek penting ke file tunggal untuk persistensi
            if not os.path.exists("file"):
                os.makedirs("file")
            
            model_and_metadata = {
                'model': model,
                'scaler': st.session_state.scaler, # Ambil scaler dari session state
                'feature_names': X.columns.tolist(),
                'class_names': label_encoder.classes_.tolist()
            }
            joblib.dump(model_and_metadata, MODEL_SAVE_FILE)
            
            # Simpan model dan data penting ke session state untuk sesi saat ini
            st.session_state.model = model
            st.session_state.feature_names = X.columns.tolist()
            st.session_state.label_encoder = label_encoder
            st.session_state.class_names = label_encoder.classes_.tolist()
            st.session_state.model_trained = True
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            
            st.success(f"🎉 Model dan Scaler berhasil dilatih, dievaluasi, dan disimpan dalam satu file! Model siap digunakan untuk prediksi.")
            st.balloons()
            st.rerun()

    if st.session_state.get('model_trained', False) and st.session_state.get('y_test') is not None:
        model = st.session_state.model
        feature_names = st.session_state.feature_names
        class_names = st.session_state.class_names
        y_test = st.session_state.y_test
        y_pred = st.session_state.y_pred

        # Hitung metrik evaluasi
        accuracy = accuracy_score(y_test, y_pred)
        
        report = classification_report(y_test, y_pred, target_names=class_names, labels=np.arange(len(class_names)), output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        st.markdown("---")
        st.subheader("🌿 Visualisasi Pohon Keputusan")
        st.info("Pohon ini adalah hasil 'belajar' dari data yang Anda unggah. Gunakan penjelasan di samping untuk membacanya.")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            try:
                tree_img = get_tree_image(model, feature_names, class_names)
                st.image(f"data:image/png;base64,{tree_img}", use_container_width=True)
                st.download_button(
                    label="💾 Unduh Pohon Keputusan (PNG)",
                    data=base64.b64decode(tree_img),
                    file_name="pohon_keputusan_c45.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat membuat visualisasi pohon: {e}")
        with col2:
            explain_tree_visual(class_names)
            
        st.subheader("📝 Aturan Utama Model dalam Format Alur")
        st.info("""
        Model C4.5 mengambil keputusan berdasarkan serangkaian aturan. Klik pada setiap alur keputusan di bawah untuk melihat detail langkahnya.
        """)
        
        rules = get_rules_list(model, feature_names, class_names)
        display_attractive_rules(rules)
        
        st.subheader("📊 Tingkat Kepentingan Fitur")
        st.info("""
        Grafik ini menunjukkan seberapa besar pengaruh setiap parameter lingkungan dalam membuat keputusan.
        Semakin tinggi nilai bar, semakin penting parameter tersebut bagi model.
        """)
        
        with st.expander("Klik untuk melihat Detail Feature Importance", expanded=True):
            feature_importance = pd.DataFrame({
                'Fitur': feature_names,
                'Kepentingan': model.feature_importances_
            }).sort_values('Kepentingan', ascending=False)
            
            st.dataframe(feature_importance)
            st.bar_chart(feature_importance.set_index('Fitur'))
        
        st.markdown("---")
        st.subheader("📝 Hasil Evaluasi Model")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Akurasi Model pada Data Uji", value=f"{accuracy*100:.2f}%")
        with col2:
            st.metric(label="Jumlah Data Uji", value=len(y_test))

        st.markdown("#### Laporan Klasifikasi")
        st.info("Tabel di bawah ini menampilkan metrik evaluasi utama seperti precision, recall, dan f1-score untuk setiap kategori.")
        
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        st.markdown("#### Matriks Kebingungan (Confusion Matrix)")
        st.info("Matriks ini menunjukkan jumlah prediksi benar dan salah. Angka di diagonal adalah prediksi yang benar.")

        conf_matrix_col1, conf_matrix_col2, conf_matrix_col3 = st.columns([1, 2, 1])
        with conf_matrix_col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_xlabel('Prediksi')
            ax.set_ylabel('Aktual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig, use_container_width=True)
    else:
        st.info("ℹ️ Silakan klik tombol '🚀 Latih dan Evaluasi Model C4.5' di atas untuk memulai pelatihan menggunakan data yang telah diunggah.")
