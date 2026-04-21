import streamlit as st
import base64
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Detektor Penyakit Daun Singkong",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_image_path = "bg singkong3.jpg"
bg_ext = "jpeg" 

try:
    bg_base64 = get_base64_of_bin_file(bg_image_path)
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/{bg_ext};base64,{bg_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        .main {{
            padding-top: 2rem;
            background-color: rgba(15, 15, 15, 0.85); 
            border-radius: 15px;
            margin-top: 20px;
            margin-bottom: 20px;
            padding: 30px;
            box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.8);
        }}
        
        .center-title {{
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
            color: #4ade80; 
            text-shadow: 2px 2px 4px rgba(0,0,0,0.9);
        }}
        
        .center-subtitle {{
            text-align: center;
            font-size: 1.1rem;
            color: #ffffff; 
            margin-bottom: 2rem;
            font-weight: bold;
        }}
        
        .divider {{
            margin: 2rem 0;
            border-top: 2px solid #27ae60;
        }}
        
        .main p, .main span, .main label, 
        .main h1, .main h2, .main h3, .main h4, .main h5, .main h6,
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] h1,
        div[data-testid="stMarkdownContainer"] h2,
        div[data-testid="stMarkdownContainer"] h3,
        div[data-testid="stMarkdownContainer"] h4,
        div[data-testid="stMarkdownContainer"] h5,
        div[data-testid="stMarkdownContainer"] h6 {{
            color: #ffffff !important; 
        }}
        
        div[data-testid="stFileUploadDropzone"] div[data-testid="stMarkdownContainer"] p, 
        div[data-testid="stFileUploadDropzone"] span {{
             color: #ffffff !important;
        }}
        
        div[data-testid="stUploadedFile"] div,
        div[data-testid="stUploadedFile"] span,
        div[data-testid="stUploadedFile"] p,
        .uploadedFileName, 
        .uploadedFileSize {{
            color: #ffffff !important;
        }}
        
        div[data-testid="stFileUploadDropzone"] {{
            border-color: #4ade80 !important;
            background-color: rgba(255, 255, 255, 0.05) !important;
        }}
        
        div[data-testid="stAlert"] p, 
        div[data-testid="stAlert"] span, 
        div[data-testid="stAlert"] div {{
            color: #ffffff !important;
        }}
        
        div[data-testid="stButton"] button {{
            background-color: #2ecc71 !important; 
            color: #ffffff !important;
            border: 2px solid #27ae60 !important;
            font-weight: bold !important;
            padding: 0.5rem 2.5rem !important; 
            width: 100% !important; 
            border-radius: 8px !important;
            transition: all 0.3s ease;
        }}
        
        div[data-testid="stButton"] button:hover {{
            background-color: #27ae60 !important; 
            border-color: #1f8449 !important;
            transform: scale(1.05); 
        }}
        
        </style>
        """,
        unsafe_allow_html=True
    )
except FileNotFoundError:
    st.error(f"❌ Gambar background '{bg_image_path}' tidak ditemukan di folder ini!")

st.markdown("<div class='center-title'>🌿 Simulator Deteksi Penyakit Daun Singkong</div>", unsafe_allow_html=True)
st.markdown("<div class='center-subtitle'>Unggah foto daun untuk mendeteksi penyakit <b>CBB, CBSD, CGM, CMD,</b> atau <b>Sehat</b>.</div>", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

col_center = st.columns([1, 2, 1])
with col_center[1]:
    uploaded_file = st.file_uploader("📁 Pilih gambar daun dari perangkat Anda...", type=["jpg", "jpeg", "png"])
    button_clicked = False 

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_btn_left, col_btn_center, col_btn_right = st.columns([1, 1, 1])
    with col_btn_center:
        button_clicked = st.button("🔍 Pindai Daun Sekarang!", type="primary", use_container_width=True)
        
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    if button_clicked:
        with st.spinner('⏳ Gambar daun sedang diperiksa...'):
            results = model.predict(image, conf=0.40)
            annotated_img = results[0].plot()
            annotated_img_rgb = annotated_img[..., ::-1]
            
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("#### 📷 Gambar Asli")
            st.image(image, use_container_width=True)
        with col2:
            st.markdown("#### 🎯 Hasil Deteksi")
            st.image(annotated_img_rgb, use_container_width=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
            
        detected_classes = [model.names[int(c)].lower() for c in results[0].boxes.cls]
        unique_classes = list(set(detected_classes))
        
        if len(detected_classes) > 0:
            penyakit = [c for c in unique_classes if c != 'healthy']
            
            if len(penyakit) > 0:
                # HITUNG HANYA BOUNDING BOX PENYAKIT (Menyingkirkan yang healthy)
                detected_penyakit = [c for c in detected_classes if c != 'healthy']
                
                st.markdown("### ⚠️ Kesimpulan: Penyakit Terdeteksi")
                col_disease1, col_disease2 = st.columns(2)
                with col_disease1:
                    st.error(f"**Jenis Penyakit:** {', '.join(penyakit).upper()}")
                with col_disease2:
                    # Menampilkan panjang dari list yang khusus berisi penyakit
                    st.warning(f"**Total Area Terjangkit:** {len(detected_penyakit)} titik")
            else:
                st.markdown("### ✨ Kesimpulan: Daun Sehat")
                st.success("Bagus! Tidak terdeteksi adanya gejala penyakit pada daun ini.", icon="✅")
        else:
            st.info("ℹ️ Tidak ada objek daun atau penyakit yang terdeteksi pada gambar ini. Harap periksa kembali gambar yang diunggah.")
            
    else:
        col_preview = st.columns([1, 2, 1])
        with col_preview[1]:
            st.markdown("#### 📷 Preview Gambar")
            st.image(image, use_container_width=True)
            
else:
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    col_empty = st.columns([1, 2, 1])
    with col_empty[1]:
        st.info("👆 Silakan unggah gambar daun terlebih dahulu untuk memulai deteksi penyakit.")