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
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

        * {{
            font-family: 'Plus Jakarta Sans', sans-serif;
        }}

        .stApp {{
            position: relative;
            background-image: url("data:image/{bg_ext};base64,{bg_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            backdrop-filter: blur(7px);
            -webkit-backdrop-filter: blur(7px);
            background: rgba(8, 12, 8, 0.22);
            pointer-events: none;
            z-index: 0;
        }}

        .stApp > * {{
            position: relative;
            z-index: 1;
        }}

        .main {{
            padding-top: 2rem;
            background-color: rgba(10, 14, 10, 0.88);
            border-radius: 20px;
            margin-top: 20px;
            margin-bottom: 20px;
            padding: 36px;
            box-shadow: 0px 12px 40px rgba(0, 0, 0, 0.9);
        }}

        .center-title {{
            text-align: center;
            font-size: 2.6rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
            color: #4ade80;
            text-shadow: 0 0 24px rgba(74, 222, 128, 0.35);
            letter-spacing: -0.5px;
        }}

        .center-subtitle {{
            text-align: center;
            font-size: 1.05rem;
            color: #f1fff5;
            margin-bottom: 0;
            font-weight: 500;
        }}

        .header-box {{
            background: #15221a;
            border: 1px solid #4ade80;
            border-radius: 14px;
            padding: 0.8rem 1rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
            margin-bottom: 1.1rem;
        }}

        .divider {{
            margin: 2rem 0;
            border: none;
            border-top: 1px solid rgba(74, 222, 128, 0.25);
        }}

        /* hide default radio widget */
        div[data-testid="stRadio"] {{
            display: none !important;
        }}

        /* ── Text colors ── */
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
            color: #4ade80 !important;
        }}

        div[data-testid="stUploadedFile"] div,
        div[data-testid="stUploadedFile"] span,
        div[data-testid="stUploadedFile"] p,
        .uploadedFileName,
        .uploadedFileSize {{
            color: #000000 !important;
        }}

        div[data-testid="stFileUploadDropzone"] {{
            border: 2px dashed rgba(74, 222, 128, 0.4) !important;
            background-color: rgba(74, 222, 128, 0.04) !important;
            border-radius: 14px !important;
            transition: all 0.2s;
        }}

        div[data-testid="stFileUploadDropzone"]:hover {{
            border-color: #4ade80 !important;
            background-color: rgba(74, 222, 128, 0.08) !important;
        }}

        /* ── Internal upload button inside file uploader ── */
        div[data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"],
        div[data-testid="stFileUploadDropzone"] button[data-testid="stBaseButton-secondary"],
        div[data-testid="stFileUploader"] button[kind="secondary"],
        div[data-testid="stFileUploadDropzone"] button[kind="secondary"] {{
            background-color: #1f3a2d !important;
            color: #ffffff !important;
            border: 1px solid #4ade80 !important;
            border-radius: 10px !important;
            box-shadow: none !important;
        }}

        div[data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"]:hover,
        div[data-testid="stFileUploadDropzone"] button[data-testid="stBaseButton-secondary"]:hover,
        div[data-testid="stFileUploader"] button[kind="secondary"]:hover,
        div[data-testid="stFileUploadDropzone"] button[kind="secondary"]:hover {{
            background-color: #244031 !important;
            color: #ffffff !important;
            border-color: #86efac !important;
        }}

        div[data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"] p,
        div[data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"] span,
        div[data-testid="stFileUploadDropzone"] button[data-testid="stBaseButton-secondary"] p,
        div[data-testid="stFileUploadDropzone"] button[data-testid="stBaseButton-secondary"] span,
        div[data-testid="stFileUploader"] button[kind="secondary"] p,
        div[data-testid="stFileUploader"] button[kind="secondary"] span,
        div[data-testid="stFileUploadDropzone"] button[kind="secondary"] p,
        div[data-testid="stFileUploadDropzone"] button[kind="secondary"] span {{
            color: #ffffff !important;
        }}

        div[data-testid="stAlert"] p,
        div[data-testid="stAlert"] span,
        div[data-testid="stAlert"] div {{
            color: #ffffff !important;
        }}

        div[data-testid="stAlert"] {{
            background: #15221a !important;
            border: 1px solid #4ade80 !important;
            border-radius: 12px !important;
        }}

        .hint-card {{
            background: #15221a;
            color: #f1fff5;
            border: 1px solid #4ade80;
            border-radius: 12px;
            padding: 0.85rem 1rem;
            font-weight: 600;
            text-align: center;
        }}

        /* ── Source selector buttons (default) ── */
        div[data-testid="stButton"] button {{
            background: #1a2920 !important;
            color: #ffffff !important;
            border: 2px solid #2f5942 !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            padding: 0.65rem 1rem !important;
            width: 100% !important;
            border-radius: 14px !important;
            transition: all 0.22s ease !important;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.35) !important;
        }}

        div[data-testid="stButton"] button:hover {{
            background: #244031 !important;
            border-color: #4ade80 !important;
            color: #e8fff0 !important;
            transform: translateY(-2px);
            box-shadow: 0 6px 24px rgba(0, 0, 0, 0.45) !important;
        }}

        /* ── Scan button override (primary) ── */
        div[data-testid="stButton"] button[kind="primary"] {{
            background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%) !important;
            color: #0a0e0a !important;
            border: none !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 18px rgba(46, 204, 113, 0.35) !important;
        }}

        div[data-testid="stButton"] button[kind="primary"]:hover {{
            background: linear-gradient(135deg, #27ae60 0%, #1f8449 100%) !important;
            box-shadow: 0 8px 28px rgba(46, 204, 113, 0.5) !important;
            transform: translateY(-1px);
        }}

        /* ── Active source button via wrapper class ── */
        .btn-active div[data-testid="stButton"] button {{
            background: #2f5942 !important;
            border-color: #4ade80 !important;
            color: #ffffff !important;
            box-shadow: 0 0 0 2px #4ade80 !important;
        }}

        /* ── Section label ── */
        .section-label {{
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 1.4px;
            text-transform: uppercase;
            color: #4ade80;
            margin-bottom: 0.6rem;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )
except FileNotFoundError:
    st.error(f"❌ Gambar background '{bg_image_path}' tidak ditemukan di folder ini!")

# ── Header ──
st.markdown(
    """
    <div class='header-box'>
        <div class='center-title'>🌿 Simulator Deteksi Penyakit Daun Singkong</div>
        <div class='center-subtitle'>Unggah foto daun untuk mendeteksi penyakit <b>CBB, CBSD, CGM, CMD,</b> atau <b>Sehat</b>.</div>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ── Model ──
@st.cache_resource
def load_model():
    return YOLO('best3.pt')

model = load_model()

# ── Init state — key TERPISAH dari key widget ──
if "active_source" not in st.session_state:
    st.session_state.active_source = None

# ── Source selector ──
col_center = st.columns([1, 2, 1])
with col_center[1]:
    col_a, col_b = st.columns(2)

    with col_a:
        if st.session_state.active_source == "Upload":
            st.markdown("<div class='btn-active'>", unsafe_allow_html=True)
        clicked_upload = st.button("📁  Upload Gambar", key="btn_upload", use_container_width=True)
        if st.session_state.active_source == "Upload":
            st.markdown("</div>", unsafe_allow_html=True)
        if clicked_upload:
            st.session_state.active_source = "Upload"
            st.rerun()

    with col_b:
        if st.session_state.active_source == "Kamera":
            st.markdown("<div class='btn-active'>", unsafe_allow_html=True)
        clicked_kamera = st.button("📷  Ambil Sekarang", key="btn_kamera", use_container_width=True)
        if st.session_state.active_source == "Kamera":
            st.markdown("</div>", unsafe_allow_html=True)
        if clicked_kamera:
            st.session_state.active_source = "Kamera"
            st.rerun()

    source = st.session_state.active_source
    image = None

    if source is not None:
        st.markdown("<br>", unsafe_allow_html=True)

        if source == "Upload":
            st.markdown("<div class='section-label'>📁 Upload File</div>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Pilih gambar daun dari perangkat Anda...",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
        elif source == "Kamera":
            st.markdown("<div class='section-label'>📷 Kamera</div>", unsafe_allow_html=True)
            camera_file = st.camera_input("Ambil foto daun", label_visibility="collapsed")
            if camera_file is not None:
                image = Image.open(camera_file)

# ── Scan & results ──
if image is not None:
    st.markdown("<br>", unsafe_allow_html=True)

    col_btn_left, col_btn_center, col_btn_right = st.columns([1, 1, 1])
    with col_btn_center:
        button_clicked = st.button("🔍 Pindai Daun Sekarang!", type="primary", use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    if button_clicked:
        with st.spinner('⏳ Gambar daun sedang diperiksa...'):
            results = model.predict(image, conf=0.1)
            annotated_img = results[0].plot()
            annotated_img_rgb = annotated_img[..., ::-1]

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("#### 📷 Gambar Asli")
            st.image(image, width='stretch')
        with col2:
            st.markdown("#### 🎯 Hasil Deteksi")
            st.image(annotated_img_rgb, width='stretch')

        st.markdown("<br>", unsafe_allow_html=True)

        detected_classes = [model.names[int(c)].lower() for c in results[0].boxes.cls]
        unique_classes = list(set(detected_classes))

        if len(detected_classes) > 0:
            penyakit = [c for c in unique_classes if c != 'healthy']

            if len(penyakit) > 0:
                detected_penyakit = [c for c in detected_classes if c != 'healthy']
                st.markdown("### ⚠️ Kesimpulan: Penyakit Terdeteksi")
                col_disease1, col_disease2 = st.columns(2)
                with col_disease1:
                    st.error(f"**Jenis Penyakit:** {', '.join(penyakit).upper()}")
                with col_disease2:
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
            st.image(image, width='stretch')
else:
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    col_empty = st.columns([1, 2, 1])
    with col_empty[1]:
        st.info("👆 Silakan unggah gambar daun terlebih dahulu untuk memulai deteksi penyakit.") 