# streamlit_app.py
import streamlit as st
from pathlib import Path
from PIL import Image
import json

st.set_page_config(page_title="AI Detection & OCR", layout="wide")

st.title("Human & Animal Detection + OCR Viewer")

# ---------------------------
# Section 1: Annotated Images
# ---------------------------
st.header("Annotated Images (Humans & Animals)")
results_dir = Path("./results")
image_files = list(results_dir.glob("*.png")) + list(results_dir.glob("*.jpg")) + list(results_dir.glob("*.jpeg"))

if image_files:
    cols = st.columns(3)
    for idx, img_path in enumerate(image_files):
        img = Image.open(img_path)
        with cols[idx % 3]:
            st.image(img, caption=img_path.name, use_column_width=True)
else:
    st.write("No annotated images found. Run main.py first!")

# ---------------------------
# Section 2: OCR Results
# ---------------------------
st.header("OCR Results (Industrial / Stenciled Text)")
ocr_file = results_dir / "ocr_results.json"

if ocr_file.exists():
    with open(ocr_file) as f:
        ocr_results = json.load(f)
    
    for img_name, text in ocr_results.items():
        st.subheader(img_name)
        if text.strip():
            st.text(text)
        else:
            st.text("No text detected")
else:
    st.write("No OCR results found. Run main.py first!")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("✅ Annotated images and OCR results are displayed from the `results/` folder.")
