import streamlit as st
import pandas as pd
import pytesseract
from PIL import Image
from groq import Groq


st.set_page_config(
    page_title="PriceProof AI",
    page_icon="🧾",
    layout="wide"
)


# -----------------------------
# Groq Configuration
# -----------------------------
groq_api_key = st.secrets.get("GROQ_API_KEY")

if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
else:
    groq_client = None


# -----------------------------
# Load Reference Price Dataset
# -----------------------------
@st.cache_data
def load_price_data():
    return pd.read_csv("products.csv")


price_data = load_price_data()


# -----------------------------
# App Header
# -----------------------------
st.title("🧾 PriceProof AI")
st.subheader("Smart Receipt-Based Price Verification")

st.write(
    "Upload your shopping receipt to analyze product prices, "
    "compare them with reference market prices, and identify "
    "potentially suspicious price differences."
)

st.divider()


# -----------------------------
# Receipt Upload
# -----------------------------
st.header("📤 Upload Your Receipt")

uploaded_file = st.file_uploader(
    "Choose a receipt image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of your shopping receipt."
)


# -----------------------------
# Receipt OCR
# -----------------------------
if uploaded_file is not None:

    st.success("Receipt uploaded successfully!")

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Receipt",
        width="stretch"
    )

    st.divider()

    if st.button("🔍 Extract Receipt Text", type="primary"):

        with st.spinner("Reading your receipt..."):

            extracted_text = pytesseract.image_to_string(image)

        st.subheader("📄 Extracted Receipt Text")

        if extracted_text.strip():

            st.text_area(
                "Receipt content",
                extracted_text,
                height=250
            )

            st.success("Receipt text extracted successfully!")

        else:

            st.warning(
                "No readable text was detected. "
                "Please upload a clearer receipt image."
            )


# -----------------------------
# AI Status
# -----------------------------
st.divider()
st.subheader("🤖 Generative AI")

if groq_client:
    st.success("Groq AI is connected and ready.")
else:
    st.warning(
        "Groq API key is not configured yet. "
        "We will configure it securely before deployment."
    )


# -----------------------------
# Reference Database
# -----------------------------
with st.expander("📊 Reference Price Database"):

    st.write(
        "PriceProof uses reference market-price data for "
        "comparison. These prices are for prototype analysis "
        "and should not be treated as official legal prices."
    )

    st.dataframe(
        price_data,
        width="stretch",
        hide_index=True
    )
