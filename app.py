import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="PriceProof AI",
    page_icon="🧾",
    layout="wide"
)


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
# Display Uploaded Receipt
# -----------------------------
if uploaded_file is not None:

    st.success("Receipt uploaded successfully!")

    st.image(
        uploaded_file,
        caption="Uploaded Receipt",
        width="stretch"
    )

    st.divider()

    st.info(
        "✅ Receipt received. PriceProof AI is ready to analyze it."
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
