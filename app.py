import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="PriceProof AI",
    page_icon="🧾",
    layout="wide"
)

st.title("🧾 PriceProof AI")
st.subheader("Smart Receipt-Based Price Verification")

st.write(
    "Analyze shopping receipts, compare prices with reference market data, "
    "and identify potentially suspicious price differences."
)

st.divider()

# Load reference price dataset
@st.cache_data
def load_price_data():
    return pd.read_csv("products.csv")

price_data = load_price_data()

st.success("Reference price database loaded successfully.")

st.write(f"📦 Products available: **{len(price_data)}**")

st.dataframe(
    price_data,
    width="stretch",
    hide_index=True
)
