import streamlit as st

st.set_page_config(
    page_title="PriceProof AI",
    page_icon="🧾",
    layout="wide"
)

st.title("🧾 PriceProof AI")
st.subheader("Smart Receipt-Based Price Verification")

st.write(
    "Upload your shopping receipt and use AI-powered tools "
    "to analyze prices and identify potentially suspicious overcharging."
)

st.info("🚧 PriceProof AI is currently under development.")

st.divider()

st.header("📤 Upload Your Receipt")

uploaded_file = st.file_uploader(
    "Choose a receipt image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.success("Receipt uploaded successfully!")
    st.image(uploaded_file, caption="Uploaded Receipt", width="stretch")
