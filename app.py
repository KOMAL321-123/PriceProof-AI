import streamlit as st
import pandas as pd
import pytesseract
from PIL import Image
from groq import Groq


# -----------------------------
# Page Configuration
# -----------------------------
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
# Price Comparison Function
# -----------------------------
def compare_price(charged_price, reference_price):
    difference = charged_price - reference_price

    percentage_difference = (
        difference / reference_price
    ) * 100

    if percentage_difference >= 20:
        status = "Potentially Overpriced"
    elif percentage_difference > 0:
        status = "Above Reference Price"
    else:
        status = "Within Reference Range"

    return {
        "difference": round(difference, 2),
        "percentage_difference": round(percentage_difference, 2),
        "status": status
    }


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
# Price Comparison Demo
# -----------------------------
st.divider()

st.header("💰 Price Comparison Engine")

st.write(
    "Test how PriceProof compares a charged price "
    "with a reference market price."
)

selected_product = st.selectbox(
    "Select a product",
    price_data["product_name"].unique()
)

product_row = price_data[
    price_data["product_name"] == selected_product
].iloc[0]

reference_price = float(product_row["reference_price"])

charged_price = st.number_input(
    "Enter charged price (PKR)",
    min_value=0.0,
    value=reference_price,
    step=10.0
)

if st.button("⚖️ Compare Price"):

    result = compare_price(
        charged_price,
        reference_price
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Reference Price",
            f"PKR {reference_price:,.0f}"
        )

    with col2:
        st.metric(
            "Charged Price",
            f"PKR {charged_price:,.0f}"
        )

    with col3:
        st.metric(
            "Difference",
            f"{result['percentage_difference']:+.1f}%"
        )

    if result["status"] == "Potentially Overpriced":
        st.error(
            f"⚠️ {result['status']}"
        )

    elif result["status"] == "Above Reference Price":
        st.warning(
            f"⚠️ {result['status']}"
        )

    else:
        st.success(
            f"✅ {result['status']}"
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
        "These are prototype reference prices used for comparison. "
        "They are not official legal prices."
    )

    st.dataframe(
        price_data,
        width="stretch",
        hide_index=True
    )
