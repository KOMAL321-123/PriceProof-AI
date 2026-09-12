import streamlit as st
import pandas as pd
from PIL import Image
from groq import Groq
import base64
import json


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
# Groq Vision Receipt Reader
# -----------------------------
def extract_receipt_data(uploaded_file):

    if not groq_client:
        return None, "Groq API key is not configured."

    try:

        image_bytes = uploaded_file.getvalue()

        base64_image = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        file_type = uploaded_file.type

        response = groq_client.chat.completions.create(
            model="qwen/qwen3.8-27b",

            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """
Analyze this shopping receipt carefully.

Extract the receipt information into the required JSON structure.

Important rules:

1. Read product names exactly as clearly as possible.
2. Extract the quantity for each item.
3. Extract the charged unit price when visible.
4. Extract the line total when visible.
5. Extract the receipt total when visible.
6. Extract store name and date when visible.
7. Do not guess information.
8. If information is not visible or cannot be read, use null.
9. Keep numbers as numbers, not strings.
10. Only include actual purchased items.
"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    f"data:{file_type};base64,"
                                    f"{base64_image}"
                                )
                            }
                        }
                    ]
                }
            ],

            temperature=0.2,

            max_completion_tokens=1500,

            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "receipt_data",
                    "strict": True,
                    "schema": {
                        "type": "object",

                        "properties": {

                            "store_name": {
                                "type": [
                                    "string",
                                    "null"
                                ]
                            },

                            "date": {
                                "type": [
                                    "string",
                                    "null"
                                ]
                            },

                            "items": {
                                "type": "array",

                                "items": {
                                    "type": "object",

                                    "properties": {

                                        "product_name": {
                                            "type": "string"
                                        },

                                        "quantity": {
                                            "type": [
                                                "number",
                                                "null"
                                            ]
                                        },

                                        "charged_price": {
                                            "type": [
                                                "number",
                                                "null"
                                            ]
                                        },

                                        "line_total": {
                                            "type": [
                                                "number",
                                                "null"
                                            ]
                                        }
                                    },

                                    "required": [
                                        "product_name",
                                        "quantity",
                                        "charged_price",
                                        "line_total"
                                    ],

                                    "additionalProperties": False
                                }
                            },

                            "total_amount": {
                                "type": [
                                    "number",
                                    "null"
                                ]
                            }
                        },

                        "required": [
                            "store_name",
                            "date",
                            "items",
                            "total_amount"
                        ],

                        "additionalProperties": False
                    }
                }
            }
        )

        content = response.choices[0].message.content

        receipt_data = json.loads(content)

        return receipt_data, None

    except Exception as error:

        return None, str(error)


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

        "percentage_difference": round(
            percentage_difference,
            2
        ),

        "status": status
    }


# -----------------------------
# AI Price Analysis Function
# -----------------------------
def analyze_with_ai(
    product_name,
    charged_price,
    reference_price,
    difference,
    percentage_difference,
    status
):

    if not groq_client:
        return "Groq API key is not configured."

    prompt = f"""
You are PriceProof AI, a careful consumer
price analysis assistant.

Analyze this price comparison:

Product: {product_name}
Charged price: PKR {charged_price:.2f}
Reference price: PKR {reference_price:.2f}
Difference: PKR {difference:.2f}
Percentage difference: {percentage_difference:.2f}%
System status: {status}

Give a short and clear explanation for the consumer.

Rules:

- Do not claim that the price is legally illegal.
- Do not invent market information.
- The reference price is only a comparison benchmark.
- Prices can vary by shop, location, brand,
  quantity, date, and promotions.
- If the charged price is significantly higher,
  explain that the consumer may want to verify it.
- Keep the answer concise and useful.
"""

    try:

        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful consumer "
                        "price verification assistant."
                    )
                },

                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2,

            max_completion_tokens=500,

            include_reasoning=False
        )

        return response.choices[0].message.content

    except Exception as error:

        return (
            f"AI analysis could not be completed: {error}"
        )


# -----------------------------
# App Header
# -----------------------------
st.title("🧾 PriceProof AI")

st.subheader(
    "Smart Receipt-Based Price Verification"
)

st.write(
    "Upload your shopping receipt to analyze product "
    "prices, compare them with reference market prices, "
    "and identify potentially suspicious price differences."
)

st.divider()


# -----------------------------
# Receipt Upload
# -----------------------------
st.header("📤 Upload Your Receipt")

uploaded_file = st.file_uploader(
    "Choose a receipt image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    help="Upload a clear image of your shopping receipt."
)


if uploaded_file is not None:

    st.success(
        "Receipt uploaded successfully!"
    )

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Receipt",
        width="stretch"
    )

    st.divider()

    if st.button(
        "🔍 Analyze Receipt with AI",
        type="primary"
    ):

        with st.spinner(
            "AI is reading and understanding your receipt..."
        ):

            receipt_data, error_message = (
                extract_receipt_data(
                    uploaded_file
                )
            )

        st.subheader(
            "📄 AI Receipt Analysis"
        )

        if receipt_data:

            # -----------------------------
            # Receipt Summary
            # -----------------------------
            st.success(
                "Receipt information extracted successfully!"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.write("**Store**")

                st.write(
                    receipt_data["store_name"]
                    if receipt_data["store_name"]
                    else "Not detected"
                )

            with col2:

                st.write("**Date**")

                st.write(
                    receipt_data["date"]
                    if receipt_data["date"]
                    else "Not detected"
                )

            st.divider()

            # -----------------------------
            # Extracted Items
            # -----------------------------
            st.write("### 🛒 Purchased Items")

            items = receipt_data["items"]

            if items:

                items_df = pd.DataFrame(items)

                items_df = items_df.rename(
                    columns={
                        "product_name": "Product",
                        "quantity": "Quantity",
                        "charged_price": "Charged Price (PKR)",
                        "line_total": "Line Total (PKR)"
                    }
                )

                st.dataframe(
                    items_df,
                    width="stretch",
                    hide_index=True
                )

            else:

                st.warning(
                    "No purchased items were detected."
                )

            # -----------------------------
            # Total Amount
            # -----------------------------
            st.divider()

            total_amount = (
                receipt_data["total_amount"]
            )

            if total_amount is not None:

                st.metric(
                    "Receipt Total",
                    f"PKR {total_amount:,.2f}"
                )

            else:

                st.info(
                    "Receipt total could not be detected."
                )

        else:

            st.error(
                "Receipt analysis failed."
            )

            st.caption(
                f"Error: {error_message}"
            )


# -----------------------------
# Price Comparison
# -----------------------------
st.divider()

st.header(
    "💰 Price Comparison Engine"
)

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

reference_price = float(
    product_row["reference_price"]
)

charged_price = st.number_input(
    "Enter charged price (PKR)",
    min_value=0.0,
    value=reference_price,
    step=10.0
)


if st.button(
    "⚖️ Compare Price"
):

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
# Generative AI Analysis
# -----------------------------
st.divider()

st.subheader(
    "🤖 AI Price Analysis"
)

if groq_client:

    with st.spinner(
        "AI is analyzing the price difference..."
    ):

        ai_analysis = analyze_with_ai(
            selected_product,
            charged_price,
            reference_price,
            result["difference"],
            result["percentage_difference"],
            result["status"]
        )

    st.info(
        ai_analysis
    )

else:

    st.warning(
        "Groq API key is not configured yet."
    )
