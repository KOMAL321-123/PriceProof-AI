import streamlit as st
import pandas as pd
from PIL import Image
from groq import Groq
import base64
import json
import re
from difflib import SequenceMatcher


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PriceProof AI",
    page_icon="🧾",
    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title("🧾 PriceProof AI")
st.subheader("Detect overpricing. Verify the price. Know your rights.")

st.write(
    "Upload a shopping receipt and let AI extract the receipt information "
    "and automatically verify purchased items against reference prices."
)

st.info(
    "⚠️ Reference prices are prototype market benchmarks, not official government prices. "
    "Actual prices may vary by city, shop, brand, package size, date, and promotions."
)


# =========================================================
# LOAD REFERENCE DATABASE
# =========================================================

try:
    price_data = pd.read_csv("products.csv")
except Exception as e:
    st.error(f"Could not load products.csv: {e}")
    st.stop()


# =========================================================
# GROQ CLIENT
# =========================================================

groq_api_key = st.secrets.get("GROQ_API_KEY")

if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
else:
    groq_client = None


# =========================================================
# AI STATUS
# =========================================================

if groq_client:
    st.success("🟢 AI system is connected")
else:
    st.warning(
        "🟡 AI system is not connected. Please add GROQ_API_KEY in Streamlit Secrets."
    )


# =========================================================
# RECEIPT UPLOAD
# =========================================================

st.header("📤 Upload Your Receipt")

uploaded_file = st.file_uploader(
    "Choose a receipt image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear photo or screenshot of your shopping receipt."
)


# =========================================================
# REFERENCE DATABASE
# =========================================================

with st.expander("📊 View Prototype Reference Price Database"):
    st.dataframe(
        price_data,
        use_container_width=True
    )


# =========================================================
# AI RECEIPT EXTRACTION
# =========================================================

def extract_receipt_data(uploaded_file):

    if not groq_client:
        return None, "Groq API key is not configured."

    try:

        image_bytes = uploaded_file.getvalue()

        image_base64 = base64.b64encode(
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
You are a receipt analysis assistant.

Carefully read the shopping receipt image.

Extract only information that is clearly visible.

Rules:
1. Do not invent information.
2. If store name is not visible, use null.
3. If date is not visible, use null.
4. Extract every clearly visible purchased item.
5. Extract the product name as accurately as possible.
6. Extract brand if clearly visible.
7. Extract quantity when visible.
8. Extract charged price for the item.
9. Extract line total when visible.
10. Extract final receipt total when visible.
11. Do not include currency symbols inside numeric fields.
12. Use numbers for prices.
13. If a value cannot be determined, use null.
"""
                        },

                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    f"data:{file_type};base64,"
                                    f"{image_base64}"
                                )
                            }
                        }
                    ]
                }
            ],

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

                                        "brand": {
                                            "type": [
                                                "string",
                                                "null"
                                            ]
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
                                        "brand",
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
            },

            temperature=0.2,

            max_completion_tokens=900
        )

        result_text = response.choices[0].message.content

        result = json.loads(result_text)

        return result, None

    except Exception as e:

        return None, str(e)


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =========================================================
# PRODUCT MATCHING
# =========================================================

def calculate_similarity(text1, text2):

    text1 = normalize_text(text1)
    text2 = normalize_text(text2)

    if not text1 or not text2:
        return 0

    return SequenceMatcher(
        None,
        text1,
        text2
    ).ratio()


def find_best_match(
    receipt_product,
    receipt_brand,
    database
):

    best_match = None
    best_score = 0

    receipt_product_normalized = normalize_text(
        receipt_product
    )

    receipt_brand_normalized = normalize_text(
        receipt_brand
    )

    for _, row in database.iterrows():

        database_product = normalize_text(
            row["product_name"]
        )

        database_brand = normalize_text(
            row["brand"]
        )

        product_score = calculate_similarity(
            receipt_product_normalized,
            database_product
        )

        brand_score = 0

        if (
            receipt_brand_normalized
            and database_brand
        ):

            brand_score = calculate_similarity(
                receipt_brand_normalized,
                database_brand
            )

        # Give extra importance to brand when available
        if brand_score > 0:
            final_score = (
                product_score * 0.70
                + brand_score * 0.30
            )
        else:
            final_score = product_score

        if final_score > best_score:

            best_score = final_score
            best_match = row

    # Minimum confidence for automatic matching
    if best_score >= 0.55:

        return best_match, best_score

    return None, best_score


# =========================================================
# PRICE COMPARISON
# =========================================================

def compare_price(
    charged_price,
    reference_price
):

    if reference_price <= 0:

        return {
            "difference": 0,
            "percentage_difference": 0,
            "status": "Unable to Compare"
        }

    difference = (
        charged_price -
        reference_price
    )

    percentage_difference = (
        difference /
        reference_price
    ) * 100

    if percentage_difference >= 20:

        status = "Potentially Overpriced"

    elif percentage_difference > 0:

        status = "Above Reference Price"

    else:

        status = "Within Reference Range"

    return {

        "difference": round(
            difference,
            2
        ),

        "percentage_difference": round(
            percentage_difference,
            2
        ),

        "status": status
    }


# =========================================================
# AI PRICE EXPLANATION
# =========================================================

def analyze_with_ai(
    product_name,
    charged_price,
    reference_price,
    difference,
    percentage_difference,
    status
):

    if not groq_client:

        return (
            "AI analysis is unavailable because "
            "the Groq API key is not configured."
        )

    try:

        prompt = f"""
You are PriceProof AI, a consumer price analysis assistant.

Analyze this price comparison:

Product: {product_name}
Charged price: {charged_price}
Reference price: {reference_price}
Difference: {difference}
Percentage difference: {percentage_difference}%
Status: {status}

Give a short and clear explanation for the consumer.

Requirements:
- Explain whether the charged price is above or within the reference benchmark.
- Do not claim that the shop committed a crime.
- Do not give legal advice.
- Mention that the reference price is only a benchmark.
- Prices can vary because of location, shop, brand, package size,
  date, discounts, and promotions.
- Keep the answer concise.
"""

        response = groq_client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2,

            max_completion_tokens=500
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"AI analysis could not be completed: {e}"


# =========================================================
# DISPLAY RECEIPT
# =========================================================

if uploaded_file:

    st.header("🧾 Receipt")

    try:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Receipt",
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Could not display the receipt image: {e}"
        )

        st.stop()


    # =====================================================
    # EXTRACT RECEIPT DATA
    # =====================================================

    st.header("🤖 AI Receipt Analysis")

    with st.spinner(
        "AI is reading your receipt..."
    ):

        receipt_data, error_message = (
            extract_receipt_data(
                uploaded_file
            )
        )


    # =====================================================
    # EXTRACTION ERROR
    # =====================================================

    if error_message:

        st.error(
            "Receipt analysis failed."
        )

        st.code(
            error_message
        )

        st.stop()


    # =====================================================
    # DISPLAY RECEIPT INFORMATION
    # =====================================================

    if receipt_data:

        st.success(
            "✅ Receipt information extracted successfully!"
        )

        col1, col2 = st.columns(2)

        with col1:

            store_name = receipt_data.get(
                "store_name"
            )

            st.metric(
                "🏪 Store",
                store_name
                if store_name
                else "Not detected"
            )

        with col2:

            receipt_date = receipt_data.get(
                "date"
            )

            st.metric(
                "📅 Date",
                receipt_date
                if receipt_date
                else "Not detected"
            )


        # =================================================
        # EXTRACTED ITEMS
        # =================================================

        st.subheader(
            "🛒 Extracted Purchased Items"
        )

        items = receipt_data.get(
            "items",
            []
        )

        if items:

            items_df = pd.DataFrame(
                items
            )

            st.dataframe(
                items_df,
                use_container_width=True
            )

        else:

            st.warning(
                "No purchased items could be detected."
            )


        # =================================================
        # RECEIPT TOTAL
        # =================================================

        total_amount = receipt_data.get(
            "total_amount"
        )

        if total_amount is not None:

            st.metric(
                "💰 Receipt Total",
                f"{total_amount:,.2f}"
            )


        # =================================================
        # AUTOMATIC PRICE VERIFICATION
        # =================================================

        if items:

            st.header(
                "🔎 Automatic Price Verification"
            )

            st.write(
                "PriceProof AI automatically compares "
                "the extracted receipt items with the "
                "prototype reference database."
            )

            verification_results = []

            for item in items:

                product_name = item.get(
                    "product_name",
                    ""
                )

                brand = item.get(
                    "brand"
                )

                charged_price = item.get(
                    "charged_price"
                )

                matched_row, match_score = (
                    find_best_match(
                        product_name,
                        brand,
                        price_data
                    )
                )

                if matched_row is not None:

                    reference_price = float(
                        matched_row[
                            "reference_price"
                        ]
                    )

                    if charged_price is not None:

                        comparison = (
                            compare_price(
                                float(
                                    charged_price
                                ),
                                reference_price
                            )
                        )

                        verification_results.append({

                            "Receipt Product":
                                product_name,

                            "Brand":
                                brand
                                if brand
                                else "Not detected",

                            "Charged Price":
                                float(
                                    charged_price
                                ),

                            "Reference Price":
                                reference_price,

                            "Difference":
                                comparison[
                                    "difference"
                                ],

                            "Difference %":
                                comparison[
                                    "percentage_difference"
                                ],

                            "Status":
                                comparison[
                                    "status"
                                ],

                            "Match Confidence":
                                round(
                                    match_score * 100,
                                    1
                                )

                        })

                    else:

                        verification_results.append({

                            "Receipt Product":
                                product_name,

                            "Brand":
                                brand
                                if brand
                                else "Not detected",

                            "Charged Price":
                                "Not detected",

                            "
