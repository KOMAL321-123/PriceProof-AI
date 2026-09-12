import streamlit as st
import pandas as pd
from PIL import Image
from groq import Groq
import base64
import json


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
    "Upload a shopping receipt and let AI extract the receipt information. "
    "You can then verify prices against our prototype reference database."
)

st.info(
    "⚠️ Reference prices are prototype market benchmarks, not official government prices. "
    "Actual prices may vary by city, shop, brand, package size, date, and promotions."
)


# =========================================================
# LOAD REFERENCE PRICE DATABASE
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

        # Read uploaded image
        image_bytes = uploaded_file.getvalue()

        # Convert image to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Get image type
        file_type = uploaded_file.type

        # Vision request
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

Extract only the information that is clearly visible.

Return the result using the required JSON structure.

Rules:
1. Do not invent information.
2. If store name is not visible, use null.
3. If date is not visible, use null.
4. Extract every clearly visible purchased item.
5. Extract product name exactly or as close as possible.
6. Extract quantity when visible.
7. Extract charged price for the item.
8. Extract line total when visible.
9. Extract the final receipt total when visible.
10. Do not include currency symbols inside numeric fields.
11. Use numbers for prices.
12. If a value cannot be determined, use null.
"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{file_type};base64,{image_base64}"
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
            },

            temperature=0.2,

            # FIXED: Groq output-token limit
            max_completion_tokens=900

        )

        result_text = response.choices[0].message.content

        result = json.loads(result_text)

        return result, None

    except Exception as e:

        return None, str(e)


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
        return "AI analysis is unavailable because the Groq API key is not configured."

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

Give a short, clear explanation for the consumer.

Requirements:
- Explain whether the charged price is above or within the reference benchmark.
- Do not claim that the shop has committed a crime.
- Do not give legal advice.
- Mention that the reference price is only a benchmark.
- Prices can vary because of location, shop, brand, package size, date, discounts, and promotions.
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

        st.error(f"Could not display the receipt image: {e}")
        st.stop()


    # =====================================================
    # EXTRACT RECEIPT DATA
    # =====================================================

    st.header("🤖 AI Receipt Analysis")

    with st.spinner("AI is reading your receipt..."):

        receipt_data, error_message = extract_receipt_data(
            uploaded_file
        )


    # =====================================================
    # EXTRACTION ERROR
    # =====================================================

    if error_message:

        st.error("Receipt analysis failed.")

        st.code(error_message)

        st.stop()


    # =====================================================
    # DISPLAY EXTRACTED INFORMATION
    # =====================================================

    if receipt_data:

        st.success("✅ Receipt information extracted successfully!")

        # -------------------------------------------------
        # STORE AND DATE
        # -------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            store_name = receipt_data.get("store_name")

            if store_name:
                st.metric(
                    "🏪 Store",
                    store_name
                )
            else:
                st.metric(
                    "🏪 Store",
                    "Not detected"
                )

        with col2:

            receipt_date = receipt_data.get("date")

            if receipt_date:
                st.metric(
                    "📅 Date",
                    receipt_date
                )
            else:
                st.metric(
                    "📅 Date",
                    "Not detected"
                )


        # -------------------------------------------------
        # PURCHASED ITEMS
        # -------------------------------------------------

        st.subheader("🛒 Purchased Items")

        items = receipt_data.get("items", [])

        if items:

            items_df = pd.DataFrame(items)

            st.dataframe(
                items_df,
                use_container_width=True
            )

        else:

            st.warning(
                "No purchased items could be detected from the receipt."
            )


        # -------------------------------------------------
        # RECEIPT TOTAL
        # -------------------------------------------------

        total_amount = receipt_data.get("total_amount")

        if total_amount is not None:

            st.metric(
                "💰 Receipt Total",
                f"{total_amount:,.2f}"
            )


        # =================================================
        # PRICE VERIFICATION
        # =================================================

        st.header("🔎 Price Verification")

        st.write(
            "Select a product from the reference database and enter "
            "the price charged on your receipt."
        )

        # -------------------------------------------------
        # PRODUCT SELECTION
        # -------------------------------------------------

        product_options = price_data["product_name"].tolist()

        selected_product = st.selectbox(
            "Select Product",
            product_options
        )

        # Get selected product information
        selected_rows = price_data[
            price_data["product_name"] == selected_product
        ]

        # Handle products with multiple brands
        if len(selected_rows) > 1:

            brand_options = selected_rows["brand"].tolist()

            selected_brand = st.selectbox(
                "Select Brand",
                brand_options
            )

            selected_row = selected_rows[
                selected_rows["brand"] == selected_brand
            ].iloc[0]

        else:

            selected_row = selected_rows.iloc[0]


        reference_price = float(
            selected_row["reference_price"]
        )

        st.info(
            f"Reference price for {selected_product}: "
            f"**{reference_price:,.2f}**"
        )


        # -------------------------------------------------
        # CHARGED PRICE
        # -------------------------------------------------

        charged_price = st.number_input(
            "Enter Charged Price",
            min_value=0.0,
            value=reference_price,
            step=1.0
        )


        # =================================================
        # PRICE COMPARISON
        # =================================================

        def compare_price(
            charged_price,
            reference_price
        ):

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


        # Run comparison
        result = compare_price(
            charged_price,
            reference_price
        )


        # =================================================
        # DISPLAY RESULTS
        # =================================================

        st.subheader("📊 Price Comparison")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Charged Price",
                f"{charged_price:,.2f}"
            )

        with col2:

            st.metric(
                "Reference Price",
                f"{reference_price:,.2f}"
            )

        with col3:

            st.metric(
                "Difference",
                f"{result['difference']:,.2f}"
            )


        # -------------------------------------------------
        # STATUS
        # -------------------------------------------------

        if result["status"] == "Potentially Overpriced":

            st.error(
                f"⚠️ {result['status']} "
                f"({result['percentage_difference']}% above reference)"
            )

        elif result["status"] == "Above Reference Price":

            st.warning(
                f"⚠️ {result['status']} "
                f"({result['percentage_difference']}% above reference)"
            )

        else:

            st.success(
                f"✅ {result['status']} "
                f"({result['percentage_difference']}% difference)"
            )


        # =================================================
        # AI EXPLANATION
        # =================================================

        st.subheader("🤖 AI Consumer Explanation")

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

        st.info(ai_analysis)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "PriceProof AI • GenAI-powered consumer price verification "
    "prototype • Reference prices are for demonstration purposes."
)
