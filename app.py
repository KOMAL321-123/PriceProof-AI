import streamlit as st
import pandas as pd
from PIL import Image
from groq import Groq
import base64
import json
import re
from difflib import SequenceMatcher
import plotly.express as px


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
    "Upload a shopping receipt and let AI extract receipt information, "
    "verify prices, visualize findings, and answer questions about your purchase."
)

st.info(
    "⚠️ Reference prices are prototype market benchmarks, not official "
    "government prices. Actual prices may vary by city, shop, brand, "
    "package size, date, and promotions."
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
        "🟡 AI system is not connected. "
        "Please add GROQ_API_KEY in Streamlit Secrets."
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
# AI RECEIPT CHAT
# =========================================================

def ask_receipt_ai(
    question,
    receipt_data,
    verification_results
):

    if not groq_client:

        return (
            "AI chat is unavailable because "
            "the Groq API key is not configured."
        )

    try:

        receipt_context = json.dumps(
            receipt_data,
            indent=2
        )

        verification_context = json.dumps(
            verification_results,
            indent=2
        )

        prompt = f"""
You are PriceProof AI, an AI consumer price assistant.

Answer the user's question using ONLY the receipt
information and price verification information provided below.

RECEIPT INFORMATION:
{receipt_context}

PRICE VERIFICATION RESULTS:
{verification_context}

USER QUESTION:
{question}

Rules:
1. Answer directly and clearly.
2. Keep the answer concise but complete.
3. Use simple language.
4. Do not invent missing information.
5. If the receipt does not contain enough information,
   clearly say that.
6. Treat reference prices as benchmarks, not official legal prices.
7. Do not claim that a shop committed a crime.
8. Do not provide legal advice.
9. If the user asks which item is most overpriced,
   compare the percentage differences.
10. If the user asks how much extra was paid,
    use the calculated differences where available.
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
            max_completion_tokens=600
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"AI chat could not be completed: {e}"


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

        verification_results = []

        if items:

            st.header(
                "🔎 Automatic Price Verification"
            )

            st.write(
                "PriceProof AI automatically compares "
                "the extracted receipt items with the "
                "prototype reference database."
            )

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

                            "Matched Product":
                                matched_row[
                                    "product_name"
                                ],

                            "Matched Brand":
                                matched_row[
                                    "brand"
                                ],

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

                            "Matched Product":
                                matched_row[
                                    "product_name"
                                ],

                            "Matched Brand":
                                matched_row[
                                    "brand"
                                ],

                            "Charged Price":
                                "Not detected",

                            "Reference Price":
                                reference_price,

                            "Difference":
                                "-",

                            "Difference %":
                                "-",

                            "Status":
                                "Price not detected",

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

                        "Matched Product":
                            "No match",

                        "Matched Brand":
                            "-",

                        "Charged Price":
                            charged_price
                            if charged_price is not None
                            else "Not detected",

                        "Reference Price":
                            "No match",

                        "Difference":
                            "-",

                        "Difference %":
                            "-",

                        "Status":
                            "No Reference Match",

                        "Match Confidence":
                            round(
                                match_score * 100,
                                1
                            )
                    })


            # =================================================
            # VERIFICATION TABLE
            # =================================================

            st.subheader(
                "📋 Verification Details"
            )

            verification_df = pd.DataFrame(
                verification_results
            )

            st.dataframe(
                verification_df,
                use_container_width=True
            )


            # =================================================
            # SMART DASHBOARD
            # =================================================

            st.header(
                "📊 PriceProof Dashboard"
            )

            valid_results = [
                result
                for result in verification_results
                if isinstance(
                    result["Reference Price"],
                    (int, float)
                )
                and isinstance(
                    result["Charged Price"],
                    (int, float)
                )
            ]

            potentially_overpriced = sum(
                1
                for result in valid_results
                if result["Status"]
                == "Potentially Overpriced"
            )

            above_reference = sum(
                1
                for result in valid_results
                if result["Status"]
                == "Above Reference Price"
            )

            matched_items = len(
                valid_results
            )

            total_items = len(
                verification_results
            )

            total_extra_amount = sum(
                max(
                    0,
                    float(
                        result["Difference"]
                    )
                )
                for result in valid_results
            )

            # -------------------------------------------------
            # DASHBOARD METRICS
            # -------------------------------------------------

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:

                st.metric(
                    "🛒 Items Checked",
                    total_items
                )

            with col2:

                st.metric(
                    "✅ Matched",
                    matched_items
                )

            with col3:

                st.metric(
                    "⚠️ Above Reference",
                    above_reference
                )

            with col4:

                st.metric(
                    "🚨 Potentially Overpriced",
                    potentially_overpriced
                )

            with col5:

                st.metric(
                    "💸 Potential Extra",
                    f"{total_extra_amount:,.2f}"
                )


            # =================================================
            # PRICE COMPARISON CHART
            # =================================================

            if valid_results:

                st.subheader(
                    "📈 Charged Price vs Reference Price"
                )

                chart_data = []

                for result in valid_results:

                    chart_data.append({

                        "Product":
                            result[
                                "Receipt Product"
                            ],

                        "Price Type":
                            "Charged Price",

                        "Price":
                            float(
                                result[
                                    "Charged Price"
                                ]
                            )
                    })

                    chart_data.append({

                        "Product":
                            result[
                                "Receipt Product"
                            ],

                        "Price Type":
                            "Reference Price",

                        "Price":
                            float(
                                result[
                                    "Reference Price"
                                ]
                            )
                    })

                chart_df = pd.DataFrame(
                    chart_data
                )

                fig = px.bar(
                    chart_df,
                    x="Product",
                    y="Price",
                    color="Price Type",
                    barmode="group",
                    title="Receipt Price Comparison"
                )

                fig.update_layout(
                    xaxis_title="Product",
                    yaxis_title="Price",
                    legend_title="Price Type"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


            # =================================================
            # OVERPRICING INSIGHT
            # =================================================

            if potentially_overpriced > 0:

                st.error(
                    f"🚨 {potentially_overpriced} item(s) "
                    "are significantly above the reference benchmark."
                )

            elif above_reference > 0:

                st.warning(
                    f"⚠️ {above_reference} item(s) "
                    "are above the reference benchmark."
                )

            else:

                st.success(
                    "✅ No significantly overpriced matched "
                    "items were detected."
                )


            st.info(
                "💡 'Potential Extra' is the sum of positive "
                "differences between charged prices and reference "
                "benchmarks. It is an estimate, not proof of illegal "
                "overcharging."
            )


        # =====================================================
        # AI CONSUMER CHAT
        # =====================================================

        st.header(
            "💬 Ask PriceProof AI"
        )

        st.write(
            "Ask questions about your receipt and its "
            "price-verification results."
        )

        st.caption(
            "Examples: Which item is most overpriced? • "
            "How much extra did I pay? • "
            "Explain this receipt in simple words."
        )

        user_question = st.text_input(
            "Enter your question",
            placeholder="Ask something about this receipt..."
        )

        if st.button(
            "🤖 Ask AI"
        ):

            if not user_question.strip():

                st.warning(
                    "Please enter a question first."
                )

            else:

                with st.spinner(
                    "AI is analyzing your receipt..."
                ):

                    answer = ask_receipt_ai(
                        user_question,
                        receipt_data,
                        verification_results
                    )

                st.subheader(
                    "💡 AI Answer"
                )

                st.info(
                    answer
                )


        # =====================================================
        # MANUAL PRICE VERIFICATION
        # =====================================================

        st.header(
            "🧮 Manual Price Verification"
        )

        st.write(
            "You can also manually check any product "
            "against the reference database."
        )

        product_options = (
            price_data[
                "product_name"
            ].tolist()
        )

        selected_product = st.selectbox(
            "Select Product",
            product_options
        )

        selected_rows = price_data[
            price_data[
                "product_name"
            ]
            == selected_product
        ]

        if len(selected_rows) > 1:

            brand_options = (
                selected_rows[
                    "brand"
                ].tolist()
            )

            selected_brand = st.selectbox(
                "Select Brand",
                brand_options
            )

            selected_row = selected_rows[
                selected_rows[
                    "brand"
                ]
                == selected_brand
            ].iloc[0]

        else:

            selected_row = (
                selected_rows.iloc[0]
            )

        reference_price = float(
            selected_row[
                "reference_price"
            ]
        )

        st.info(
            f"Reference price for "
            f"{selected_product}: "
            f"**{reference_price:,.2f}**"
        )

        charged_price = st.number_input(
            "Enter Charged Price",
            min_value=0.0,
            value=reference_price,
            step=1.0
        )

        result = compare_price(
            charged_price,
            reference_price
        )


        # =================================================
        # MANUAL COMPARISON RESULTS
        # =================================================

        st.subheader(
            "📊 Manual Comparison"
        )

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


        if (
            result["status"]
            == "Potentially Overpriced"
        ):

            st.error(
                f"⚠️ {result['status']} "
                f"({result['percentage_difference']}% "
                f"above reference)"
            )

        elif (
            result["status"]
            == "Above Reference Price"
        ):

            st.warning(
                f"⚠️ {result['status']} "
                f"({result['percentage_difference']}% "
                f"above reference)"
            )

        else:

            st.success(
                f"✅ {result['status']} "
                f"({result['percentage_difference']}% "
                f"difference)"
            )


        # =================================================
        # AI EXPLANATION
        # =================================================

        st.subheader(
            "🤖 AI Consumer Explanation"
        )

        with st.spinner(
            "AI is analyzing the price difference..."
        ):

            ai_analysis = analyze_with_ai(

                selected_product,

                charged_price,

                reference_price,

                result["difference"],

                result[
                    "percentage_difference"
                ],

                result["status"]
            )

        st.info(
            ai_analysis
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "PriceProof AI • GenAI-powered consumer price "
    "verification prototype • Reference prices are "
    "for demonstration purposes."
)
