import streamlit as st
import pandas as pd
from PIL import Image
from groq import Groq
import base64
import json
import re
from difflib import SequenceMatcher
from datetime import datetime
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
    "Upload a shopping receipt and let AI extract products, "
    "compare prices with reference market data, explain possible "
    "price differences, and generate a consumer report."
)


# =========================================================
# HOW PRICEPROOF AI WORKS
# =========================================================

with st.expander("🚀 How PriceProof AI Works", expanded=True):

    step1, step2, step3, step4, step5 = st.columns(5)

    with step1:
        st.markdown("### 1️⃣")
        st.markdown("**Upload Receipt**")
        st.caption("Upload your shopping receipt.")

    with step2:
        st.markdown("### 2️⃣")
        st.markdown("**AI Reads It**")
        st.caption("AI extracts products and prices.")

    with step3:
        st.markdown("### 3️⃣")
        st.markdown("**Verify Prices**")
        st.caption("Prices are compared with reference data.")

    with step4:
        st.markdown("### 4️⃣")
        st.markdown("**AI Explains**")
        st.caption("AI explains suspicious differences.")

    with step5:
        st.markdown("### 5️⃣")
        st.markdown("**Generate Report**")
        st.caption("Create a consumer verification report.")


# =========================================================
# LOAD REFERENCE DATABASE
# =========================================================

try:

    products_df = pd.read_csv("products.csv")

except Exception as e:

    st.error(f"Could not load products.csv: {e}")
    st.stop()


# =========================================================
# GROQ CLIENT
# =========================================================

groq_api_key = st.secrets.get("GROQ_API_KEY")

if groq_api_key:

    groq_client = Groq(
        api_key=groq_api_key
    )

else:

    groq_client = None


# =========================================================
# AI STATUS
# =========================================================

if groq_client:

    st.success(
        "🤖 AI services are connected."
    )

else:

    st.warning(
        "⚠️ Groq API key is not configured. "
        "AI features will not work until the secret is added."
    )


# =========================================================
# RECEIPT UPLOAD
# =========================================================

st.header("📤 Upload Your Receipt")

uploaded_file = st.file_uploader(
    "Choose a receipt image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# REFERENCE DATABASE
# =========================================================

with st.expander("📚 View Reference Price Database"):

    st.dataframe(
        products_df,
        use_container_width=True
    )


st.caption(
    "⚠️ Reference prices are prototype benchmark data for this "
    "hackathon demo. They are not official government prices and "
    "may vary by city, shop, date, brand, package size, promotions, "
    "and market conditions."
)


# =========================================================
# FUNCTION: EXTRACT RECEIPT DATA USING AI VISION
# =========================================================

def extract_receipt_data(uploaded_file):

    if not groq_client:

        return None, (
            "Groq API key is not configured."
        )

    try:

        image = Image.open(
            uploaded_file
        )

        image_format = (
            image.format.lower()
            if image.format
            else "jpeg"
        )

        if image_format == "jpg":

            image_format = "jpeg"

        image_bytes = uploaded_file.getvalue()

        encoded_image = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        prompt = """
You are a receipt extraction AI.

Read the uploaded shopping receipt carefully.

Extract:

1. Store name
2. Receipt date
3. Every purchased item
4. Product name
5. Brand if visible
6. Quantity
7. Charged price
8. Line total
9. Receipt total

Return ONLY valid JSON.

Use this exact structure:

{
  "store_name": "",
  "date": "",
  "items": [
    {
      "product_name": "",
      "brand": "",
      "quantity": 1,
      "charged_price": 0,
      "line_total": 0
    }
  ],
  "total_amount": 0
}

Important:
- Do not invent information.
- If something is not visible, use an empty string or 0.
- Keep product names readable.
- Return JSON only.
"""

        response = groq_client.chat.completions.create(

            model="qwen/qwen3.8-27b",

            messages=[

                {
                    "role": "system",
                    "content": prompt
                },

                {
                    "role": "user",
                    "content": [

                        {
                            "type": "text",
                            "text": "Extract the receipt information."
                        },

                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    f"data:image/{image_format};"
                                    f"base64,{encoded_image}"
                                )
                            }
                        }

                    ]
                }

            ],

            temperature=0.2,

            max_completion_tokens=900
        )

        raw_response = (
            response.choices[0]
            .message.content
        )

        cleaned_response = raw_response.strip()

        if cleaned_response.startswith("```"):

            cleaned_response = re.sub(
                r"```json|```",
                "",
                cleaned_response
            ).strip()

        receipt_data = json.loads(
            cleaned_response
        )

        return receipt_data, None

    except Exception as e:

        return None, str(e)


# =========================================================
# FUNCTION: NORMALIZE TEXT
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
    )

    return text.strip()


# =========================================================
# FUNCTION: CALCULATE SIMILARITY
# =========================================================

def calculate_similarity(
    text1,
    text2
):

    text1 = normalize_text(
        text1
    )

    text2 = normalize_text(
        text2
    )

    if not text1 or not text2:

        return 0

    return SequenceMatcher(
        None,
        text1,
        text2
    ).ratio()


# =========================================================
# FUNCTION: FIND BEST PRODUCT MATCH
# =========================================================

def find_best_match(
    product_name,
    brand=""
):

    best_match = None
    best_score = 0

    for _, row in products_df.iterrows():

        name_score = calculate_similarity(
            product_name,
            row["product_name"]
        )

        brand_score = 0

        if (
            brand
            and str(row["brand"]).strip()
        ):

            brand_score = calculate_similarity(
                brand,
                row["brand"]
            )

        if brand_score > 0:

            final_score = (
                name_score * 0.7
            ) + (
                brand_score * 0.3
            )

        else:

            final_score = name_score

        if final_score > best_score:

            best_score = final_score

            best_match = row

    if best_match is not None:

        return (
            best_match,
            round(best_score * 100, 2)
        )

    return None, 0


# =========================================================
# FUNCTION: COMPARE PRICE
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
        charged_price
        - reference_price
    )

    percentage_difference = (
        difference
        / reference_price
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
# FUNCTION: AI PRICE EXPLANATION
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
            "AI explanation is unavailable because "
            "the Groq API key is not configured."
        )

    prompt = f"""
You are PriceProof AI, a consumer price analysis assistant.

Explain this price comparison in simple language.

Product: {product_name}
Charged Price: {charged_price}
Reference Price: {reference_price}
Difference: {difference}
Percentage Difference: {percentage_difference}%
Status: {status}

Rules:
- Be concise.
- Do not make legal claims.
- Do not say the seller definitely violated the law.
- Explain that reference prices are benchmarks.
- Mention that actual prices can vary.
- Give useful consumer-friendly advice.
"""

    try:

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

        return (
            response.choices[0]
            .message.content
        )

    except Exception as e:

        return f"AI analysis error: {e}"


# =========================================================
# FUNCTION: AI CONSUMER CHAT
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

    prompt = f"""
You are PriceProof AI, a helpful consumer price analysis assistant.

Answer the user's question using ONLY the receipt information
and price verification results provided below.

Receipt data:
{json.dumps(receipt_data, indent=2)}

Price verification:
{json.dumps(verification_results, indent=2)}

User question:
{question}

Rules:
- Answer clearly and concisely.
- Use simple language.
- Do not invent information.
- Do not make legal claims.
- Do not say a seller definitely violated the law.
- Reference prices are benchmarks, not guaranteed official prices.
"""

    try:

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

        return (
            response.choices[0]
            .message.content
        )

    except Exception as e:

        return f"AI chat error: {e}"


# =========================================================
# FUNCTION: GENERATE CONSUMER REPORT
# =========================================================

def generate_consumer_report(
    receipt_data,
    verification_results
):

    generated_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    store_name = receipt_data.get(
        "store_name",
        "Not available"
    )

    receipt_date = receipt_data.get(
        "date",
        "Not available"
    )

    total_amount = receipt_data.get(
        "total_amount",
        0
    )

    items_checked = len(
        verification_results
    )

    matched_items = sum(
        1
        for item in verification_results
        if item.get(
            "match_confidence",
            0
        ) >= 60
    )

    above_reference = sum(
        1
        for item in verification_results
        if item.get(
            "percentage_difference",
            0
        ) > 0
    )

    potentially_overpriced = sum(
        1
        for item in verification_results
        if item.get("status")
        == "Potentially Overpriced"
    )

    potential_extra = sum(
        max(
            0,
            item.get(
                "difference",
                0
            )
        )
        for item in verification_results
    )

    report_lines = []

    report_lines.append(
        "============================================================"
    )

    report_lines.append(
        "                       PRICEPROOF AI"
    )

    report_lines.append(
        "            CONSUMER PRICE VERIFICATION REPORT"
    )

    report_lines.append(
        "============================================================"
    )

    report_lines.append("")

    report_lines.append(
        "Generated: "
        + generated_time
    )

    report_lines.append(
        "Store: "
        + str(store_name)
    )

    report_lines.append(
        "Receipt Date: "
        + str(receipt_date)
    )

    report_lines.append(
        f"Receipt Total: Rs. {total_amount}"
    )

    report_lines.append("")

    report_lines.append(
        "-------------------- SUMMARY --------------------"
    )

    report_lines.append(
        f"Items Checked: {items_checked}"
    )

    report_lines.append(
        f"Matched with Reference Database: {matched_items}"
    )

    report_lines.append(
        f"Items Above Reference Price: {above_reference}"
    )

    report_lines.append(
        f"Potentially Overpriced Items: "
        f"{potentially_overpriced}"
    )

    report_lines.append(
        f"Potential Extra Amount: "
        f"Rs. {round(potential_extra, 2)}"
    )

    report_lines.append("")

    report_lines.append(
        "---------------- ITEM VERIFICATION ----------------"
    )

    if verification_results:

        for index, item in enumerate(
            verification_results,
            start=1
        ):

            report_lines.append("")

            report_lines.append(
                f"{index}. "
                f"{item.get('receipt_product', 'Unknown Product')}"
            )

            report_lines.append(
                f"   Brand: "
                f"{item.get('receipt_brand', 'Not available')}"
            )

            report_lines.append(
                f"   Quantity: "
                f"{item.get('quantity', 1)}"
            )

            report_lines.append(
                f"   Charged Price: "
                f"Rs. {item.get('charged_price', 0)}"
            )

            report_lines.append(
                f"   Reference Product: "
                f"{item.get('matched_product', 'No match')}"
            )

            report_lines.append(
                f"   Reference Price: "
                f"Rs. {item.get('reference_price', 0)}"
            )

            report_lines.append(
                f"   Difference: "
                f"Rs. {item.get('difference', 0)}"
            )

            report_lines.append(
                f"   Difference %: "
                f"{item.get('percentage_difference', 0)}%"
            )

            report_lines.append(
                f"   Status: "
                f"{item.get('status', 'Unknown')}"
            )

            report_lines.append(
                f"   Match Confidence: "
                f"{item.get('match_confidence', 0)}%"
            )

    else:

        report_lines.append(
            "No verified items were available."
        )

    report_lines.append("")

    report_lines.append(
        "---------------- FLAGGED ITEMS ----------------"
    )

    flagged_items = [

        item
        for item in verification_results

        if item.get("status")
        == "Potentially Overpriced"

    ]

    if flagged_items:

        for item in flagged_items:

            report_lines.append(

                f"- {item.get('receipt_product', 'Unknown')}: "
                f"charged Rs. {item.get('charged_price', 0)}, "
                f"reference Rs. {item.get('reference_price', 0)}, "
                f"difference Rs. {item.get('difference', 0)} "
                f"({item.get('percentage_difference', 0)}%)"

            )

    else:

        report_lines.append(
            "No potentially overpriced items were detected "
            "using the current reference dataset."
        )

    report_lines.append("")

    report_lines.append(
        "---------------- CONSUMER SUMMARY ----------------"
    )

    if potentially_overpriced > 0:

        report_lines.append(

            f"PriceProof AI identified "
            f"{potentially_overpriced} item(s) with a "
            f"price difference of 20% or more above "
            f"the current reference benchmark."

        )

        report_lines.append(

            f"The estimated positive price difference "
            f"across checked items is "
            f"Rs. {round(potential_extra, 2)}."

        )

    else:

        report_lines.append(

            "No item reached the potentially overpriced "
            "threshold of 20% above the reference benchmark."

        )

    report_lines.append("")

    report_lines.append(
        "---------------- SUGGESTED NEXT ACTIONS ----------------"
    )

    report_lines.append(
        "1. Keep the original receipt as supporting evidence."
    )

    report_lines.append(
        "2. Confirm the product brand, package size, and quantity."
    )

    report_lines.append(
        "3. Check the current local market price for the same product."
    )

    report_lines.append(
        "4. If the difference remains significant, contact the "
        "retailer or relevant consumer protection authority."
    )

    report_lines.append("")

    report_lines.append(
        "---------------- IMPORTANT NOTICE ----------------"
    )

    report_lines.append(

        "This report is an AI-assisted price comparison based "
        "on the uploaded receipt and the prototype reference dataset."

    )

    report_lines.append(

        "Reference prices are benchmarks and are not official "
        "government prices or guaranteed legal prices."

    )

    report_lines.append(

        "Actual prices may vary because of location, date, shop, "
        "brand, package size, promotions, taxes, and market conditions."

    )

    report_lines.append(

        "A flagged item does not by itself prove illegal overcharging."

    )

    report_lines.append("")

    report_lines.append(
        "============================================================"
    )

    report_lines.append(
        "Generated by PriceProof AI"
    )

    report_lines.append(
        "Detect overpricing. Verify the price. Know your rights."
    )

    report_lines.append(
        "============================================================"
    )

    return "\n".join(
        report_lines
    )


# =========================================================
# MAIN RECEIPT PROCESSING
# =========================================================

if uploaded_file:

    st.header("🧾 Receipt Analysis")

    image = Image.open(
        uploaded_file
    )

    st.image(
        image,
        caption="Uploaded Receipt",
        use_container_width=True
    )

    st.divider()


    # =====================================================
    # EXTRACT RECEIPT
    # =====================================================

    with st.spinner(
        "🤖 AI is reading your receipt..."
    ):

        receipt_data, error_message = (
            extract_receipt_data(
                uploaded_file
            )
        )


    if error_message:

        st.error(
            f"Receipt extraction failed: "
            f"{error_message}"
        )

        st.stop()


    if not receipt_data:

        st.error(
            "No receipt data could be extracted."
        )

        st.stop()


    # =====================================================
    # RECEIPT INFORMATION
    # =====================================================

    st.subheader(
        "📋 Extracted Receipt Information"
    )

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:

        st.metric(
            "Store",
            receipt_data.get(
                "store_name",
                "Unknown"
            )
        )

    with info_col2:

        st.metric(
            "Receipt Date",
            receipt_data.get(
                "date",
                "Unknown"
            )
        )

    with info_col3:

        st.metric(
            "Total Amount",
            f"Rs. {receipt_data.get('total_amount', 0)}"
        )


    # =====================================================
    # EXTRACTED ITEMS
    # =====================================================

    items = receipt_data.get(
        "items",
        []
    )

    st.subheader(
        "🛒 Purchased Items"
    )

    if items:

        extracted_items_df = pd.DataFrame(
            items
        )

        st.dataframe(
            extracted_items_df,
            use_container_width=True
        )

    else:

        st.warning(
            "No purchased items were detected."
        )


    # =====================================================
    # AUTOMATIC PRICE VERIFICATION
    # =====================================================

    verification_results = []

    if items:

        st.header(
            "🔎 Automatic Price Verification"
        )

        for item in items:

            product_name = item.get(
                "product_name",
                ""
            )

            brand = item.get(
                "brand",
                ""
            )

            charged_price = float(
                item.get(
                    "charged_price",
                    0
                ) or 0
            )

            quantity = int(
                item.get(
                    "quantity",
                    1
                ) or 1
            )

            best_match, confidence = (
                find_best_match(
                    product_name,
                    brand
                )
            )


            if (
                best_match is not None
                and confidence >= 60
            ):

                reference_price = float(
                    best_match[
                        "reference_price"
                    ]
                )

                result = compare_price(
                    charged_price,
                    reference_price
                )

                verification_results.append(

                    {
                        "receipt_product": product_name,
                        "receipt_brand": brand,
                        "quantity": quantity,
                        "charged_price": charged_price,
                        "matched_product": best_match[
                            "product_name"
                        ],
                        "reference_price": reference_price,
                        "difference": result[
                            "difference"
                        ],
                        "percentage_difference": result[
                            "percentage_difference"
                        ],
                        "status": result[
                            "status"
                        ],
                        "match_confidence": confidence
                    }

                )

            else:

                verification_results.append(

                    {
                        "receipt_product": product_name,
                        "receipt_brand": brand,
                        "quantity": quantity,
                        "charged_price": charged_price,
                        "matched_product": "No confident match",
                        "reference_price": 0,
                        "difference": 0,
                        "percentage_difference": 0,
                        "status": "No Confident Match",
                        "match_confidence": confidence
                    }

                )


        # =================================================
        # VERIFICATION TABLE
        # =================================================

        verification_df = pd.DataFrame(
            verification_results
        )

        display_df = verification_df.rename(

            columns={

                "receipt_product":
                    "Receipt Product",

                "receipt_brand":
                    "Brand",

                "quantity":
                    "Quantity",

                "charged_price":
                    "Charged Price",

                "matched_product":
                    "Reference Product",

                "reference_price":
                    "Reference Price",

                "difference":
                    "Difference",

                "percentage_difference":
                    "Difference %",

                "status":
                    "Status",

                "match_confidence":
                    "Match Confidence %"
            }
        )

        st.dataframe(
            display_df,
            use_container_width=True
        )


        # =================================================
        # SMART PRICE DASHBOARD
        # =================================================

        st.header(
            "📊 Smart Price Dashboard"
        )

        items_checked = len(
            verification_results
        )

        matched_items = sum(

            1
            for item in verification_results
            if item["match_confidence"] >= 60

        )

        above_reference = sum(

            1
            for item in verification_results
            if item["percentage_difference"] > 0

        )

        potentially_overpriced = sum(

            1
            for item in verification_results
            if item["status"]
            == "Potentially Overpriced"

        )

        potential_extra = sum(

            max(
                0,
                item["difference"]
            )

            for item in verification_results

        )


        metric1, metric2, metric3, metric4, metric5 = (
            st.columns(5)
        )


        with metric1:

            st.metric(
                "Items Checked",
                items_checked
            )


        with metric2:

            st.metric(
                "Matched",
                matched_items
            )


        with metric3:

            st.metric(
                "Above Reference",
                above_reference
            )


        with metric4:

            st.metric(
                "Potentially Overpriced",
                potentially_overpriced
            )


        with metric5:

            st.metric(
                "Potential Extra",
                f"Rs. {round(potential_extra, 2)}"
            )


        # =================================================
        # PRICE CHART
        # =================================================

        chart_data = []

        for item in verification_results:

            if item["match_confidence"] >= 60:

                chart_data.append(

                    {
                        "Product":
                            item["receipt_product"],

                        "Charged Price":
                            item["charged_price"],

                        "Reference Price":
                            item["reference_price"]
                    }

                )


        if chart_data:

            chart_df = pd.DataFrame(
                chart_data
            )

            fig = px.bar(

                chart_df,

                x="Product",

                y=[
                    "Charged Price",
                    "Reference Price"
                ],

                barmode="group",

                title="Charged Price vs Reference Price"

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        # =================================================
        # PRICE INSIGHT
        # =================================================

        if potentially_overpriced > 0:

            st.warning(

                f"⚠️ {potentially_overpriced} item(s) "
                "are potentially overpriced based on "
                "the current reference benchmark."

            )

            st.subheader(
                "🔎 What Should You Check?"
            )

            flagged_items = [

                item
                for item in verification_results
                if item["status"]
                == "Potentially Overpriced"

            ]

            for item in flagged_items:

                difference = item[
                    "difference"
                ]

                percentage = item[
                    "percentage_difference"
                ]

                with st.expander(

                    f"⚠️ {item['receipt_product']} "
                    f"— {percentage}% above reference"

                ):

                    st.write(

                        f"**Charged Price:** "
                        f"Rs. {item['charged_price']}"

                    )

                    st.write(

                        f"**Reference Price:** "
                        f"Rs. {item['reference_price']}"

                    )

                    st.write(

                        f"**Price Difference:** "
                        f"Rs. {difference}"

                    )

                    st.write(

                        "PriceProof AI flagged this item because "
                        "the charged price is significantly higher "
                        "than the current reference benchmark."

                    )

                    st.info(

                        "💡 Check the product brand, package size, "
                        "quantity, receipt date, promotions, and "
                        "current local market price before making "
                        "a complaint."

                    )

        else:

            st.success(

                "✅ No potentially overpriced items were detected "
                "using the current reference benchmark."

            )


        # =================================================
        # AI CONSUMER CHAT
        # =================================================

        st.header(
            "💬 Ask PriceProof AI"
        )

        st.write(
            "Ask questions about your receipt and price analysis."
        )

        user_question = st.text_input(

            "Enter your question",

            placeholder=(
                "Example: Which item is most overpriced?"
            )

        )


        if st.button(
            "🤖 Ask AI"
        ):

            if user_question.strip():

                with st.spinner(
                    "AI is analyzing your receipt..."
                ):

                    chat_answer = ask_receipt_ai(

                        user_question,

                        receipt_data,

                        verification_results

                    )

                st.info(
                    chat_answer
                )

            else:

                st.warning(
                    "Please enter a question first."
                )


        # =================================================
        # CONSUMER REPORT
        # =================================================

        st.header(
            "📄 Consumer Price Verification Report"
        )

        st.write(

            "Generate a downloadable report containing "
            "the receipt analysis and price verification results."

        )


        if st.button(
            "📄 Generate Consumer Report"
        ):

            report_text = generate_consumer_report(

                receipt_data,

                verification_results

            )

            st.success(
                "✅ Consumer report generated successfully."
            )

            st.text_area(

                "Report Preview",

                report_text,

                height=500

            )

            st.download_button(

                label="⬇️ Download Consumer Report",

                data=report_text,

                file_name=(
                    "PriceProof_Consumer_Report.txt"
                ),

                mime="text/plain"

            )


    # =====================================================
    # MANUAL PRICE VERIFICATION
    # =====================================================

    st.divider()

    st.header(
        "🔍 Manual Price Verification"
    )

    st.write(

        "You can also manually check a product "
        "against the reference database."

    )


    selected_product = st.selectbox(

        "Select a product",

        products_df[
            "product_name"
        ].tolist()

    )


    selected_row = products_df[
        products_df[
            "product_name"
        ] == selected_product
    ].iloc[0]


    reference_price = float(
        selected_row[
            "reference_price"
        ]
    )


    charged_price = st.number_input(

        "Enter charged price",

        min_value=0.0,

        value=reference_price,

        step=1.0

    )


    if st.button(
        "⚡ Verify Manual Price"
    ):

        result = compare_price(

            charged_price,

            reference_price

        )


        st.metric(

            "Reference Price",

            f"Rs. {reference_price}"

        )


        st.metric(

            "Charged Price",

            f"Rs. {charged_price}"

        )


        st.metric(

            "Difference",

            f"Rs. {result['difference']}"

        )


        if result["status"] == "Potentially Overpriced":

            st.error(

                f"🚨 {result['status']} "
                f"({result['percentage_difference']}% above reference)"

            )

        elif result["status"] == "Above Reference Price":

            st.warning(

                f"⚠️ {result['status']} "
                f"({result['percentage_difference']}% above reference)"

            )

        else:

            st.success(

                f"✅ {result['status']}"

            )


        if groq_client:

            with st.spinner(

                "🤖 AI is analyzing the price difference..."

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


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(

    "PriceProof AI | AI-powered consumer price verification "
    "and receipt analysis | Hackathon MVP"

)
