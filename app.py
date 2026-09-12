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
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 18px;
    color: #666;
    margin-top: 0px;
}

.metric-card {
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #ddd;
    text-align: center;
    background-color: rgba(128,128,128,0.05);
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 20px;
}

.small-note {
    font-size: 13px;
    color: #777;
}

.verdict-box {
    padding: 20px;
    border-radius: 15px;
    margin: 10px 0px;
    border: 1px solid #ddd;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🧾 PriceProof AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Detect overpricing. Verify the price. Know your rights.</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a shopping receipt and PriceProof AI will extract products, "
    "compare prices with benchmark data, identify items that need review, "
    "and generate a consumer-friendly verification report."
)


# =========================================================
# SESSION STATE
# =========================================================

if "receipt_data" not in st.session_state:
    st.session_state.receipt_data = None

if "verification_results" not in st.session_state:
    st.session_state.verification_results = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "report_text" not in st.session_state:
    st.session_state.report_text = ""

if "uploaded_receipt_name" not in st.session_state:
    st.session_state.uploaded_receipt_name = None


# =========================================================
# HOW IT WORKS
# =========================================================

with st.expander("🔎 How PriceProof AI Works", expanded=False):

    cols = st.columns(5)

    with cols[0]:
        st.markdown("### 1️⃣")
        st.write("Upload Receipt")
        st.caption("Upload a clear receipt image.")

    with cols[1]:
        st.markdown("### 2️⃣")
        st.write("AI Reads It")
        st.caption("AI extracts products and prices.")

    with cols[2]:
        st.markdown("### 3️⃣")
        st.write("Verify Prices")
        st.caption("Compare against benchmark data.")

    with cols[3]:
        st.markdown("### 4️⃣")
        st.write("AI Explains")
        st.caption("Understand unusual differences.")

    with cols[4]:
        st.markdown("### 5️⃣")
        st.write("Generate Report")
        st.caption("Download your verification report.")


# =========================================================
# LOAD REFERENCE DATABASE
# =========================================================

@st.cache_data
def load_products():

    try:
        df = pd.read_csv("products.csv")

        df.columns = [
            str(col).strip().lower().replace(" ", "_")
            for col in df.columns
        ]

        return df

    except Exception as e:

        st.error(
            "Could not load products.csv. "
            "Make sure products.csv is uploaded to the same GitHub repository."
        )

        st.error(str(e))

        return pd.DataFrame()


products_df = load_products()

if products_df.empty:
    st.stop()


# =========================================================
# API CONNECTION
# =========================================================

api_key = st.secrets.get("GROQ_API_KEY", "")

groq_client = None

if api_key:

    try:

        groq_client = Groq(api_key=api_key)

    except Exception as e:

        st.warning("Groq API connection could not be initialized.")
        st.caption(str(e))

else:

    st.warning(
        "⚠️ GROQ_API_KEY is not configured. "
        "Receipt AI features will not work until the key is added to Streamlit Secrets."
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ PriceProof AI")

    st.write("### Project Status")

    if groq_client:
        st.success("🟢 Groq AI Connected")
    else:
        st.warning("🟡 AI Not Connected")

    st.divider()

    st.write("### Features")

    st.write("🧾 Receipt Analysis")
    st.write("🔍 Price Verification")
    st.write("📊 Price Health Score")
    st.write("💬 AI Consumer Chat")
    st.write("📄 Consumer Report")
    st.write("📥 Data Export")

    st.divider()

    st.caption(
        "PriceProof AI is a prototype benchmark tool. "
        "Reference prices may vary by store, location, date, "
        "brand, quantity, and market conditions."
    )


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text).lower()

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================================================
# SIMILARITY
# =========================================================

def calculate_similarity(text1, text2):

    text1 = normalize_text(text1)
    text2 = normalize_text(text2)

    if not text1 or not text2:
        return 0

    return SequenceMatcher(None, text1, text2).ratio()


# =========================================================
# SAFE NUMBER
# =========================================================

def safe_number(value, default=0):

    try:

        if value is None:
            return default

        if isinstance(value, str):

            value = value.replace(",", "")
            value = re.sub(r"[^\d.\-]", "", value)

        return float(value)

    except Exception:

        return default


# =========================================================
# FIND COLUMN
# =========================================================

def find_column(df, possible_names):

    for name in possible_names:

        if name in df.columns:
            return name

    return None


# =========================================================
# REFERENCE DATABASE COLUMNS
# =========================================================

product_name_column = find_column(
    products_df,
    [
        "product_name",
        "product",
        "name",
        "item",
        "item_name"
    ]
)

brand_column = find_column(
    products_df,
    [
        "brand",
        "brand_name"
    ]
)

price_column = find_column(
    products_df,
    [
        "reference_price",
        "market_price",
        "price",
        "reference",
        "benchmark_price"
    ]
)

if not product_name_column or not price_column:

    st.error(
        "products.csv must contain a product name column and a price column."
    )

    st.info(
        "Example columns: product_name, brand, reference_price"
    )

    st.stop()


# =========================================================
# PRODUCT MATCHING
# =========================================================

def find_best_match(product_name, brand=""):

    best_row = None
    best_score = 0

    for _, row in products_df.iterrows():

        reference_name = row.get(product_name_column, "")

        name_score = calculate_similarity(
            product_name,
            reference_name
        )

        reference_brand = ""

        if brand_column:
            reference_brand = row.get(brand_column, "")

        brand_score = 0

        if brand and reference_brand:

            brand_score = calculate_similarity(
                brand,
                reference_brand
            )

            total_score = (
                name_score * 0.70 +
                brand_score * 0.30
            )

        else:

            total_score = name_score

        if total_score > best_score:

            best_score = total_score
            best_row = row

    if best_row is None:
        return None, 0

    return best_row, round(best_score * 100, 2)


# =========================================================
# PRICE COMPARISON
# =========================================================

def compare_price(charged_price, reference_price):

    charged_price = safe_number(charged_price)
    reference_price = safe_number(reference_price)

    if reference_price <= 0:

        return {
            "difference": 0,
            "percentage": 0,
            "status": "Unable to Verify"
        }

    difference = charged_price - reference_price

    percentage = (
        difference / reference_price
    ) * 100

    if percentage >= 20:

        status = "Potentially Overpriced"

    elif percentage > 0:

        status = "Above Reference Price"

    else:

        status = "Within Reference Range"

    return {
        "difference": round(difference, 2),
        "percentage": round(percentage, 2),
        "status": status
    }


# =========================================================
# RECEIPT AI EXTRACTION
# =========================================================

def extract_receipt_data(uploaded_file):

    if not groq_client:
        raise Exception("Groq API is not connected.")

    image = Image.open(uploaded_file)

    image_format = image.format or "PNG"

    mime_type = (
        "image/jpeg"
        if image_format.upper() in ["JPG", "JPEG"]
        else "image/png"
    )

    uploaded_file.seek(0)

    image_bytes = uploaded_file.read()

    encoded_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    prompt = """
You are a receipt extraction AI.

Read the shopping receipt image carefully.

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

Rules:

1. Extract every visible purchasable item.
2. Preserve product names as accurately as possible.
3. Extract quantity when visible.
4. Extract the charged unit price when possible.
5. Extract line total when possible.
6. Do not invent missing information.
7. Use 0 when a numeric value cannot be identified.
8. Return JSON only.
"""

    response = groq_client.chat.completions.create(

        model="meta-llama/llama-4-scout-17b-16e-instruct",

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
                        "text": "Extract the receipt data from this image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{mime_type};base64,"
                                f"{encoded_image}"
                            )
                        }
                    }
                ]
            }
        ],

        temperature=0

    )

    content = response.choices[0].message.content

    content = content.strip()

    content = re.sub(
        r"```json",
        "",
        content,
        flags=re.IGNORECASE
    )

    content = re.sub(
        r"```",
        "",
        content
    )

    start = content.find("{")
    end = content.rfind("}")

    if start != -1 and end != -1:

        content = content[start:end + 1]

    data = json.loads(content)

    return data


# =========================================================
# AI PRICE EXPLANATION
# =========================================================

def analyze_with_ai(
    product_name,
    brand,
    charged_price,
    reference_price,
    percentage,
    status
):

    if not groq_client:
        return "AI explanation is unavailable because Groq is not connected."

    prompt = f"""
You are a consumer price verification assistant.

Explain this price comparison in simple language.

Product: {product_name}
Brand: {brand}
Charged price: {charged_price}
Reference price: {reference_price}
Difference percentage: {percentage}%
Status: {status}

Give a short explanation.

Do not make legal claims.
Do not say the store definitely committed an offense.
Explain that reference prices can vary.
Suggest a reasonable consumer action if appropriate.
"""

    response = groq_client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "system",
                "content": "You are a concise and helpful consumer assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2

    )

    return response.choices[0].message.content


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
            "AI chat is unavailable because the Groq API key "
            "is not connected."
        )

    context = {

        "receipt": receipt_data,

        "verification": verification_results

    }

    prompt = f"""
You are PriceProof AI, a consumer price verification assistant.

Answer the user's question using ONLY the supplied receipt
and verification information.

User question:
{question}

Data:
{json.dumps(context, indent=2, default=str)}

Rules:

- Be concise.
- Be clear.
- Use simple language.
- Do not invent information.
- Do not make legal claims.
- If the data does not contain the answer, say so.
- Explain uncertainty when necessary.
"""

    response = groq_client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[

            {
                "role": "system",
                "content": "You are a careful consumer assistant."
            },

            {
                "role": "user",
                "content": prompt
            }

        ],

        temperature=0.2

    )

    return response.choices[0].message.content


# =========================================================
# RECEIPT TOTAL CHECK
# =========================================================

def calculate_total_check(receipt_data):

    items = receipt_data.get("items", [])

    calculated_total = 0

    for item in items:

        line_total = safe_number(
            item.get("line_total", 0)
        )

        if line_total > 0:

            calculated_total += line_total

        else:

            quantity = safe_number(
                item.get("quantity", 1),
                1
            )

            charged_price = safe_number(
                item.get("charged_price", 0)
            )

            calculated_total += (
                quantity * charged_price
            )

    receipt_total = safe_number(
        receipt_data.get("total_amount", 0)
    )

    difference = receipt_total - calculated_total

    return (
        round(calculated_total, 2),
        round(receipt_total, 2),
        round(difference, 2)
    )


# =========================================================
# PRICE HEALTH SCORE
# =========================================================

def calculate_health_score(results):

    if not results:
        return 0

    score = 100

    matched = 0
    above = 0
    suspicious = 0

    for result in results:

        if result["confidence"] >= 60:

            matched += 1

            percentage = result["percentage"]

            if percentage > 0:
                above += 1

            if percentage >= 20:
                suspicious += 1

        else:

            score -= 5

    if matched > 0:

        above_ratio = above / matched
        suspicious_ratio = suspicious / matched

        score -= above_ratio * 20
        score -= suspicious_ratio * 35

    score = max(0, min(100, score))

    return round(score)


# =========================================================
# RECEIPT VERDICT
# =========================================================

def get_verdict(score):

    if score >= 85:

        return (
            "🟢 Good",
            "Most checked prices appear reasonable against the benchmark."
        )

    elif score >= 65:

        return (
            "🟡 Needs Review",
            "Some prices differ from the benchmark and may need checking."
        )

    else:

        return (
            "🔴 High Review Priority",
            "Several prices show significant differences from the benchmark."
        )


# =========================================================
# CONSUMER REPORT
# =========================================================

def generate_consumer_report(
    receipt_data,
    verification_results,
    health_score
):

    now = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    store = receipt_data.get(
        "store_name",
        "Unknown"
    )

    date = receipt_data.get(
        "date",
        "Unknown"
    )

    total = safe_number(
        receipt_data.get(
            "total_amount",
            0
        )
    )

    verdict, verdict_description = get_verdict(
        health_score
    )

    matched = sum(
        1
        for r in verification_results
        if r["confidence"] >= 60
    )

    suspicious = [
        r
        for r in verification_results
        if r["percentage"] >= 20
        and r["confidence"] >= 60
    ]

    above_reference = [
        r
        for r in verification_results
        if r["percentage"] > 0
        and r["confidence"] >= 60
    ]

    potential_extra = sum(
        max(0, r["difference"])
        for r in verification_results
        if r["confidence"] >= 60
    )

    lines = []

    lines.append(
        "PRICEPROOF AI - CONSUMER VERIFICATION REPORT"
    )

    lines.append("=" * 55)

    lines.append(
        f"Generated: {now}"
    )

    lines.append(
        f"Store: {store}"
    )

    lines.append(
        f"Receipt Date: {date}"
    )

    lines.append(
        f"Receipt Total: {total:.2f}"
    )

    lines.append("")

    lines.append("OVERALL ASSESSMENT")
    lines.append("-" * 30)

    lines.append(
        f"Price Health Score: {health_score}/100"
    )

    lines.append(
        f"Verdict: {verdict}"
    )

    lines.append(
        f"Summary: {verdict_description}"
    )

    lines.append("")

    lines.append("SUMMARY")
    lines.append("-" * 30)

    lines.append(
        f"Items checked: {len(verification_results)}"
    )

    lines.append(
        f"Matched items: {matched}"
    )

    lines.append(
        f"Items above reference: {len(above_reference)}"
    )

    lines.append(
        f"Potentially overpriced items: {len(suspicious)}"
    )

    lines.append(
        f"Potential extra paid: {potential_extra:.2f}"
    )

    lines.append("")

    lines.append("ITEM VERIFICATION")
    lines.append("-" * 30)

    for r in verification_results:

        lines.append(
            f"""
Product: {r['product']}
Brand: {r['brand']}
Quantity: {r['quantity']}
Charged Price: {r['charged_price']:.2f}
Reference Price: {r['reference_price']:.2f}
Difference: {r['difference']:.2f}
Difference %: {r['percentage']:.2f}%
Status: {r['status']}
Match Confidence: {r['confidence']:.2f}%
"""
        )

    lines.append("")

    lines.append("RECOMMENDED NEXT STEPS")
    lines.append("-" * 30)

    if suspicious:

        lines.append(
            "1. Recheck the receipt for the flagged items."
        )

        lines.append(
            "2. Compare the same product, brand and quantity "
            "with other local prices."
        )

        lines.append(
            "3. Ask the retailer to clarify significant differences."
        )

    else:

        lines.append(
            "1. Keep the receipt for your records."
        )

        lines.append(
            "2. Recheck prices if market conditions change."
        )

    lines.append("")

    lines.append("IMPORTANT NOTICE")
    lines.append("-" * 30)

    lines.append(
        "PriceProof AI uses benchmark/reference data for analysis. "
        "Reference prices are not guaranteed official government prices "
        "and can vary by location, date, retailer, brand, quantity and "
        "market conditions. A difference does not by itself prove illegal "
        "overcharging."
    )

    return "\n".join(lines)


# =========================================================
# RECEIPT UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">🧾 Upload Your Receipt</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a receipt image",
    type=["jpg", "jpeg", "png"],
    help="Use a clear and readable receipt image."
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
        "Benchmark data is for prototype verification purposes "
        "and may not represent official market prices."
    )


# =========================================================
# MAIN RECEIPT PROCESSING
# =========================================================

if uploaded_file:

    st.divider()

    left, right = st.columns([1, 2])

    with left:

        st.subheader("📷 Uploaded Receipt")

        st.image(
            uploaded_file,
            use_container_width=True
        )

    with right:

        if (
            st.session_state.uploaded_receipt_name
            != uploaded_file.name
        ):

            st.session_state.receipt_data = None
            st.session_state.verification_results = []
            st.session_state.chat_history = []
            st.session_state.report_text = ""

            st.session_state.uploaded_receipt_name = (
                uploaded_file.name
            )

        if st.session_state.receipt_data is None:

            if not groq_client:

                st.warning(
                    "Add your GROQ_API_KEY to use AI receipt extraction."
                )

            else:

                with st.spinner(
                    "🤖 AI is reading your receipt..."
                ):

                    try:

                        receipt_data = extract_receipt_data(
                            uploaded_file
                        )

                        st.session_state.receipt_data = receipt_data

                        st.success(
                            "✅ Receipt successfully analyzed."
                        )

                    except Exception as e:

                        st.error(
                            "Could not analyze the receipt."
                        )

                        st.code(str(e))

        receipt_data = st.session_state.receipt_data


    # =====================================================
    # RECEIPT INFORMATION
    # =====================================================

    if receipt_data:

        st.divider()

        st.subheader("📋 Receipt Information")

        info1, info2, info3 = st.columns(3)

        with info1:

            st.metric(
                "🏪 Store",
                receipt_data.get(
                    "store_name",
                    "Unknown"
                )
            )

        with info2:

            st.metric(
                "📅 Date",
                receipt_data.get(
                    "date",
                    "Unknown"
                )
            )

        with info3:

            st.metric(
                "💵 Receipt Total",
                f"{safe_number(receipt_data.get('total_amount', 0)):.2f}"
            )


        # =================================================
        # EXTRACTED ITEMS
        # =================================================

        st.subheader("📦 Extracted Items")

        extracted_items = receipt_data.get(
            "items",
            []
        )

        if extracted_items:

            extracted_df = pd.DataFrame(
                extracted_items
            )

            st.dataframe(
                extracted_df,
                use_container_width=True
            )

        else:

            st.warning(
                "No purchasable items were detected."
            )


        # =================================================
        # VERIFY PRODUCTS
        # =================================================

        st.subheader("🔍 Price Verification")

        if not st.session_state.verification_results:

            verification_results = []

            for item in extracted_items:

                product_name = item.get(
                    "product_name",
                    ""
                )

                brand = item.get(
                    "brand",
                    ""
                )

                quantity = safe_number(
                    item.get(
                        "quantity",
                        1
                    ),
                    1
                )

                charged_price = safe_number(
                    item.get(
                        "charged_price",
                        0
                    )
                )

                line_total = safe_number(
                    item.get(
                        "line_total",
                        0
                    )
                )

                best_row, confidence = find_best_match(
                    product_name,
                    brand
                )

                if (
                    best_row is not None
                    and confidence >= 60
                ):

                    reference_price = safe_number(
                        best_row.get(
                            price_column,
                            0
                        )
                    )

                    comparison = compare_price(
                        charged_price,
                        reference_price
                    )

                    reference_product = best_row.get(
                        product_name_column,
                        ""
                    )

                    reference_brand = ""

                    if brand_column:

                        reference_brand = best_row.get(
                            brand_column,
                            ""
                        )

                else:

                    reference_price = 0
                    comparison = {
                        "difference": 0,
                        "percentage": 0,
                        "status": "No Confident Match"
                    }

                    reference_product = ""
                    reference_brand = ""

                verification_results.append({

                    "product": product_name,

                    "brand": brand,

                    "quantity": quantity,

                    "charged_price": charged_price,

                    "line_total": line_total,

                    "reference_product": reference_product,

                    "reference_brand": reference_brand,

                    "reference_price": reference_price,

                    "difference": comparison["difference"],

                    "percentage": comparison["percentage"],

                    "status": comparison["status"],

                    "confidence": confidence

                })

            st.session_state.verification_results = (
                verification_results
            )

        verification_results = (
            st.session_state.verification_results
        )


        # =================================================
        # VERIFICATION TABLE
        # =================================================

        if verification_results:

            verification_df = pd.DataFrame(
                verification_results
            )

            display_columns = [

                "product",
                "brand",
                "quantity",
                "charged_price",
                "reference_price",
                "difference",
                "percentage",
                "status",
                "confidence"

            ]

            display_columns = [
                col
                for col in display_columns
                if col in verification_df.columns
            ]

            st.dataframe(
                verification_df[display_columns],
                use_container_width=True
            )


        # =================================================
        # ADVANCED PRICE HEALTH
        # =================================================

        st.divider()

        st.subheader("📊 Price Health Dashboard")

        health_score = calculate_health_score(
            verification_results
        )

        verdict, verdict_description = get_verdict(
            health_score
        )

        matched_items = [
            r
            for r in verification_results
            if r["confidence"] >= 60
        ]

        above_items = [
            r
            for r in matched_items
            if r["percentage"] > 0
        ]

        suspicious_items = [
            r
            for r in matched_items
            if r["percentage"] >= 20
        ]

        potential_extra = sum(
            max(0, r["difference"])
            for r in matched_items
        )

        potential_savings = sum(
            max(0, -r["difference"])
            for r in matched_items
        )

        most_suspicious = None

        if suspicious_items:

            most_suspicious = max(
                suspicious_items,
                key=lambda x: x["percentage"]
            )


        # =================================================
        # SCORE
        # =================================================

        score_col1, score_col2 = st.columns(
            [1, 2]
        )

        with score_col1:

            st.metric(
                "🛡️ Price Health Score",
                f"{health_score}/100"
            )

        with score_col2:

            st.markdown(
                f"""
                <div class="verdict-box">
                <h3>{verdict}</h3>
                <p>{verdict_description}</p>
                </div>
                """,
                unsafe_allow_html=True
            )


        # =================================================
        # DASHBOARD METRICS
        # =================================================

        m1, m2, m3, m4, m5 = st.columns(5)

        with m1:

            st.metric(
                "📦 Items Checked",
                len(verification_results)
            )

        with m2:

            st.metric(
                "🔗 Matched",
                len(matched_items)
            )

        with m3:

            st.metric(
                "⚠️ Above Reference",
                len(above_items)
            )

        with m4:

            st.metric(
                "🚨 Review Priority",
                len(suspicious_items)
            )

        with m5:

            st.metric(
                "💸 Potential Extra",
                f"{potential_extra:.2f}"
            )


        # =================================================
        # SAVINGS
        # =================================================

        st.info(
            f"💰 Potential benchmark-based savings: "
            f"**{potential_savings:.2f}**"
        )


        # =================================================
        # MOST SUSPICIOUS ITEM
        # =================================================

        if most_suspicious:

            st.warning(
                f"""
                🔎 **Item needing the most attention:**
                {most_suspicious['product']} — 
                {most_suspicious['percentage']:.2f}% above the reference price.
                """
            )


        # =================================================
        # RECEIPT TOTAL CHECK
        # =================================================

        st.subheader("🧮 Receipt Total Consistency Check")

        calculated_total, receipt_total, total_difference = (
            calculate_total_check(receipt_data)
        )

        t1, t2, t3 = st.columns(3)

        with t1:

            st.metric(
                "Calculated Item Total",
                f"{calculated_total:.2f}"
            )

        with t2:

            st.metric(
                "Receipt Total",
                f"{receipt_total:.2f}"
            )

        with t3:

            st.metric(
                "Difference",
                f"{total_difference:.2f}"
            )

        if abs(total_difference) <= 1:

            st.success(
                "✅ The extracted item totals are reasonably consistent with the receipt total."
            )

        else:

            st.warning(
                "⚠️ The extracted item totals differ from the receipt total. "
                "This may happen because of tax, discounts, missing items, "
                "rounding, or OCR/AI extraction limitations."
            )


        # =================================================
        # CHART
        # =================================================

        chart_rows = []

        for r in matched_items:

            chart_rows.append({

                "Product": r["product"],

                "Price Type": "Charged",

                "Price": r["charged_price"]

            })

            chart_rows.append({

                "Product": r["product"],

                "Price Type": "Reference",

                "Price": r["reference_price"]

            })

        if chart_rows:

            chart_df = pd.DataFrame(
                chart_rows
            )

            fig = px.bar(

                chart_df,

                x="Product",

                y="Price",

                color="Price Type",

                barmode="group",

                title="Charged Price vs Reference Price"

            )

            fig.update_layout(
                xaxis_title="Product",
                yaxis_title="Price"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        # =================================================
        # FLAGGED ITEMS
        # =================================================

        if suspicious_items:

            st.subheader(
                "🚨 Items Requiring Review"
            )

            for item in suspicious_items:

                with st.expander(
                    f"🔎 {item['product']} — {item['percentage']:.2f}% above reference"
                ):

                    st.write(
                        f"**Charged price:** {item['charged_price']:.2f}"
                    )

                    st.write(
                        f"**Reference price:** {item['reference_price']:.2f}"
                    )

                    st.write(
                        f"**Difference:** {item['difference']:.2f}"
                    )

                    st.write(
                        f"**Match confidence:** {item['confidence']:.2f}%"
                    )

                    if groq_client:

                        if st.button(
                            "🤖 Explain This Item",
                            key=f"explain_{item['product']}"
                        ):

                            with st.spinner(
                                "AI is analyzing..."
                            ):

                                try:

                                    explanation = analyze_with_ai(

                                        item["product"],

                                        item["brand"],

                                        item["charged_price"],

                                        item["reference_price"],

                                        item["percentage"],

                                        item["status"]

                                    )

                                    st.info(
                                        explanation
                                    )

                                except Exception as e:

                                    st.error(
                                        str(e)
                                    )


        # =================================================
        # AI CHAT
        # =================================================

        st.divider()

        st.subheader(
            "🤖 Ask PriceProof AI"
        )

        st.write(
            "Ask any question about this receipt or its verification results."
        )

        question = st.text_input(
            "Your question",
            placeholder=(
                "Example: Which item needs the most attention?"
            ),
            key="receipt_question"
        )

        ask_button = st.button(
            "🤖 Ask AI",
            use_container_width=False
        )

        if ask_button:

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            elif not groq_client:

                st.error(
                    "Groq API is not connected."
                )

            else:

                with st.spinner(
                    "🤖 Thinking..."
                ):

                    try:

                        answer = ask_receipt_ai(

                            question,

                            receipt_data,

                            verification_results

                        )

                        st.session_state.chat_history.append({

                            "question": question,

                            "answer": answer

                        })

                    except Exception as e:

                        st.error(
                            str(e)
                        )


        if st.session_state.chat_history:

            st.subheader(
                "💬 AI Conversation"
            )

            for index, chat in enumerate(
                reversed(st.session_state.chat_history)
            ):

                st.markdown(
                    f"**You:** {chat['question']}"
                )

                st.info(
                    chat["answer"]
                )

                if index < len(
                    st.session_state.chat_history
                ) - 1:

                    st.divider()


        # =================================================
        # CONSUMER REPORT
        # =================================================

        st.divider()

        st.subheader(
            "📄 Consumer Verification Report"
        )

        if st.button(
            "📄 Generate Consumer Report",
            use_container_width=True
        ):

            st.session_state.report_text = (
                generate_consumer_report(

                    receipt_data,

                    verification_results,

                    health_score

                )
            )

        if st.session_state.report_text:

            st.text_area(
                "Report Preview",
                st.session_state.report_text,
                height=450
            )

            st.download_button(

                label="📥 Download Consumer Report",

                data=st.session_state.report_text,

                file_name="priceproof_consumer_report.txt",

                mime="text/plain",

                use_container_width=True

            )


        # =================================================
        # DATA EXPORT
        # =================================================

        st.subheader(
            "📥 Export Verification Data"
        )

        export1, export2 = st.columns(2)

        export_df = pd.DataFrame(
            verification_results
        )

        with export1:

            csv_data = export_df.to_csv(
                index=False
            )

            st.download_button(

                label="📥 Download CSV",

                data=csv_data,

                file_name="priceproof_verification.csv",

                mime="text/csv",

                use_container_width=True

            )

        with export2:

            json_data = json.dumps(
                {
                    "receipt": receipt_data,
                    "verification": verification_results,
                    "price_health_score": health_score,
                    "verdict": verdict
                },
                indent=2,
                default=str
            )

            st.download_button(

                label="📥 Download JSON",

                data=json_data,

                file_name="priceproof_receipt_data.json",

                mime="application/json",

                use_container_width=True

            )


# =========================================================
# MANUAL PRICE VERIFICATION
# =========================================================

st.divider()

st.subheader(
    "⚡ Manual Price Verification"
)

st.write(
    "You can also verify a product manually without uploading a receipt."
)

product_options = products_df[
    product_name_column
].dropna().astype(str).tolist()

if product_options:

    selected_product = st.selectbox(
        "Select Product",
        product_options
    )

    selected_row = products_df[
        products_df[product_name_column].astype(str)
        == selected_product
    ].iloc[0]

    manual_reference_price = safe_number(
        selected_row[price_column]
    )

    manual_charged_price = st.number_input(
        "Charged Price",
        min_value=0.0,
        value=manual_reference_price,
        step=1.0
    )

    if st.button(
        "⚡ Verify Manual Price"
    ):

        manual_result = compare_price(
            manual_charged_price,
            manual_reference_price
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Reference Price",
                f"{manual_reference_price:.2f}"
            )

        with c2:

            st.metric(
                "Charged Price",
                f"{manual_charged_price:.2f}"
            )

        with c3:

            st.metric(
                "Difference",
                f"{manual_result['difference']:.2f}"
            )

        if manual_result["status"] == "Potentially Overpriced":

            st.error(
                f"🚨 {manual_result['status']} — "
                f"{manual_result['percentage']:.2f}% above reference."
            )

        elif manual_result["status"] == "Above Reference Price":

            st.warning(
                f"⚠️ {manual_result['status']} — "
                f"{manual_result['percentage']:.2f}% above reference."
            )

        else:

            st.success(
                f"✅ {manual_result['status']}"
            )

        if groq_client:

            with st.spinner(
                "AI is preparing an explanation..."
            ):

                try:

                    explanation = analyze_with_ai(

                        selected_product,

                        "",

                        manual_charged_price,

                        manual_reference_price,

                        manual_result["percentage"],

                        manual_result["status"]

                    )

                    st.info(
                        explanation
                    )

                except Exception as e:

                    st.error(
                        str(e)
                    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; padding:20px;">

    <b>PriceProof AI</b><br>

    AI-powered consumer price verification and receipt analysis

    <br><br>

    <span style="font-size:13px; color:#777;">
    Hackathon MVP • Benchmark-based analysis •
    Results should be independently verified
    </span>

    </div>
    """,
    unsafe_allow_html=True
)
