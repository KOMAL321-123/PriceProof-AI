import streamlit as st
import pandas as pd
from PIL import Image
from groq import Groq
import base64
import json
import re
import io
from difflib import SequenceMatcher
import plotly.express as px


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PriceProof AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    /* HERO */
    .hero {
        padding: 34px 32px;
        border-radius: 22px;
        margin-bottom: 22px;
        border: 1px solid rgba(128,128,128,0.20);
        background: linear-gradient(
            135deg,
            rgba(30, 100, 255, 0.10),
            rgba(0, 180, 140, 0.08)
        );
    }

    .hero-title {
        font-size: 46px;
        font-weight: 800;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        font-size: 20px;
        font-weight: 500;
        margin-bottom: 10px;
    }

    .hero-description {
        font-size: 16px;
        opacity: 0.80;
        line-height: 1.6;
    }

    /* FEATURE CARDS */
    .feature-card {
        padding: 20px;
        border-radius: 17px;
        border: 1px solid rgba(128,128,128,0.18);
        min-height: 150px;
        margin-bottom: 10px;
    }

    .feature-icon {
        font-size: 28px;
        margin-bottom: 8px;
    }

    .feature-title {
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .feature-text {
        font-size: 14px;
        opacity: 0.75;
        line-height: 1.5;
    }

    /* DEMO CARDS */
    .demo-card {
        padding: 22px;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,0.20);
        min-height: 190px;
        margin-bottom: 12px;
    }

    .demo-icon {
        font-size: 32px;
        margin-bottom: 10px;
    }

    .demo-title {
        font-size: 19px;
        font-weight: 750;
        margin-bottom: 7px;
    }

    .demo-text {
        font-size: 14px;
        opacity: 0.78;
        line-height: 1.55;
    }

    /* IMPACT CARD */
    .impact-card {
        padding: 25px;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,0.20);
        margin: 8px 0;
        line-height: 1.65;
    }

    /* METRIC CARDS */
    .metric-card {
        padding: 20px;
        border-radius: 17px;
        border: 1px solid rgba(128,128,128,0.20);
        text-align: center;
        min-height: 125px;
    }

    .metric-title {
        font-size: 14px;
        opacity: 0.72;
        margin-bottom: 7px;
    }

    .metric-value {
        font-size: 27px;
        font-weight: 750;
    }

    /* SECTION LABEL */
    .section-label {
        font-size: 24px;
        font-weight: 750;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    /* STATUS BOX */
    .status-box {
        padding: 18px;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,0.20);
        margin: 8px 0 18px 0;
    }

    /* BADGE */
    .badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        border: 1px solid rgba(128,128,128,0.20);
        font-size: 13px;
        font-weight: 600;
        margin-right: 5px;
    }

    /* DISCLAIMER */
    .disclaimer {
        padding: 18px;
        border-radius: 15px;
        border: 1px solid rgba(255, 170, 0, 0.35);
        background: rgba(255, 170, 0, 0.07);
        line-height: 1.6;
        font-size: 14px;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.15);
    }

    /* BUTTONS */
    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
    }

    /* FILE UPLOADER */
    [data-testid="stFileUploader"] {
        border-radius: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="badge">🤖 AI-Powered</div>
        <div class="badge">💰 Price Verification</div>
        <div class="badge">📊 Consumer Intelligence</div>

        <div class="hero-title">
            💰 PriceProof AI
        </div>

        <div class="hero-subtitle">
            AI-Powered Consumer Price Verification
        </div>

        <div class="hero-description">
            Upload a shopping receipt and let AI extract products,
            compare prices against benchmark data, identify important
            price differences, and generate a clear consumer-friendly
            verification report.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HACKATHON DEMO MODE
# ============================================================

st.markdown(
    '<div class="section-label">🎯 Hackathon Demo</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="impact-card">

        <h3>💡 The Problem</h3>

        Consumers often receive a receipt but have no quick,
        simple way to understand whether the prices they paid
        are reasonable compared with available reference prices.

        <br><br>

        <h3>🚀 The Solution</h3>

        <strong>PriceProof AI</strong> uses AI to read a receipt,
        identify products and prices, compare them with benchmark
        data, highlight important differences, and explain the
        results in consumer-friendly language.

    </div>
    """,
    unsafe_allow_html=True
)


demo1, demo2, demo3 = st.columns(3)

with demo1:

    st.markdown(
        """
        <div class="demo-card">

            <div class="demo-icon">🧾</div>

            <div class="demo-title">
                1. Upload Receipt
            </div>

            <div class="demo-text">
                Upload a receipt image. AI reads the visible
                store, date, products, quantities and prices.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with demo2:

    st.markdown(
        """
        <div class="demo-card">

            <div class="demo-icon">🤖</div>

            <div class="demo-title">
                2. AI Verification
            </div>

            <div class="demo-text">
                Extracted products are matched against the
                reference database and price differences
                are calculated automatically.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with demo3:

    st.markdown(
        """
        <div class="demo-card">

            <div class="demo-icon">📊</div>

            <div class="demo-title">
                3. Understand Results
            </div>

            <div class="demo-text">
                View scores, suspicious items, charts,
                explanations, AI answers and downloadable
                verification reports.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


demo4, demo5, demo6 = st.columns(3)

with demo4:

    st.markdown(
        """
        <div class="demo-card">

            <div class="demo-icon">⚡</div>

            <div class="demo-title">
                Instant Comparison
            </div>

            <div class="demo-text">
                Quickly compare charged prices with benchmark
                prices and identify products that deserve review.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with demo5:

    st.markdown(
        """
        <div class="demo-card">

            <div class="demo-icon">💬</div>

            <div class="demo-title">
                Ask Your Own Questions
            </div>

            <div class="demo-text">
                Ask PriceProof AI anything about the analyzed
                receipt and receive concise, understandable answers.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with demo6:

    st.markdown(
        """
        <div class="demo-card">

            <div class="demo-icon">📄</div>

            <div class="demo-title">
                Consumer Report
            </div>

            <div class="demo-text">
                Download analysis results as CSV, JSON or
                a consumer-friendly verification report.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


st.info(
    "🏆 Demo message: PriceProof AI turns a simple receipt "
    "into actionable consumer price intelligence."
)


# ============================================================
# VALUE PROPOSITION
# ============================================================

st.markdown(
    '<div class="section-label">🚀 What PriceProof AI Does</div>',
    unsafe_allow_html=True
)

feature1, feature2, feature3, feature4 = st.columns(4)

with feature1:

    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">🧾</div>
            <div class="feature-title">Read Receipts</div>
            <div class="feature-text">
                AI extracts store information, products,
                quantities and prices from receipt images.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with feature2:

    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">🔎</div>
            <div class="feature-title">Match Products</div>
            <div class="feature-text">
                Extracted products are matched with
                benchmark reference data.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with feature3:

    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Compare Prices</div>
            <div class="feature-text">
                See charged prices, reference prices,
                percentage differences and trends.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with feature4:

    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">💡</div>
            <div class="feature-title">Understand Results</div>
            <div class="feature-text">
                Ask AI questions and generate a simple
                consumer verification report.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# IMPACT
# ============================================================

with st.expander("🏆 Why PriceProof AI Matters"):

    impact1, impact2, impact3 = st.columns(3)

    with impact1:

        st.markdown(
            """
            ### 👤 For Consumers

            - Understand receipt prices
            - Find products worth reviewing
            - Compare against reference data
            - Ask questions in natural language
            - Get a simple verification report
            """
        )

    with impact2:

        st.markdown(
            """
            ### 🤖 For AI Innovation

            - Multimodal receipt understanding
            - AI-powered information extraction
            - Intelligent product matching
            - Natural-language receipt analysis
            - Automated consumer insights
            """
        )

    with impact3:

        st.markdown(
            """
            ### 🚀 Future Potential

            - Larger price databases
            - Real-time market prices
            - More stores and retailers
            - Historical price tracking
            - Mobile consumer application
            """
        )


# ============================================================
# HOW IT WORKS
# ============================================================

with st.expander("🔎 How PriceProof AI Works"):

    step1, step2, step3 = st.columns(3)

    with step1:

        st.markdown(
            """
            ### 1️⃣ Upload
            Upload a clear shopping receipt.

            ### 2️⃣ Extract
            AI reads the receipt and extracts
            products, quantities and prices.

            ### 3️⃣ Match
            Products are compared with benchmark data.
            """
        )

    with step2:

        st.markdown(
            """
            ### 4️⃣ Analyze
            Price differences are calculated automatically.

            ### 5️⃣ Score
            A Price Health Score summarizes the result.

            ### 6️⃣ Explain
            AI explains important differences in simple language.
            """
        )

    with step3:

        st.markdown(
            """
            ### 7️⃣ Ask
            Ask your own questions about the receipt.

            ### 8️⃣ Verify
            Check receipt totals and manually verify prices.

            ### 9️⃣ Report
            Download CSV, JSON and consumer reports.
            """
        )

    st.info(
        "Benchmark prices are reference values only. "
        "A price difference does not automatically prove illegal overcharging."
    )


# ============================================================
# LOAD PRODUCTS DATABASE
# ============================================================

@st.cache_data
def load_products():

    try:

        df = pd.read_csv("products.csv")

        df.columns = [
            str(col).strip().lower().replace(" ", "_")
            for col in df.columns
        ]

        return df

    except Exception:

        return pd.DataFrame()


products_df = load_products()


# ============================================================
# GROQ CLIENT
# ============================================================

api_key = st.secrets.get(
    "GROQ_API_KEY",
    ""
)

if not api_key:

    st.error(
        "❌ GROQ_API_KEY is missing. Add it to Streamlit Cloud Secrets."
    )

    st.stop()


groq_client = Groq(
    api_key=api_key
)


# ============================================================
# SESSION STATE
# ============================================================

if "receipt_data" not in st.session_state:

    st.session_state.receipt_data = None


if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


if "processed" not in st.session_state:

    st.session_state.processed = False


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_text(text):

    if text is None:

        return ""

    try:

        if pd.isna(text):

            return ""

    except Exception:

        pass

    text = str(text).lower().strip()

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

    return text


def safe_float(
    value,
    default=0.0
):

    if value is None:

        return default

    if isinstance(
        value,
        (int, float)
    ):

        try:

            return float(value)

        except Exception:

            return default

    text = str(value)

    text = text.replace(",", "")
    text = text.replace("Rs.", "")
    text = text.replace("Rs", "")
    text = text.replace("PKR", "")

    match = re.search(
        r"-?\d+(?:\.\d+)?",
        text
    )

    if not match:

        return default

    try:

        return float(
            match.group()
        )

    except Exception:

        return default


def safe_int(
    value,
    default=1
):

    number = safe_float(
        value,
        default
    )

    try:

        return max(
            1,
            int(round(number))
        )

    except Exception:

        return default


def calculate_similarity(
    text1,
    text2
):

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
    product_name,
    brand=""
):

    if products_df.empty:

        return None, 0

    product_name = normalize_text(
        product_name
    )

    brand = normalize_text(
        brand
    )

    if not product_name:

        return None, 0

    best_row = None

    best_score = 0

    for _, row in products_df.iterrows():

        reference_product = row.get(
            "product_name",
            ""
        )

        reference_brand = row.get(
            "brand",
            ""
        )

        product_score = calculate_similarity(
            product_name,
            reference_product
        )

        if brand:

            brand_score = calculate_similarity(
                brand,
                reference_brand
            )

            final_score = (
                product_score * 0.75
                +
                brand_score * 0.25
            )

        else:

            final_score = product_score

        if final_score > best_score:

            best_score = final_score

            best_row = row

    return best_row, best_score


def compare_price(
    charged_price,
    reference_price
):

    charged_price = safe_float(
        charged_price
    )

    reference_price = safe_float(
        reference_price
    )

    if reference_price <= 0:

        return {
            "difference": 0,
            "percentage": 0,
            "status": "No Reference"
        }

    difference = (
        charged_price
        -
        reference_price
    )

    percentage = (
        difference
        /
        reference_price
    ) * 100

    if percentage > 10:

        status = "Higher"

    elif percentage < -10:

        status = "Lower"

    else:

        status = "Similar"

    return {
        "difference": difference,
        "percentage": percentage,
        "status": status
    }


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_image(
    uploaded_file
):

    image = Image.open(
        uploaded_file
    )

    image = image.convert(
        "RGB"
    )

    max_dimension = 2200

    if max(image.size) > max_dimension:

        ratio = (
            max_dimension
            /
            max(image.size)
        )

        new_size = (
            int(image.size[0] * ratio),
            int(image.size[1] * ratio)
        )

        image = image.resize(
            new_size
        )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=85,
        optimize=True
    )

    image_bytes = buffer.getvalue()

    encoded_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    return (
        encoded_image,
        "image/jpeg",
        image
    )


# ============================================================
# AI RECEIPT EXTRACTION
# ============================================================

def extract_receipt_data(
    uploaded_file
):

    try:

        (
            encoded_image,
            mime_type,
            preview_image
        ) = prepare_image(
            uploaded_file
        )

        prompt = """
Extract receipt information.

Return ONLY valid JSON.

Use this structure:

{
  "store_name": "",
  "date": "",
  "currency": "PKR",
  "receipt_total": 0,
  "items": [
    {
      "product_name": "",
      "brand": "",
      "quantity": 1,
      "charged_price": 0,
      "line_total": 0
    }
  ]
}

Rules:
- Extract clearly readable products.
- charged_price should be unit price when possible.
- line_total should be the item's total.
- Use 0 if a value cannot be read.
- Do not invent information.
- Keep values concise.
"""

        response = groq_client.chat.completions.create(

            model="qwen/qwen3.6-27b",

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
                            "text": "Extract the receipt data."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    f"data:{mime_type};"
                                    f"base64,{encoded_image}"
                                )
                            }
                        }
                    ]
                }
            ],

            temperature=0,

            max_completion_tokens=800,

            response_format={
                "type": "json_object"
            }
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if not content:

            raise ValueError(
                "AI returned an empty response."
            )

        data = json.loads(
            content
        )

        if "items" not in data:

            data["items"] = []

        if not isinstance(
            data["items"],
            list
        ):

            data["items"] = []

        return (
            data,
            preview_image
        )

    except Exception as e:

        raise Exception(
            f"Receipt extraction failed: {str(e)}"
        )


# ============================================================
# PROCESS RECEIPT ITEMS
# ============================================================

def process_items(
    receipt_data
):

    processed_items = []

    for item in receipt_data.get(
        "items",
        []
    ):

        product_name = item.get(
            "product_name",
            ""
        )

        brand = item.get(
            "brand",
            ""
        )

        quantity = safe_int(
            item.get(
                "quantity",
                1
            )
        )

        charged_price = safe_float(
            item.get(
                "charged_price",
                0
            )
        )

        line_total = safe_float(
            item.get(
                "line_total",
                0
            )
        )

        (
            reference_row,
            similarity
        ) = find_best_match(
            product_name,
            brand
        )

        reference_price = 0

        reference_product = ""

        reference_brand = ""

        if reference_row is not None:

            reference_price = safe_float(
                reference_row.get(
                    "reference_price",
                    0
                )
            )

            reference_product = str(
                reference_row.get(
                    "product_name",
                    ""
                )
            )

            reference_brand = str(
                reference_row.get(
                    "brand",
                    ""
                )
            )

        comparison = compare_price(
            charged_price,
            reference_price
        )

        processed_items.append(
            {
                "product_name": product_name,
                "brand": brand,
                "quantity": quantity,
                "charged_price": charged_price,
                "line_total": line_total,
                "reference_product": reference_product,
                "reference_brand": reference_brand,
                "reference_price": reference_price,
                "match_score": similarity,
                "difference": comparison["difference"],
                "percentage": comparison["percentage"],
                "status": comparison["status"]
            }
        )

    return processed_items


# ============================================================
# AI PRICE EXPLANATION
# ============================================================

def analyze_with_ai(
    item
):

    prompt = f"""
Explain this receipt price difference briefly.

Product: {item['product_name']}
Brand: {item['brand']}
Charged: PKR {item['charged_price']}
Reference: PKR {item['reference_price']}
Difference: {item['percentage']:.1f}%

Give a short consumer-friendly explanation.

Do not claim illegal overcharging.
Say that the benchmark is only a reference.
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

            max_completion_tokens=300
        )

        return (
            response
            .choices[0]
            .message
            .content
        )

    except Exception:

        return (
            "AI explanation unavailable. "
            f"The price difference is "
            f"{item['percentage']:.1f}%."
        )


# ============================================================
# RECEIPT AI CHAT
# ============================================================

def ask_receipt_ai(
    question,
    receipt_data,
    processed_items
):

    compact_items = []

    for item in processed_items:

        compact_items.append(
            {
                "product": item["product_name"],
                "brand": item["brand"],
                "quantity": item["quantity"],
                "charged": item["charged_price"],
                "reference": item["reference_price"],
                "difference_percent": round(
                    item["percentage"],
                    1
                ),
                "status": item["status"]
            }
        )

    context = {
        "store": receipt_data.get(
            "store_name",
            ""
        ),
        "date": receipt_data.get(
            "date",
            ""
        ),
        "receipt_total": receipt_data.get(
            "receipt_total",
            0
        ),
        "items": compact_items
    }

    prompt = f"""
You are PriceProof AI.

Receipt:
{json.dumps(context, ensure_ascii=False)}

Question:
{question}

Answer directly and briefly.

Use only the provided information.
Do not invent facts.
If something cannot be determined, say so.
A benchmark difference does not automatically mean illegal overcharging.
"""

    try:

        response = groq_client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer receipt questions "
                        "clearly and concisely."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2,

            max_completion_tokens=500
        )

        return (
            response
            .choices[0]
            .message
            .content
        )

    except Exception as e:

        return (
            f"AI response unavailable: {str(e)}"
        )


# ============================================================
# PRICE HEALTH SCORE
# ============================================================

def calculate_health_score(
    processed_items
):

    matched_items = [
        item
        for item in processed_items
        if item["reference_price"] > 0
    ]

    if not matched_items:

        return 0

    score = 100

    for item in matched_items:

        percentage = item["percentage"]

        if percentage > 20:

            score -= 15

        elif percentage > 10:

            score -= 8

        elif percentage > 5:

            score -= 3

        elif percentage < -10:

            score += 2

    return max(
        0,
        min(
            100,
            round(score)
        )
    )


def get_verdict(
    score
):

    if score >= 85:

        return "🟢 Good Price Health"

    if score >= 65:

        return "🟡 Mostly Reasonable"

    if score >= 45:

        return "🟠 Needs Review"

    return "🔴 Significant Differences"


# ============================================================
# RECEIPT TOTAL CHECK
# ============================================================

def calculate_total_consistency(
    receipt_data,
    processed_items
):

    receipt_total = safe_float(
        receipt_data.get(
            "receipt_total",
            0
        )
    )

    calculated_total = 0

    for item in processed_items:

        line_total = safe_float(
            item.get(
                "line_total",
                0
            )
        )

        quantity = safe_int(
            item.get(
                "quantity",
                1
            )
        )

        charged_price = safe_float(
            item.get(
                "charged_price",
                0
            )
        )

        if line_total > 0:

            calculated_total += line_total

        else:

            calculated_total += (
                charged_price
                *
                quantity
            )

    difference = (
        receipt_total
        -
        calculated_total
    )

    if receipt_total <= 0:

        status = "Unavailable"

    elif abs(difference) <= 2:

        status = "Consistent"

    else:

        status = "Review"

    return {
        "receipt_total": receipt_total,
        "calculated_total": calculated_total,
        "difference": difference,
        "status": status
    }


# ============================================================
# CONSUMER REPORT
# ============================================================

def generate_consumer_report(
    receipt_data,
    processed_items,
    score,
    verdict,
    total_check
):

    report = []

    report.append(
        "PRICEPROOF AI - CONSUMER VERIFICATION REPORT"
    )

    report.append(
        "=" * 55
    )

    report.append("")

    report.append(
        f"Store: {receipt_data.get('store_name', 'Unknown')}"
    )

    report.append(
        f"Date: {receipt_data.get('date', 'Unknown')}"
    )

    report.append(
        f"Receipt Total: PKR "
        f"{safe_float(receipt_data.get('receipt_total', 0)):,.2f}"
    )

    report.append(
        f"Price Health Score: {score}/100"
    )

    report.append(
        f"Verdict: {verdict}"
    )

    report.append("")

    report.append(
        "ITEM ANALYSIS"
    )

    report.append(
        "-" * 55
    )

    for item in processed_items:

        report.append(
            f"Product: {item['product_name']}"
        )

        report.append(
            f"Brand: {item['brand'] or 'N/A'}"
        )

        report.append(
            f"Quantity: {item['quantity']}"
        )

        report.append(
            f"Charged Unit Price: "
            f"PKR {item['charged_price']:,.2f}"
        )

        report.append(
            f"Reference Unit Price: "
            f"PKR {item['reference_price']:,.2f}"
        )

        report.append(
            f"Difference: "
            f"{item['percentage']:.1f}%"
        )

        report.append(
            f"Status: {item['status']}"
        )

        report.append("")

    report.append(
        "TOTAL CONSISTENCY"
    )

    report.append(
        "-" * 55
    )

    report.append(
        f"Receipt Total: "
        f"PKR {total_check['receipt_total']:,.2f}"
    )

    report.append(
        f"Calculated Total: "
        f"PKR {total_check['calculated_total']:,.2f}"
    )

    report.append(
        f"Difference: "
        f"PKR {total_check['difference']:,.2f}"
    )

    report.append(
        f"Status: {total_check['status']}"
    )

    report.append("")

    report.append(
        "DISCLAIMER"
    )

    report.append(
        "-" * 55
    )

    report.append(
        "Benchmark prices are reference values only."
    )

    report.append(
        "A price difference does not automatically "
        "prove illegal overcharging."
    )

    return "\n".join(
        report
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        # 💰 PriceProof AI

        **AI-powered consumer price verification**
        """
    )

    st.divider()

    st.markdown(
        "### 🏆 Hackathon Demo"
    )

    st.markdown(
        """
        **Problem:**  
        Consumers need a simple way to understand receipt prices.

        **Solution:**  
        AI reads receipts and compares prices with benchmark data.

        **Impact:**  
        Faster price awareness and easier consumer verification.
        """
    )

    st.divider()

    st.markdown(
        "### 🧠 AI Features"
    )

    st.markdown(
        """
        ✅ Receipt Reading  
        ✅ Product Matching  
        ✅ Price Comparison  
        ✅ Price Health Score  
        ✅ Suspicious Item Detection  
        ✅ Receipt Total Check  
        ✅ AI Explanations  
        ✅ AI Receipt Chat
        """
    )

    st.markdown(
        "### 📁 Consumer Tools"
    )

    st.markdown(
        """
        ✅ Consumer Report  
        ✅ CSV Export  
        ✅ JSON Export  
        ✅ Manual Verification
        """
    )

    st.divider()

    if products_df.empty:

        st.warning(
            "⚠️ Reference database unavailable"
        )

    else:

        st.success(
            f"📚 {len(products_df)} reference products loaded"
        )

    st.divider()

    st.caption(
        "PriceProof AI • Hackathon Prototype"
    )

    st.caption(
        "Benchmark data is for reference only."
    )


# ============================================================
# UPLOAD RECEIPT
# ============================================================

st.markdown(
    '<div class="section-label">📷 Upload Your Receipt</div>',
    unsafe_allow_html=True
)

st.write(
    "Start by uploading a clear receipt image. "
    "PriceProof AI will extract and analyze the visible information."
)

uploaded_file = st.file_uploader(
    "Choose a receipt image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    help="Upload a clear receipt image."
)


# ============================================================
# DATABASE PREVIEW
# ============================================================

with st.expander(
    "📚 View Reference Product Database"
):

    if products_df.empty:

        st.warning(
            "products.csv was not found."
        )

    else:

        st.write(
            f"Reference products available: "
            f"**{len(products_df)}**"
        )

        st.dataframe(
            products_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PROCESS RECEIPT
# ============================================================

if uploaded_file is not None:

    st.divider()

    col1, col2 = st.columns(
        [1, 1]
    )

    with col1:

        st.subheader(
            "🧾 Receipt Preview"
        )

        try:

            preview = Image.open(
                uploaded_file
            )

            st.image(
                preview,
                use_container_width=True
            )

        except Exception:

            st.error(
                "Unable to display this image."
            )

    with col2:

        st.subheader(
            "🤖 AI Verification"
        )

        st.info(
            "AI will extract readable receipt information "
            "and compare it with the benchmark database."
        )

        if st.button(
            "🚀 Analyze Receipt",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "🤖 AI is reading and analyzing your receipt..."
            ):

                try:

                    (
                        receipt_data,
                        preview_image
                    ) = extract_receipt_data(
                        uploaded_file
                    )

                    processed_items = process_items(
                        receipt_data
                    )

                    receipt_data["items"] = (
                        processed_items
                    )

                    st.session_state.receipt_data = (
                        receipt_data
                    )

                    st.session_state.processed = True

                    st.session_state.chat_history = []

                    st.success(
                        "✅ Receipt analyzed successfully!"
                    )

                except Exception as e:

                    st.error(
                        str(e)
                    )


# ============================================================
# RESULTS
# ============================================================

if (
    st.session_state.processed
    and st.session_state.receipt_data
):

    receipt_data = (
        st.session_state.receipt_data
    )

    processed_items = (
        receipt_data.get(
            "items",
            []
        )
    )

    st.divider()

    st.markdown(
        '<div class="section-label">📊 Verification Dashboard</div>',
        unsafe_allow_html=True
    )

    # ========================================================
    # DEMO RESULT SUMMARY
    # ========================================================

    st.success(
        "🎉 Demo result: Your receipt has been transformed "
        "into an AI-powered price verification analysis."
    )

    # ========================================================
    # RECEIPT INFO
    # ========================================================

    st.subheader(
        "🧾 Receipt Information"
    )

    info1, info2, info3, info4 = st.columns(4)

    with info1:

        st.metric(
            "Store",
            receipt_data.get(
                "store_name",
                "Unknown"
            )
        )

    with info2:

        st.metric(
            "Date",
            receipt_data.get(
                "date",
                "Unknown"
            )
        )

    with info3:

        st.metric(
            "Items",
            len(processed_items)
        )

    with info4:

        st.metric(
            "Receipt Total",
            f"PKR {safe_float(receipt_data.get('receipt_total', 0)):,.2f}"
        )

    # ========================================================
    # SCORE
    # ========================================================

    score = calculate_health_score(
        processed_items
    )

    verdict = get_verdict(
        score
    )

    st.subheader(
        "📊 Price Health"
    )

    score_col, verdict_col = st.columns(2)

    with score_col:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    Price Health Score
                </div>
                <div class="metric-value">
                    {score}/100
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with verdict_col:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    Receipt Verdict
                </div>
                <div class="metric-value">
                    {verdict}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.progress(
        score / 100
    )

    # ========================================================
    # EXTRA / SAVINGS
    # ========================================================

    potential_extra = 0

    potential_savings = 0

    for item in processed_items:

        quantity = safe_int(
            item["quantity"]
        )

        difference = safe_float(
            item["difference"]
        )

        total_difference = (
            difference
            *
            quantity
        )

        if total_difference > 0:

            potential_extra += (
                total_difference
            )

        elif total_difference < 0:

            potential_savings += abs(
                total_difference
            )

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Potential Extra Paid",
            f"PKR {potential_extra:,.2f}"
        )

    with metric2:

        st.metric(
            "Potential Savings",
            f"PKR {potential_savings:,.2f}"
        )

    with metric3:

        suspicious_count = sum(
            1
            for item in processed_items
            if item["percentage"] > 10
        )

        st.metric(
            "Items to Review",
            suspicious_count
        )

    # ========================================================
    # TOTAL CHECK
    # ========================================================

    total_check = (
        calculate_total_consistency(
            receipt_data,
            processed_items
        )
    )

    st.subheader(
        "🧮 Receipt Total Check"
    )

    if total_check["status"] == "Consistent":

        st.success(
            f"✅ Receipt total is consistent. "
            f"Difference: "
            f"PKR {total_check['difference']:,.2f}"
        )

    elif total_check["status"] == "Review":

        st.warning(
            f"⚠️ Receipt total needs review. "
            f"Difference: "
            f"PKR {total_check['difference']:,.2f}"
        )

    else:

        st.info(
            "ℹ️ Receipt total could not be fully verified."
        )

    # ========================================================
    # PRODUCT ANALYSIS
    # ========================================================

    st.subheader(
        "🛒 Product Price Analysis"
    )

    table_data = []

    for item in processed_items:

        table_data.append(
            {
                "Product": item["product_name"],
                "Brand": item["brand"],
                "Qty": item["quantity"],
                "Charged": item["charged_price"],
                "Reference": item["reference_price"],
                "Difference %": round(
                    item["percentage"],
                    1
                ),
                "Status": item["status"],
                "Match %": round(
                    item["match_score"] * 100,
                    1
                )
            }
        )

    if table_data:

        display_df = pd.DataFrame(
            table_data
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No products were extracted."
        )

    # ========================================================
    # CHART
    # ========================================================

    st.subheader(
        "📈 Charged vs Reference Prices"
    )

    chart_rows = []

    for item in processed_items:

        if item["reference_price"] > 0:

            chart_rows.append(
                {
                    "Product": item["product_name"],
                    "Charged Price": item["charged_price"],
                    "Reference Price": item["reference_price"]
                }
            )

    if chart_rows:

        chart_df = pd.DataFrame(
            chart_rows
        )

        chart_long = chart_df.melt(
            id_vars=["Product"],
            value_vars=[
                "Charged Price",
                "Reference Price"
            ],
            var_name="Price Type",
            value_name="Price"
        )

        fig = px.bar(
            chart_long,
            x="Product",
            y="Price",
            color="Price Type",
            barmode="group",
            title="Charged vs Reference Price"
        )

        fig.update_layout(
            xaxis_title="Product",
            yaxis_title="Price (PKR)",
            xaxis_tickangle=-45,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=100
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "No matched reference prices available."
        )

    # ========================================================
    # SUSPICIOUS ITEMS
    # ========================================================

    st.subheader(
        "🚨 Items Worth Reviewing"
    )

    suspicious_items = [
        item
        for item in processed_items
        if (
            item["percentage"] > 10
            and
            item["reference_price"] > 0
        )
    ]

    if suspicious_items:

        suspicious_items = sorted(
            suspicious_items,
            key=lambda x: x["percentage"],
            reverse=True
        )

        for index, item in enumerate(
            suspicious_items
        ):

            with st.expander(
                f"⚠️ {item['product_name']} — "
                f"{item['percentage']:.1f}% higher"
            ):

                detail1, detail2, detail3 = st.columns(3)

                with detail1:

                    st.metric(
                        "Charged",
                        f"PKR {item['charged_price']:,.2f}"
                    )

                with detail2:

                    st.metric(
                        "Reference",
                        f"PKR {item['reference_price']:,.2f}"
                    )

                with detail3:

                    st.metric(
                        "Difference",
                        f"{item['percentage']:.1f}%"
                    )

                st.write(
                    f"**Quantity:** {item['quantity']}"
                )

                st.write(
                    f"**Difference per unit:** "
                    f"PKR {item['difference']:,.2f}"
                )

                if st.button(
                    "🤖 Explain This Difference",
                    key=f"explain_{index}"
                ):

                    with st.spinner(
                        "AI is analyzing..."
                    ):

                        explanation = (
                            analyze_with_ai(
                                item
                            )
                        )

                    st.info(
                        explanation
                    )

    else:

        st.success(
            "✅ No major benchmark differences detected."
        )

    # ========================================================
    # MOST SIGNIFICANT DIFFERENCE
    # ========================================================

    st.subheader(
        "🔎 Most Significant Difference"
    )

    matched_items = [
        item
        for item in processed_items
        if item["reference_price"] > 0
    ]

    if matched_items:

        most_suspicious = max(
            matched_items,
            key=lambda x: x["percentage"]
        )

        st.info(
            f"**{most_suspicious['product_name']}** "
            f"has the largest positive price difference: "
            f"**{most_suspicious['percentage']:.1f}%**."
        )

    # ========================================================
    # AI CHAT
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-label">💬 Ask PriceProof AI</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Ask any question about your analyzed receipt."
    )

    for chat in st.session_state.chat_history:

        with st.chat_message(
            chat["role"]
        ):

            st.write(
                chat["content"]
            )

    question = st.chat_input(
        "Example: Which product has the highest price?"
    )

    if question:

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message(
            "user"
        ):

            st.write(
                question
            )

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Thinking..."
            ):

                answer = ask_receipt_ai(
                    question,
                    receipt_data,
                    processed_items
                )

            st.write(
                answer
            )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    # ========================================================
    # MANUAL VERIFICATION
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-label">✍️ Manual Price Verification</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Use this tool when you want to manually compare "
        "a charged price with a reference price."
    )

    manual_col1, manual_col2, manual_col3 = st.columns(3)

    with manual_col1:

        manual_product = st.text_input(
            "Product Name"
        )

    with manual_col2:

        manual_charged = st.number_input(
            "Charged Price",
            min_value=0.0,
            step=1.0
        )

    with manual_col3:

        manual_reference = st.number_input(
            "Reference Price",
            min_value=0.0,
            step=1.0
        )

    if st.button(
        "🔍 Verify Manual Price"
    ):

        if (
            manual_charged > 0
            and
            manual_reference > 0
        ):

            manual_result = compare_price(
                manual_charged,
                manual_reference
            )

            st.write(
                f"### {manual_product or 'Product'}"
            )

            st.write(
                f"Charged: "
                f"PKR {manual_charged:,.2f}"
            )

            st.write(
                f"Reference: "
                f"PKR {manual_reference:,.2f}"
            )

            st.write(
                f"Difference: "
                f"{manual_result['percentage']:.1f}%"
            )

            if manual_result["status"] == "Higher":

                st.warning(
                    "⚠️ Charged price is higher than the reference."
                )

            elif manual_result["status"] == "Lower":

                st.success(
                    "✅ Charged price is lower than the reference."
                )

            else:

                st.info(
                    "ℹ️ Prices are relatively similar."
                )

        else:

            st.warning(
                "Enter both charged and reference prices."
            )

    # ========================================================
    # DOWNLOADS
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-label">📥 Download Results</div>',
        unsafe_allow_html=True
    )

    report = generate_consumer_report(
        receipt_data,
        processed_items,
        score,
        verdict,
        total_check
    )

    download_col1, download_col2, download_col3 = st.columns(3)

    with download_col1:

        csv_df = pd.DataFrame(
            table_data
        )

        csv_data = csv_df.to_csv(
            index=False
        )

        st.download_button(
            "📄 Download CSV",
            data=csv_data,
            file_name="priceproof_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    with download_col2:

        json_data = json.dumps(
            receipt_data,
            indent=2,
            ensure_ascii=False
        )

        st.download_button(
            "🧾 Download JSON",
            data=json_data,
            file_name="priceproof_receipt.json",
            mime="application/json",
            use_container_width=True
        )

    with download_col3:

        st.download_button(
            "📋 Consumer Report",
            data=report,
            file_name="priceproof_consumer_report.txt",
            mime="text/plain",
            use_container_width=True
        )

    # ========================================================
    # FINAL DISCLAIMER
    # ========================================================

    st.divider()

    st.markdown(
        """
        <div class="disclaimer">

        <strong>⚠️ Important Disclaimer</strong><br><br>

        PriceProof AI uses benchmark/reference prices for comparison.
        Reference prices may differ from actual market prices,
        promotions, taxes, package sizes, or store-specific pricing.

        A price difference does <strong>not automatically establish
        illegal overcharging</strong>.

        PriceProof AI is designed as a consumer-awareness and
        price-verification tool.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<br><center><small>💰 PriceProof AI • AI-powered consumer price verification • Hackathon Demo</small></center>",
        unsafe_allow_html=True
    )
