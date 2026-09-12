import streamlit as st
import pandas as pd
from PIL import Image
from groq import Groq
import base64
import json
import re
import difflib
import plotly.express as px


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PriceProof AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS — PROFESSIONAL UX
# =========================================================

st.markdown(
    """
    <style>

    /* ---------- General ---------- */

    .stApp {
        background: #f7f9fc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    /* ---------- Hero ---------- */

    .hero {
        padding: 2.2rem 2.4rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #111827, #1f2937);
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    }

    .hero-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.12);
        color: #ffffff;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 0.4rem;
        margin-bottom: 0.7rem;
    }

    .hero-title {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0.3rem 0;
    }

    .hero-subtitle {
        color: #d1d5db;
        font-size: 1.15rem;
        margin-top: 0.7rem;
        max-width: 850px;
        line-height: 1.6;
    }

    /* ---------- Section Titles ---------- */

    .section-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #111827;
        margin-top: 1.4rem;
        margin-bottom: 0.35rem;
    }

    .section-description {
        color: #6b7280;
        font-size: 0.92rem;
        margin-bottom: 1rem;
    }

    /* ---------- Cards ---------- */

    .feature-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1.2rem;
        min-height: 145px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.035);
    }

    .feature-icon {
        font-size: 1.65rem;
        margin-bottom: 0.35rem;
    }

    .feature-title {
        font-weight: 750;
        font-size: 1rem;
        color: #111827;
    }

    .feature-text {
        color: #6b7280;
        font-size: 0.85rem;
        line-height: 1.45;
        margin-top: 0.35rem;
    }

    .summary-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1.15rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.035);
    }

    .summary-label {
        color: #6b7280;
        font-size: 0.78rem;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .summary-value {
        color: #111827;
        font-size: 1.65rem;
        font-weight: 800;
        margin-top: 0.25rem;
    }

    .summary-small {
        color: #6b7280;
        font-size: 0.78rem;
        margin-top: 0.15rem;
    }

    /* ---------- Status ---------- */

    .status-box {
        padding: 0.85rem 1rem;
        border-radius: 12px;
        margin: 0.6rem 0;
        font-weight: 650;
    }

    .status-success {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
    }

    .status-warning {
        background: #fffbeb;
        border: 1px solid #fde68a;
        color: #92400e;
    }

    .status-danger {
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #991b1b;
    }

    .status-info {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e40af;
    }

    /* ---------- Confidence ---------- */

    .confidence-bar {
        width: 100%;
        height: 8px;
        background: #e5e7eb;
        border-radius: 20px;
        overflow: hidden;
        margin-top: 0.4rem;
    }

    .confidence-fill {
        height: 100%;
        border-radius: 20px;
    }

    /* ---------- Upload ---------- */

    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 16px;
    }

    /* ---------- Buttons ---------- */

    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
    }

    .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 650;
    }

    /* ---------- Tables ---------- */

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ---------- Sidebar ---------- */

    [data-testid="stSidebar"] {
        background: #ffffff;
    }

    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.78rem;
        padding-top: 2rem;
        margin-top: 2rem;
    }

    /* ---------- Mobile ---------- */

    @media (max-width: 768px) {

        .hero-title {
            font-size: 2.1rem;
        }

        .hero {
            padding: 1.5rem;
        }

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero">

        <div>
            <span class="hero-badge">🤖 AI-POWERED</span>
            <span class="hero-badge">🧾 RECEIPT ANALYSIS</span>
            <span class="hero-badge">📊 PRICE VERIFICATION</span>
        </div>

        <div class="hero-title">
            PriceProof AI
        </div>

        <div class="hero-subtitle">
            Upload a receipt, let AI understand the products and prices,
            and instantly review how the charged prices compare with
            benchmark data.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HOW IT WORKS
# =========================================================

st.markdown(
    '<div class="section-title">⚡ How PriceProof AI Works</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">A simple workflow from receipt upload to price verification.</div>',
    unsafe_allow_html=True
)

feature_cols = st.columns(4)

features = [
    (
        "📷",
        "Read Receipt",
        "AI reads the uploaded receipt and extracts important information."
    ),
    (
        "🔎",
        "Match Products",
        "Extracted products are matched with the reference database."
    ),
    (
        "💰",
        "Compare Prices",
        "Charged prices are compared against benchmark prices."
    ),
    (
        "🧠",
        "Understand Results",
        "AI explains important differences in simple language."
    )
]

for col, feature in zip(feature_cols, features):
    icon, title, text = feature

    with col:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-text">{text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


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

    except Exception:
        return pd.DataFrame()


products_df = load_products()


# =========================================================
# GROQ CLIENT
# =========================================================

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    groq_client = Groq(api_key=GROQ_API_KEY)

except Exception:
    groq_client = None


# =========================================================
# SESSION STATE
# =========================================================

if "receipt_data" not in st.session_state:
    st.session_state.receipt_data = None

if "processed_items" not in st.session_state:
    st.session_state.processed_items = []

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

if "manual_results" not in st.session_state:
    st.session_state.manual_results = []


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def normalize_text(value):

    if value is None:
        return ""

    value = str(value).lower().strip()

    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value


def safe_float(value, default=0.0):

    try:

        if value is None:
            return default

        if isinstance(value, str):
            value = value.replace(",", "").replace("Rs.", "")
            value = value.replace("PKR", "").strip()

        return float(value)

    except Exception:
        return default


def safe_int(value, default=1):

    try:
        return max(1, int(float(value)))

    except Exception:
        return default


def similarity_score(a, b):

    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0

    return difflib.SequenceMatcher(None, a, b).ratio()


def prepare_image(image):

    image = image.convert("RGB")

    max_size = 2200

    if max(image.size) > max_size:

        ratio = max_size / max(image.size)

        new_size = (
            int(image.width * ratio),
            int(image.height * ratio)
        )

        image = image.resize(new_size)

    return image


def image_to_base64(image):

    from io import BytesIO

    buffer = BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=85,
        optimize=True
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# =========================================================
# RECEIPT EXTRACTION
# =========================================================

def extract_receipt(image):

    if groq_client is None:

        return {
            "error": "Groq API key is not configured."
        }

    image = prepare_image(image)

    image_base64 = image_to_base64(image)

    prompt = """
Analyze this receipt image carefully.

Extract the receipt information and return ONLY valid JSON.

Use this structure:

{
  "store": "",
  "date": "",
  "receipt_total": 0,
  "items": [
    {
      "product": "",
      "brand": "",
      "quantity": 1,
      "charged_price": 0,
      "line_total": 0
    }
  ]
}

Rules:

- Extract only information visible on the receipt.
- If a value is unknown, use an empty string or 0.
- charged_price means the price for one unit when possible.
- line_total means the total charged for that item.
- Do not add explanations outside JSON.
"""

    try:

        response = groq_client.chat.completions.create(

            model="qwen/qwen3.6-27b",

            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    "data:image/jpeg;base64,"
                                    + image_base64
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

        content = response.choices[0].message.content

        return json.loads(content)

    except Exception as e:

        return {
            "error": str(e)
        }


# =========================================================
# PRODUCT MATCHING
# =========================================================

def match_product(product_name):

    if products_df.empty:
        return None, 0

    possible_name_columns = [
        "product",
        "product_name",
        "name",
        "item",
        "item_name"
    ]

    name_column = None

    for col in possible_name_columns:

        if col in products_df.columns:
            name_column = col
            break

    if name_column is None:
        return None, 0

    best_index = None
    best_score = 0

    for index, row in products_df.iterrows():

        reference_name = row.get(name_column, "")

        score = similarity_score(
            product_name,
            reference_name
        )

        if score > best_score:

            best_score = score
            best_index = index

    if best_index is None:
        return None, 0

    return products_df.loc[best_index], best_score


# =========================================================
# REFERENCE PRICE EXTRACTION
# =========================================================

def get_reference_price(reference_row):

    if reference_row is None:
        return 0

    possible_price_columns = [
        "reference_price",
        "benchmark_price",
        "price",
        "unit_price",
        "market_price",
        "average_price"
    ]

    for col in possible_price_columns:

        if col in reference_row.index:

            value = safe_float(
                reference_row[col]
            )

            if value > 0:
                return value

    return 0


# =========================================================
# PROCESS RECEIPT ITEMS
# =========================================================

def process_items(receipt_data):

    processed = []

    items = receipt_data.get("items", [])

    for item in items:

        product_name = item.get(
            "product",
            ""
        )

        brand = item.get(
            "brand",
            ""
        )

        quantity = safe_int(
            item.get("quantity", 1)
        )

        charged_price = safe_float(
            item.get("charged_price", 0)
        )

        line_total = safe_float(
            item.get("line_total", 0)
        )

        reference_row, confidence = match_product(
            product_name
        )

        reference_price = get_reference_price(
            reference_row
        )

        difference = (
            charged_price - reference_price
        )

        difference_percent = 0

        if reference_price > 0:

            difference_percent = (
                difference / reference_price
            ) * 100

        total_difference = (
            difference * quantity
        )

        status = "No Reference"

        if reference_price > 0:

            if difference_percent > 10:
                status = "Review"

            elif difference_percent > 0:
                status = "Above Benchmark"

            elif difference_percent < -10:
                status = "Below Benchmark"

            else:
                status = "Within Range"

        processed.append(
            {
                "Product": product_name,
                "Brand": brand,
                "Quantity": quantity,
                "Charged Price": charged_price,
                "Reference Price": reference_price,
                "Difference": difference,
                "Difference %": difference_percent,
                "Total Difference": total_difference,
                "Match Confidence": confidence,
                "Status": status,
                "Line Total": line_total
            }
        )

    return processed


# =========================================================
# AI PRICE EXPLANATION
# =========================================================

def explain_price(item):

    if groq_client is None:
        return "AI explanation is unavailable because the Groq API key is not configured."

    prompt = f"""
Explain this product price comparison in simple language.

Product: {item.get("Product")}
Charged price: {item.get("Charged Price")}
Reference price: {item.get("Reference Price")}
Difference: {item.get("Difference")}
Difference percentage: {item.get("Difference %")}

Give a short explanation.
Mention if the price is higher, lower, or close to the benchmark.
Do not make legal accusations.
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

        return response.choices[0].message.content.strip()

    except Exception as e:

        return f"AI explanation unavailable: {e}"


# =========================================================
# AI RECEIPT CHAT
# =========================================================

def ask_receipt_ai(question):

    if groq_client is None:

        return "AI chat is unavailable because the Groq API key is not configured."

    receipt_data = st.session_state.receipt_data
    processed_items = st.session_state.processed_items

    context = {
        "receipt": receipt_data,
        "analysis": processed_items
    }

    prompt = f"""
You are PriceProof AI, a receipt and price-analysis assistant.

Answer the user's question using the receipt information below.

Receipt and analysis:
{json.dumps(context, indent=2, default=str)}

User question:
{question}

Rules:
- Answer directly.
- Be complete but concise.
- Avoid unnecessary information.
- If the receipt does not contain the answer, say so.
- Do not invent facts.
- Do not claim that a price difference proves illegal overcharging.
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

        return response.choices[0].message.content.strip()

    except Exception as e:

        return f"AI response unavailable: {e}"


# =========================================================
# PRICE HEALTH SCORE
# =========================================================

def calculate_health_score(items):

    if not items:
        return 0

    referenced_items = [
        item for item in items
        if item["Reference Price"] > 0
    ]

    if not referenced_items:
        return 0

    scores = []

    for item in referenced_items:

        difference_percent = abs(
            item["Difference %"]
        )

        if difference_percent <= 5:
            item_score = 100

        elif difference_percent <= 10:
            item_score = 85

        elif difference_percent <= 20:
            item_score = 65

        elif difference_percent <= 30:
            item_score = 45

        else:
            item_score = 25

        scores.append(item_score)

    return round(
        sum(scores) / len(scores)
    )


def get_verdict(score):

    if score >= 90:
        return "Excellent"

    if score >= 75:
        return "Good"

    if score >= 60:
        return "Needs Review"

    if score > 0:
        return "High Review Needed"

    return "Insufficient Data"


# =========================================================
# RECEIPT TOTAL CHECK
# =========================================================

def check_receipt_total(receipt_data, items):

    receipt_total = safe_float(
        receipt_data.get("receipt_total", 0)
    )

    calculated_total = sum(
        safe_float(item.get("Line Total", 0))
        for item in items
    )

    if receipt_total <= 0:
        return {
            "receipt_total": receipt_total,
            "calculated_total": calculated_total,
            "difference": 0,
            "status": "Unavailable"
        }

    difference = (
        receipt_total - calculated_total
    )

    tolerance = max(
        2,
        receipt_total * 0.01
    )

    if abs(difference) <= tolerance:
        status = "Consistent"
    else:
        status = "Review"

    return {
        "receipt_total": receipt_total,
        "calculated_total": calculated_total,
        "difference": difference,
        "status": status
    }


# =========================================================
# CONSUMER REPORT
# =========================================================

def generate_report():

    receipt = st.session_state.receipt_data
    items = st.session_state.processed_items

    score = calculate_health_score(items)
    verdict = get_verdict(score)

    total_check = check_receipt_total(
        receipt,
        items
    )

    report = []

    report.append("PRICEPROOF AI — CONSUMER VERIFICATION REPORT")
    report.append("=" * 55)
    report.append("")

    report.append(
        f"Store: {receipt.get('store', 'Unknown')}"
    )

    report.append(
        f"Date: {receipt.get('date', 'Unknown')}"
    )

    report.append(
        f"Receipt Total: {receipt.get('receipt_total', 0):,.2f}"
    )

    report.append("")

    report.append("PRICE HEALTH")
    report.append("-" * 30)

    report.append(
        f"Score: {score}/100"
    )

    report.append(
        f"Verdict: {verdict}"
    )

    report.append("")

    report.append("PRODUCT ANALYSIS")
    report.append("-" * 30)

    for item in items:

        report.append(
            f"Product: {item['Product']}"
        )

        report.append(
            f"Charged: {item['Charged Price']:,.2f}"
        )

        report.append(
            f"Reference: {item['Reference Price']:,.2f}"
        )

        report.append(
            f"Difference: {item['Difference']:,.2f} "
            f"({item['Difference %']:.1f}%)"
        )

        report.append(
            f"Status: {item['Status']}"
        )

        report.append("")

    report.append("RECEIPT TOTAL CHECK")
    report.append("-" * 30)

    report.append(
        f"Receipt total: {total_check['receipt_total']:,.2f}"
    )

    report.append(
        f"Calculated item total: "
        f"{total_check['calculated_total']:,.2f}"
    )

    report.append(
        f"Status: {total_check['status']}"
    )

    report.append("")

    report.append(
        "DISCLAIMER: Reference prices are benchmark data "
        "and may not represent official market prices. "
        "A price difference alone does not prove illegal "
        "overcharging."
    )

    return "\n".join(report)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 💰 PriceProof AI")

    st.caption(
        "AI-powered consumer price verification"
    )

    st.divider()

    st.markdown("### 📋 Analysis")

    st.write("1. Upload receipt")
    st.write("2. Extract receipt data")
    st.write("3. Match products")
    st.write("4. Compare prices")
    st.write("5. Review results")

    st.divider()

    st.markdown("### 🗂️ Reference Database")

    if products_df.empty:

        st.warning(
            "products.csv not found or empty."
        )

    else:

        st.success(
            f"{len(products_df)} reference records loaded"
        )

    st.divider()

    st.markdown("### 🤖 AI Models")

    st.caption(
        "Receipt Vision: qwen/qwen3.6-27b"
    )

    st.caption(
        "Analysis AI: openai/gpt-oss-20b"
    )

    st.divider()

    st.caption(
        "PriceProof AI • Consumer Intelligence"
    )


# =========================================================
# RECEIPT UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">📷 Upload Your Receipt</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">Upload a clear receipt image for AI-powered analysis.</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a receipt image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ],
    label_visibility="collapsed"
)


# =========================================================
# IMAGE PREVIEW + PROCESSING
# =========================================================

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file)

        st.session_state.uploaded_image = image

        preview_col, action_col = st.columns(
            [1, 1.5]
        )

        with preview_col:

            st.markdown("#### 🧾 Receipt Preview")

            st.image(
                image,
                use_container_width=True
            )

        with action_col:

            st.markdown("#### 🔍 Ready for Analysis")

            st.info(
                "Your receipt is ready. Click the button below "
                "to extract and verify its information."
            )

            analyze_button = st.button(
                "🚀 Analyze Receipt",
                type="primary",
                use_container_width=True
            )

            if analyze_button:

                with st.spinner(
                    "AI is reading and analyzing your receipt..."
                ):

                    receipt_data = extract_receipt(
                        image
                    )

                if "error" in receipt_data:

                    st.error(
                        "Receipt analysis failed."
                    )

                    st.code(
                        receipt_data["error"]
                    )

                    st.session_state.analysis_complete = False

                else:

                    processed_items = process_items(
                        receipt_data
                    )

                    st.session_state.receipt_data = (
                        receipt_data
                    )

                    st.session_state.processed_items = (
                        processed_items
                    )

                    st.session_state.chat_history = []

                    st.session_state.analysis_complete = True

                    st.success(
                        "Analysis completed successfully!"
                    )

                    st.rerun()


# =========================================================
# REFERENCE DATABASE PREVIEW
# =========================================================

if not st.session_state.analysis_complete:

    if not products_df.empty:

        with st.expander(
            "🗂️ Preview Reference Database"
        ):

            st.dataframe(
                products_df.head(10),
                use_container_width=True,
                hide_index=True
            )


# =========================================================
# RESULTS
# =========================================================

if st.session_state.analysis_complete:

    receipt = st.session_state.receipt_data
    items = st.session_state.processed_items

    st.divider()

    st.markdown(
        '<div class="section-title">✅ Verification Complete</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">Here is your receipt analysis at a glance.</div>',
        unsafe_allow_html=True
    )

    # =====================================================
    # RECEIPT INFORMATION
    # =====================================================

    info_cols = st.columns(4)

    with info_cols[0]:

        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">Store</div>
                <div class="summary-value">
                    {receipt.get("store", "Unknown")}
                </div>
                <div class="summary-small">
                    Receipt merchant
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with info_cols[1]:

        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">Date</div>
                <div class="summary-value">
                    {receipt.get("date", "Unknown")}
                </div>
                <div class="summary-small">
                    Receipt date
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with info_cols[2]:

        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">Items</div>
                <div class="summary-value">
                    {len(items)}
                </div>
                <div class="summary-small">
                    Products detected
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with info_cols[3]:

        receipt_total = safe_float(
            receipt.get("receipt_total", 0)
        )

        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">Receipt Total</div>
                <div class="summary-value">
                    Rs. {receipt_total:,.2f}
                </div>
                <div class="summary-small">
                    Total shown on receipt
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # PRICE HEALTH DASHBOARD
    # =====================================================

    st.markdown(
        '<div class="section-title">📊 Price Health Dashboard</div>',
        unsafe_allow_html=True
    )

    score = calculate_health_score(items)
    verdict = get_verdict(score)

    total_reference = sum(
        item["Reference Price"] * item["Quantity"]
        for item in items
        if item["Reference Price"] > 0
    )

    total_charged = sum(
        item["Charged Price"] * item["Quantity"]
        for item in items
        if item["Reference Price"] > 0
    )

    total_difference = (
        total_charged - total_reference
    )

    potential_extra = max(
        0,
        total_difference
    )

    potential_savings = max(
        0,
        -total_difference
    )

    review_items = [
        item for item in items
        if item["Status"] in [
            "Review",
            "Above Benchmark"
        ]
    ]

    score_cols = st.columns(4)

    with score_cols[0]:

        st.metric(
            "Price Health Score",
            f"{score}/100"
        )

    with score_cols[1]:

        st.metric(
            "Verdict",
            verdict
        )

    with score_cols[2]:

        st.metric(
            "Potential Extra",
            f"Rs. {potential_extra:,.2f}"
        )

    with score_cols[3]:

        st.metric(
            "Items to Review",
            len(review_items)
        )


    # =====================================================
    # SCORE STATUS
    # =====================================================

    if score >= 75:

        st.markdown(
            f"""
            <div class="status-box status-success">
                ✅ Overall price pattern looks reasonable based on
                the available benchmark data.
            </div>
            """,
            unsafe_allow_html=True
        )

    elif score >= 60:

        st.markdown(
            f"""
            <div class="status-box status-warning">
                ⚠️ Some prices deserve a closer review.
                The score is {score}/100.
            </div>
            """,
            unsafe_allow_html=True
        )

    elif score > 0:

        st.markdown(
            f"""
            <div class="status-box status-danger">
                🔎 Several price differences require attention.
                The score is {score}/100.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="status-box status-info">
                ℹ️ There is not enough reference data to calculate
                a meaningful price health score.
            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # TOTAL CONSISTENCY
    # =====================================================

    total_check = check_receipt_total(
        receipt,
        items
    )

    st.markdown(
        '<div class="section-title">🧮 Receipt Total Check</div>',
        unsafe_allow_html=True
    )

    total_cols = st.columns(3)

    with total_cols[0]:

        st.metric(
            "Receipt Total",
            f"Rs. {total_check['receipt_total']:,.2f}"
        )

    with total_cols[1]:

        st.metric(
            "Calculated Item Total",
            f"Rs. {total_check['calculated_total']:,.2f}"
        )

    with total_cols[2]:

        st.metric(
            "Difference",
            f"Rs. {total_check['difference']:,.2f}"
        )


    if total_check["status"] == "Consistent":

        st.markdown(
            """
            <div class="status-box status-success">
                ✅ The receipt total is consistent with the detected
                item totals within a small tolerance.
            </div>
            """,
            unsafe_allow_html=True
        )

    elif total_check["status"] == "Review":

        st.markdown(
            """
            <div class="status-box status-warning">
                ⚠️ The receipt total differs from the calculated
                item total. Review the receipt details.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.info(
            "Receipt total could not be fully checked."
        )


    # =====================================================
    # PRODUCT ANALYSIS
    # =====================================================

    st.markdown(
        '<div class="section-title">🛒 Product-by-Product Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">Review each detected product, benchmark price, difference, and matching confidence.</div>',
        unsafe_allow_html=True
    )

    if items:

        display_df = pd.DataFrame(items)

        table_df = display_df[
            [
                "Product",
                "Brand",
                "Quantity",
                "Charged Price",
                "Reference Price",
                "Difference",
                "Difference %",
                "Match Confidence",
                "Status"
            ]
        ].copy()

        table_df["Charged Price"] = table_df[
            "Charged Price"
        ].map(
            lambda x: f"Rs. {x:,.2f}"
        )

        table_df["Reference Price"] = table_df[
            "Reference Price"
        ].map(
            lambda x: (
                f"Rs. {x:,.2f}"
                if x > 0
                else "No reference"
            )
        )

        table_df["Difference"] = table_df[
            "Difference"
        ].map(
            lambda x: f"Rs. {x:,.2f}"
        )

        table_df["Difference %"] = table_df[
            "Difference %"
        ].map(
            lambda x: f"{x:+.1f}%"
        )

        table_df["Match Confidence"] = table_df[
            "Match Confidence"
        ].map(
            lambda x: f"{x * 100:.0f}%"
        )

        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True
        )


    # =====================================================
    # CHART
    # =====================================================

    if items:

        chart_df = pd.DataFrame(items)

        chart_df = chart_df[
            chart_df["Reference Price"] > 0
        ].copy()

        if not chart_df.empty:

            st.markdown(
                '<div class="section-title">📈 Charged vs Reference Prices</div>',
                unsafe_allow_html=True
            )

            chart_data = chart_df[
                [
                    "Product",
                    "Charged Price",
                    "Reference Price"
                ]
            ].copy()

            chart_data = chart_data.melt(
                id_vars=["Product"],
                value_vars=[
                    "Charged Price",
                    "Reference Price"
                ],
                var_name="Price Type",
                value_name="Price"
            )

            fig = px.bar(
                chart_data,
                x="Product",
                y="Price",
                color="Price Type",
                barmode="group",
                text_auto=".2f"
            )

            fig.update_layout(
                height=480,
                margin=dict(
                    l=20,
                    r=20,
                    t=30,
                    b=100
                ),
                xaxis_title="Product",
                yaxis_title="Price",
                legend_title="",
                hovermode="x unified"
            )

            fig.update_xaxes(
                tickangle=-35
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # =====================================================
    # SUSPICIOUS / REVIEW ITEMS
    # =====================================================

    st.markdown(
        '<div class="section-title">🚨 Items Worth Reviewing</div>',
        unsafe_allow_html=True
    )

    if review_items:

        st.markdown(
            '<div class="section-description">These items show comparatively higher differences from the available benchmark.</div>',
            unsafe_allow_html=True
        )

        for item in review_items:

            difference = item["Difference"]
            difference_percent = item["Difference %"]

            with st.container(border=True):

                left, middle, right = st.columns(
                    [2.2, 1, 1]
                )

                with left:

                    st.markdown(
                        f"### {item['Product']}"
                    )

                    if item["Brand"]:

                        st.caption(
                            f"Brand: {item['Brand']}"
                        )

                with middle:

                    st.metric(
                        "Charged",
                        f"Rs. {item['Charged Price']:,.2f}"
                    )

                with right:

                    st.metric(
                        "Difference",
                        f"{difference_percent:+.1f}%"
                    )

                if item["Reference Price"] > 0:

                    confidence = item[
                        "Match Confidence"
                    ]

                    st.caption(
                        f"Reference: Rs. "
                        f"{item['Reference Price']:,.2f} "
                        f" • Match confidence: "
                        f"{confidence * 100:.0f}%"
                    )

                    explanation_key = (
                        "explain_"
                        + normalize_text(
                            item["Product"]
                        ).replace(" ", "_")
                    )

                    if st.button(
                        "🧠 Explain this difference",
                        key=explanation_key
                    ):

                        with st.spinner(
                            "AI is explaining..."
                        ):

                            explanation = explain_price(
                                item
                            )

                        st.info(
                            explanation
                        )

    else:

        st.markdown(
            """
            <div class="status-box status-success">
                ✅ No major items were automatically flagged
                for review based on the available benchmark data.
            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # MOST SIGNIFICANT DIFFERENCE
    # =====================================================

    reference_items = [
        item for item in items
        if item["Reference Price"] > 0
    ]

    if reference_items:

        most_significant = max(
            reference_items,
            key=lambda x: abs(
                x["Difference %"]
            )
        )

        st.markdown(
            '<div class="section-title">🎯 Most Significant Difference</div>',
            unsafe_allow_html=True
        )

        significant_cols = st.columns(3)

        with significant_cols[0]:

            st.metric(
                "Product",
                most_significant["Product"]
            )

        with significant_cols[1]:

            st.metric(
                "Charged",
                f"Rs. "
                f"{most_significant['Charged Price']:,.2f}"
            )

        with significant_cols[2]:

            st.metric(
                "Difference",
                f"{most_significant['Difference %']:+.1f}%"
            )


    # =====================================================
    # AI CHAT
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-title">🤖 Ask PriceProof AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">Ask questions about this receipt and its price analysis.</div>',
        unsafe_allow_html=True
    )

    if st.session_state.chat_history:

        for message in st.session_state.chat_history:

            if message["role"] == "user":

                with st.chat_message("user"):
                    st.write(message["content"])

            else:

                with st.chat_message("assistant"):
                    st.write(message["content"])

    user_question = st.chat_input(
        "Ask something about this receipt..."
    )

    if user_question:

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                answer = ask_receipt_ai(
                    user_question
                )

            st.write(answer)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


    # =====================================================
    # MANUAL PRICE VERIFICATION
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-title">✍️ Manual Price Verification</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">Check a product manually against the reference database.</div>',
        unsafe_allow_html=True
    )

    manual_col1, manual_col2 = st.columns(
        [2, 1]
    )

    with manual_col1:

        manual_product = st.text_input(
            "Product name",
            placeholder="e.g. Nestle Milk"
        )

    with manual_col2:

        manual_price = st.number_input(
            "Charged price",
            min_value=0.0,
            step=10.0
        )

    if st.button(
        "🔎 Verify Manual Price",
        use_container_width=True
    ):

        if manual_product.strip():

            reference_row, confidence = match_product(
                manual_product
            )

            reference_price = get_reference_price(
                reference_row
            )

            if reference_price > 0:

                difference = (
                    manual_price - reference_price
                )

                difference_percent = (
                    difference / reference_price
                ) * 100

                st.success(
                    f"Reference price: "
                    f"Rs. {reference_price:,.2f}"
                )

                st.info(
                    f"Difference: Rs. "
                    f"{difference:,.2f} "
                    f"({difference_percent:+.1f}%)"
                )

                st.caption(
                    f"Match confidence: "
                    f"{confidence * 100:.0f}%"
                )

            else:

                st.warning(
                    "No suitable reference price was found."
                )

        else:

            st.warning(
                "Please enter a product name."
            )


    # =====================================================
    # DOWNLOADS
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-title">📥 Download Verification Results</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">Save your analysis in the format you need.</div>',
        unsafe_allow_html=True
    )

    download_col1, download_col2, download_col3 = st.columns(
        3
    )

    # ---------- CSV ----------

    results_df = pd.DataFrame(items)

    with download_col1:

        if not results_df.empty:

            csv_data = results_df.to_csv(
                index=False
            )

            st.download_button(
                "📊 Download CSV",
                data=csv_data,
                file_name="priceproof_analysis.csv",
                mime="text/csv",
                use_container_width=True
            )

    # ---------- JSON ----------

    with download_col2:

        json_data = json.dumps(
            {
                "receipt": receipt,
                "analysis": items
            },
            indent=2,
            default=str
        )

        st.download_button(
            "📄 Download JSON",
            data=json_data,
            file_name="priceproof_analysis.json",
            mime="application/json",
            use_container_width=True
        )

    # ---------- REPORT ----------

    with download_col3:

        report = generate_report()

        st.download_button(
            "📝 Download Report",
            data=report,
            file_name="priceproof_consumer_report.txt",
            mime="text/plain",
            use_container_width=True
        )


    # =====================================================
    # DISCLAIMER
    # =====================================================

    st.divider()

    st.markdown(
        """
        <div class="status-box status-info">

        <strong>ℹ️ Important:</strong><br><br>

        PriceProof AI uses benchmark/reference data to help
        consumers review receipt prices. Reference prices may
        differ from actual prices because of location, date,
        promotions, taxes, store policies, or other factors.

        A difference between a charged price and a benchmark
        does <strong>not</strong> by itself prove illegal
        overcharging or wrongdoing.

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        PriceProof AI • AI-Powered Consumer Price Verification
    </div>
    """,
    unsafe_allow_html=True
)
