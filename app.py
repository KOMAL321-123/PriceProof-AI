```python
import streamlit as st
import pandas as pd
from PIL import Image
from groq import Groq
import base64
import json
import re
import difflib
import plotly.express as px


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PriceProof AI",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL UI
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background: #f7f9fc;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }

        h1, h2, h3 {
            letter-spacing: -0.3px;
        }

        .hero {
            background: linear-gradient(
                135deg,
                #ffffff 0%,
                #f1f5ff 100%
            );
            border: 1px solid #e4e9f2;
            border-radius: 18px;
            padding: 28px;
            margin-bottom: 20px;
        }

        .hero-title {
            font-size: 2.4rem;
            font-weight: 800;
            margin-bottom: 6px;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            color: #5f6b7a;
            line-height: 1.6;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 750;
            margin-top: 12px;
            margin-bottom: 12px;
        }

        .status-card {
            border-radius: 14px;
            padding: 18px;
            border: 1px solid #e5e9f0;
            background: white;
            min-height: 105px;
        }

        .status-label {
            color: #6b7280;
            font-size: 0.85rem;
            margin-bottom: 4px;
        }

        .status-value {
            font-size: 1.35rem;
            font-weight: 750;
        }

        .info-box {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 18px;
            margin: 10px 0;
        }

        .success-box {
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 14px;
            padding: 16px;
        }

        .warning-box {
            background: #fffbeb;
            border: 1px solid #fde68a;
            border-radius: 14px;
            padding: 16px;
        }

        .danger-box {
            background: #fef2f2;
            border: 1px solid #fecaca;
            border-radius: 14px;
            padding: 16px;
        }

        .metric-note {
            color: #6b7280;
            font-size: 0.8rem;
            margin-top: 3px;
        }

        .small-muted {
            color: #6b7280;
            font-size: 0.88rem;
        }

        .footer {
            text-align: center;
            color: #7b8491;
            font-size: 0.82rem;
            padding-top: 28px;
            padding-bottom: 10px;
        }

        [data-testid="stMetric"] {
            background: white;
            border: 1px solid #e5e9f0;
            border-radius: 14px;
            padding: 12px;
        }

        [data-testid="stFileUploader"] {
            background: white;
            border-radius: 14px;
            padding: 8px;
        }

        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
        }

        div[data-testid="stExpander"] {
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# OPTIONAL INTRODUCTION
# ============================================================

with st.expander("ℹ️ About PriceProof AI"):
    st.markdown(
        """
        ### PriceProof AI

        **PriceProof AI** is an AI-powered receipt verification assistant.

        Upload a receipt and the system can:

        - 🧾 Extract receipt information
        - 🔎 Identify products
        - 💰 Compare charged prices with benchmark prices
        - 📊 Calculate price differences
        - 🧠 Explain unusual price differences
        - 💬 Answer questions about the receipt
        - 📥 Generate downloadable verification reports

        **Important:** Benchmark prices are reference values and may not represent
        official market prices.
        """
    )


with st.expander("⚡ How PriceProof AI Works"):
    st.markdown(
        """
        **1. Upload Receipt** → Add a receipt image.

        **2. AI Extraction** → The vision model reads important receipt information.

        **3. Product Matching** → Extracted products are matched with the reference database.

        **4. Price Comparison** → Charged prices are compared with benchmark prices.

        **5. Verification** → The system calculates differences and a price-health score.

        **6. AI Explanation** → AI can explain significant price differences.

        **7. Ask Questions** → Ask your own questions about the receipt.

        **8. Manual Verification** → You can manually verify a product price.

        **9. Download Results** → Export CSV, JSON, or a consumer report.
        """
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🧾 PriceProof AI</div>
        <div class="hero-subtitle">
            AI-powered consumer price verification from receipt images.
            Upload a receipt, review extracted products, compare prices,
            and understand the results.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD REFERENCE DATABASE
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

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = None


if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "analysis_complete": False,
    "receipt_data": None,
    "processed_items": [],
    "receipt_image": None,
    "chat_history": [],
    "manual_results": {},
    "last_report": "",
    "last_error": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        if isinstance(value, str):
            value = (
                value.replace(",", "")
                .replace("Rs.", "")
                .replace("Rs", "")
                .replace("PKR", "")
                .strip()
            )

        return float(value)

    except Exception:
        return default


def safe_int(value, default=1):
    try:
        if value is None:
            return default

        return max(1, int(float(value)))

    except Exception:
        return default


def normalize_text(value):
    if value is None:
        return ""

    value = str(value).lower().strip()

    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def similarity_score(a, b):
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0

    return difflib.SequenceMatcher(None, a, b).ratio()


def find_column(df, possible_names):
    if df.empty:
        return None

    columns = list(df.columns)

    for name in possible_names:
        if name in columns:
            return name

    return None


def prepare_image(image):
    image = image.copy()

    max_size = 2200

    if image.width > max_size or image.height > max_size:
        image.thumbnail((max_size, max_size))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def image_to_base64(image):
    import io

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=85,
        optimize=True
    )

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ============================================================
# REFERENCE MATCHING
# ============================================================

def match_product(product_name, df):
    if df.empty:
        return None, 0

    product_col = find_column(
        df,
        [
            "product",
            "product_name",
            "name",
            "item",
            "item_name"
        ]
    )

    if not product_col:
        return None, 0

    best_match = None
    best_score = 0

    for _, row in df.iterrows():

        reference_name = str(row.get(product_col, ""))

        score = similarity_score(
            product_name,
            reference_name
        )

        if score > best_score:
            best_score = score
            best_match = row

    return best_match, best_score


def get_reference_price(row, df):
    if row is None or df.empty:
        return None

    price_col = find_column(
        df,
        [
            "price",
            "reference_price",
            "benchmark_price",
            "unit_price",
            "benchmark",
            "avg_price",
            "average_price"
        ]
    )

    if not price_col:
        return None

    return safe_float(row.get(price_col), None)


# ============================================================
# RECEIPT EXTRACTION
# ============================================================

def extract_receipt(image):

    if not groq_client:
        raise RuntimeError(
            "GROQ_API_KEY is not configured in Streamlit secrets."
        )

    image_b64 = image_to_base64(image)

    prompt = """
Extract the receipt information from this image.

Return ONLY valid JSON.

Use this exact structure:

{
  "store": "",
  "date": "",
  "items": [
    {
      "product": "",
      "brand": "",
      "quantity": 1,
      "unit_price": 0,
      "line_total": 0
    }
  ],
  "receipt_total": 0
}

Rules:
- Use numbers for quantity, unit_price, line_total and receipt_total.
- If a value is unavailable, use an empty string for text and 0 for numeric values.
- Do not invent products.
- Keep product names concise.
"""

    response = groq_client.chat.completions.create(
        model="qwen/qwen3.6-27b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a receipt extraction assistant. "
                    "Return accurate structured JSON only."
                )
            },
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
                                + image_b64
                            )
                        }
                    }
                ]
            }
        ],
        temperature=0,
        max_completion_tokens=800,
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content

    return json.loads(content)


# ============================================================
# PROCESS RECEIPT ITEMS
# ============================================================

def process_items(receipt_data, df):

    processed = []

    items = receipt_data.get("items", [])

    for item in items:

        product_name = str(
            item.get("product", "")
        ).strip()

        quantity = safe_int(
            item.get("quantity", 1)
        )

        unit_price = safe_float(
            item.get("unit_price", 0)
        )

        line_total = safe_float(
            item.get("line_total", 0)
        )

        if unit_price == 0 and line_total > 0:
            unit_price = line_total / quantity

        reference_row, match_score = match_product(
            product_name,
            df
        )

        reference_price = get_reference_price(
            reference_row,
            df
        )

        difference = None
        difference_percent = None

        if reference_price is not None:
            difference = unit_price - reference_price

            if reference_price != 0:
                difference_percent = (
                    difference / reference_price
                ) * 100

        if difference is not None:

            if difference > 0:
                status = "Above benchmark"

            elif difference < 0:
                status = "Below benchmark"

            else:
                status = "At benchmark"

        else:
            status = "No benchmark"

        processed.append(
            {
                "product": product_name,
                "brand": str(
                    item.get("brand", "")
                ),
                "quantity": quantity,
                "charged_unit_price": unit_price,
                "line_total": line_total,
                "reference_price": reference_price,
                "difference": difference,
                "difference_percent": difference_percent,
                "match_score": match_score,
                "status": status
            }
        )

    return processed


# ============================================================
# AI PRICE EXPLANATION
# ============================================================

def explain_price_difference(item):

    if not groq_client:
        return "AI explanation is unavailable because the Groq API key is not configured."

    reference = item.get("reference_price")
    charged = item.get("charged_unit_price")
    difference = item.get("difference")
    percentage = item.get("difference_percent")

    prompt = f"""
Explain this receipt price difference briefly and clearly.

Product: {item.get("product")}
Charged price: {charged}
Reference price: {reference}
Difference: {difference}
Difference percentage: {percentage}%

Give:
1. What the difference means.
2. One or two possible reasons.
3. A short consumer recommendation.

Do not claim illegal overcharging.
Keep the answer concise.
"""

    try:

        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful consumer price "
                        "verification assistant."
                    )
                },
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
        return f"AI explanation unavailable: {str(e)}"


# ============================================================
# AI RECEIPT CHAT
# ============================================================

def ask_receipt_question(question):

    if not groq_client:
        return (
            "AI chat is unavailable because the Groq API key "
            "is not configured."
        )

    receipt_data = st.session_state.receipt_data
    processed = st.session_state.processed_items

    context = {
        "receipt": receipt_data,
        "analysis": processed
    }

    prompt = f"""
Answer the user's question using only the receipt information below.

Receipt information:
{json.dumps(context, indent=2, default=str)}

User question:
{question}

Instructions:
- Be accurate.
- Be concise.
- Give the important information first.
- Do not invent missing information.
- If the receipt does not contain the answer, clearly say so.
"""

    try:

        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise receipt analysis assistant."
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

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"AI answer unavailable: {str(e)}"


# ============================================================
# PRICE HEALTH SCORE
# ============================================================

def calculate_health_score(items):

    benchmark_items = [
        item
        for item in items
        if item.get("reference_price") is not None
    ]

    if not benchmark_items:
        return None

    above_count = sum(
        1
        for item in benchmark_items
        if safe_float(item.get("difference")) > 0
    )

    total = len(benchmark_items)

    score = 100 - (
        above_count / total * 100
    )

    return round(max(0, min(100, score)))


def get_verdict(score):

    if score is None:
        return "Insufficient benchmark data"

    if score >= 85:
        return "Looks healthy"

    if score >= 65:
        return "Mostly reasonable"

    if score >= 45:
        return "Review recommended"

    return "Multiple price differences"


# ============================================================
# RECEIPT TOTAL CHECK
# ============================================================

def check_receipt_total(receipt_data, items):

    receipt_total = safe_float(
        receipt_data.get("receipt_total", 0)
    )

    calculated_total = sum(
        safe_float(item.get("line_total", 0))
        for item in items
    )

    difference = receipt_total - calculated_total

    if receipt_total == 0:
        return {
            "available": False,
            "receipt_total": receipt_total,
            "calculated_total": calculated_total,
            "difference": difference
        }

    tolerance = max(1.0, receipt_total * 0.02)

    consistent = abs(difference) <= tolerance

    return {
        "available": True,
        "receipt_total": receipt_total,
        "calculated_total": calculated_total,
        "difference": difference,
        "consistent": consistent
    }


# ============================================================
# CONSUMER REPORT
# ============================================================

def create_report():

    receipt = st.session_state.receipt_data
    items = st.session_state.processed_items

    score = calculate_health_score(items)
    verdict = get_verdict(score)

    total_check = check_receipt_total(
        receipt,
        items
    )

    lines = []

    lines.append("PRICEPROOF AI - CONSUMER VERIFICATION REPORT")
    lines.append("=" * 55)
    lines.append("")

    lines.append(
        f"Store: {receipt.get('store', 'Unknown')}"
    )

    lines.append(
        f"Date: {receipt.get('date', 'Unknown')}"
    )

    lines.append("")

    lines.append(
        f"Receipt Total: "
        f"{safe_float(receipt.get('receipt_total', 0)):.2f}"
    )

    if score is not None:
        lines.append(
            f"Price Health Score: {score}/100"
        )

    lines.append(
        f"Verdict: {verdict}"
    )

    lines.append("")

    lines.append("PRODUCT ANALYSIS")
    lines.append("-" * 55)

    for item in items:

        lines.append(
            f"Product: {item.get('product')}"
        )

        lines.append(
            f"Quantity: {item.get('quantity')}"
        )

        lines.append(
            f"Charged Unit Price: "
            f"{safe_float(item.get('charged_unit_price')):.2f}"
        )

        reference = item.get("reference_price")

        if reference is not None:

            lines.append(
                f"Reference Price: "
                f"{safe_float(reference):.2f}"
            )

            lines.append(
                f"Difference: "
                f"{safe_float(item.get('difference')):.2f}"
            )

        else:
            lines.append(
                "Reference Price: Not available"
            )

        lines.append(
            f"Status: {item.get('status')}"
        )

        lines.append("")

    if total_check.get("available"):

        lines.append("RECEIPT TOTAL CHECK")
        lines.append("-" * 55)

        lines.append(
            f"Receipt total: "
            f"{total_check['receipt_total']:.2f}"
        )

        lines.append(
            f"Calculated item total: "
            f"{total_check['calculated_total']:.2f}"
        )

        lines.append(
            f"Difference: "
            f"{total_check['difference']:.2f}"
        )

        lines.append(
            "Status: "
            + (
                "Consistent"
                if total_check["consistent"]
                else "Review recommended"
            )
        )

        lines.append("")

    lines.append(
        "DISCLAIMER: Benchmark prices are reference values and "
        "may not represent official market prices. A price "
        "difference does not by itself prove illegal overcharging."
    )

    return "\n".join(lines)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🧾 PriceProof AI")

    st.caption(
        "Consumer price verification assistant"
    )

    st.divider()

    st.markdown("### 📌 Analysis")

    st.write("1. Upload receipt")
    st.write("2. Extract receipt data")
    st.write("3. Match products")
    st.write("4. Compare prices")
    st.write("5. Review results")

    st.divider()

    st.markdown("### 🗃️ Reference Database")

    if products_df.empty:
        st.error("products.csv not available")
    else:
        st.success(
            f"{len(products_df)} reference records loaded"
        )

    st.divider()

    if st.session_state.analysis_complete:

        st.markdown("### ✅ Current Status")

        st.success(
            "Receipt analysis complete"
        )

        if st.button(
            "🔄 Start New Analysis",
            use_container_width=True
        ):

            for key, value in defaults.items():
                st.session_state[key] = value

            st.rerun()


# ============================================================
# RECEIPT UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">📤 Upload Receipt</div>',
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
    help="Upload a clear receipt image for best extraction accuracy."
)


if uploaded_file:

    try:

        image = Image.open(
            uploaded_file
        )

        image = prepare_image(image)

        st.session_state.receipt_image = image

        left, right = st.columns(
            [1, 1]
        )

        with left:

            st.image(
                image,
                caption="Uploaded receipt",
                use_container_width=True
            )

        with right:

            st.markdown(
                """
                <div class="info-box">
                    <strong>Receipt ready for analysis</strong><br>
                    The image has been prepared for AI extraction.
                    Click the button below to begin.
                </div>
                """,
                unsafe_allow_html=True
            )

            analyze_clicked = st.button(
                "🔍 Analyze Receipt",
                type="primary",
                use_container_width=True,
                key="analyze_receipt_button"
            )

            if analyze_clicked:

                if not groq_client:

                    st.error(
                        "GROQ_API_KEY is missing. "
                        "Please add it to Streamlit Secrets."
                    )

                elif products_df.empty:

                    st.error(
                        "products.csv could not be loaded. "
                        "Please check that the file exists in your GitHub repository."
                    )

                else:

                    with st.spinner(
                        "AI is reading and analyzing the receipt..."
                    ):

                        try:

                            receipt_data = extract_receipt(
                                image
                            )

                            processed_items = process_items(
                                receipt_data,
                                products_df
                            )

                            st.session_state.receipt_data = receipt_data

                            st.session_state.processed_items = processed_items

                            st.session_state.analysis_complete = True

                            st.session_state.chat_history = []

                            st.session_state.manual_results = {}

                            st.session_state.last_report = create_report()

                            st.success(
                                "Receipt analyzed successfully."
                            )

                            st.rerun()

                        except Exception as e:

                            st.session_state.analysis_complete = False

                            st.error(
                                "The receipt could not be analyzed."
                            )

                            st.code(
                                str(e)
                            )

    except Exception as e:

        st.error(
            "The uploaded image could not be opened."
        )

        st.code(
            str(e)
        )


# ============================================================
# DATABASE PREVIEW
# ============================================================

if not st.session_state.analysis_complete:

    if not products_df.empty:

        with st.expander(
            "🗃️ Preview Reference Database"
        ):

            st.caption(
                "Reference data used for product matching and price comparison."
            )

            st.dataframe(
                products_df.head(20),
                use_container_width=True,
                hide_index=True
            )

    st.info(
        "Upload a receipt above to start the analysis."
    )


# ============================================================
# RESULTS
# ============================================================

if st.session_state.analysis_complete:

    receipt = st.session_state.receipt_data

    items = st.session_state.processed_items

    # --------------------------------------------------------
    # RESULTS HEADER
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-title">📊 Verification Dashboard</div>',
        unsafe_allow_html=True
    )

    st.success(
        "Analysis complete — review the verification results below."
    )

    # --------------------------------------------------------
    # RECEIPT INFORMATION
    # --------------------------------------------------------

    store = receipt.get(
        "store",
        "Unknown"
    )

    date = receipt.get(
        "date",
        "Unknown"
    )

    receipt_total = safe_float(
        receipt.get("receipt_total", 0)
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🏪 Store",
            store or "Unknown"
        )

    with col2:
        st.metric(
            "📅 Date",
            date or "Unknown"
        )

    with col3:
        st.metric(
            "🧾 Receipt Total",
            f"{receipt_total:,.2f}"
        )

    # --------------------------------------------------------
    # QUICK SUMMARY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">⚡ Quick Summary</div>',
        unsafe_allow_html=True
    )

    benchmarked = [
        item
        for item in items
        if item.get("reference_price") is not None
    ]

    above = [
        item
        for item in benchmarked
        if safe_float(item.get("difference")) > 0
    ]

    below = [
        item
        for item in benchmarked
        if safe_float(item.get("difference")) < 0
    ]

    potential_extra = sum(
        max(
            0,
            safe_float(item.get("difference"))
        )
        * safe_int(item.get("quantity", 1))
        for item in items
        if item.get("reference_price") is not None
    )

    potential_savings = sum(
        max(
            0,
            -safe_float(item.get("difference"))
        )
        * safe_int(item.get("quantity", 1))
        for item in items
        if item.get("reference_price") is not None
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🛒 Products",
            len(items)
        )

    with col2:
        st.metric(
            "🔎 Benchmarked",
            len(benchmarked)
        )

    with col3:
        st.metric(
            "📈 Above Benchmark",
            len(above)
        )

    with col4:
        st.metric(
            "📉 Below Benchmark",
            len(below)
        )

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    score = calculate_health_score(
        items
    )

    verdict = get_verdict(
        score
    )

    st.markdown(
        '<div class="section-title">🩺 Price Health</div>',
        unsafe_allow_html=True
    )

    score_col, verdict_col = st.columns(
        [1, 2]
    )

    with score_col:

        if score is not None:

            st.metric(
                "Price Health Score",
                f"{score}/100"
            )

            st.progress(
                score / 100
            )

        else:

            st.metric(
                "Price Health Score",
                "N/A"
            )

    with verdict_col:

        if score is None:

            st.info(
                "There is not enough benchmark data "
                "to calculate a reliable score."
            )

        elif score >= 85:

            st.success(
                f"**{verdict}** — Most benchmarked items "
                "are reasonably aligned with reference prices."
            )

        elif score >= 65:

            st.warning(
                f"**{verdict}** — Some items differ from "
                "the reference prices."
            )

        else:

            st.error(
                f"**{verdict}** — Several benchmarked items "
                "show price differences worth reviewing."
            )

    # --------------------------------------------------------
    # EXTRA / SAVINGS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "💰 Potential Extra",
            f"{potential_extra:,.2f}"
        )

        st.caption(
            "Estimated amount above the available benchmark prices."
        )

    with col2:

        st.metric(
            "💵 Potential Savings",
            f"{potential_savings:,.2f}"
        )

        st.caption(
            "Estimated amount below the available benchmark prices."
        )

    # --------------------------------------------------------
    # RECEIPT TOTAL CONSISTENCY
    # --------------------------------------------------------

    total_check = check_receipt_total(
        receipt,
        items
    )

    st.markdown(
        '<div class="section-title">🧮 Receipt Total Check</div>',
        unsafe_allow_html=True
    )

    if total_check.get("available"):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Receipt Total",
                f"{total_check['receipt_total']:,.2f}"
            )

        with col2:
            st.metric(
                "Calculated Total",
                f"{total_check['calculated_total']:,.2f}"
            )

        with col3:
            st.metric(
                "Difference",
                f"{total_check['difference']:,.2f}"
            )

        if total_check["consistent"]:

            st.success(
                "The receipt total is reasonably consistent "
                "with the extracted line totals."
            )

        else:

            st.warning(
                "The receipt total and extracted line totals "
                "do not match closely. Review the receipt image."
            )

    else:

        st.info(
            "A receipt total was not available for this check."
        )

    # --------------------------------------------------------
    # PRODUCT ANALYSIS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🛍️ Product Analysis</div>',
        unsafe_allow_html=True
    )

    if items:

        table_rows = []

        for item in items:

            reference = item.get(
                "reference_price"
            )

            difference = item.get(
                "difference"
            )

            percentage = item.get(
                "difference_percent"
            )

            table_rows.append(
                {
                    "Product": item.get("product"),
                    "Qty": item.get("quantity"),
                    "Charged": round(
                        safe_float(
                            item.get("charged_unit_price")
                        ),
                        2
                    ),
                    "Reference": (
                        round(
                            safe_float(reference),
                            2
                        )
                        if reference is not None
                        else "N/A"
                    ),
                    "Difference": (
                        round(
                            safe_float(difference),
                            2
                        )
                        if difference is not None
                        else "N/A"
                    ),
                    "Difference %": (
                        f"{safe_float(percentage):.1f}%"
                        if percentage is not None
                        else "N/A"
                    ),
                    "Match": (
                        f"{safe_float(item.get('match_score')) * 100:.0f}%"
                        if item.get("match_score") is not None
                        else "N/A"
                    ),
                    "Status": item.get("status")
                }
            )

        table_df = pd.DataFrame(
            table_rows
        )

        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No products were extracted from the receipt."
        )

    # --------------------------------------------------------
    # CHARGED VS REFERENCE CHART
    # --------------------------------------------------------

    chart_data = []

    for item in items:

        if item.get("reference_price") is not None:

            chart_data.append(
                {
                    "Product": item.get("product"),
                    "Charged Price": safe_float(
                        item.get("charged_unit_price")
                    ),
                    "Reference Price": safe_float(
                        item.get("reference_price")
                    )
                }
            )

    if chart_data:

        st.markdown(
            '<div class="section-title">📈 Charged vs Reference Prices</div>',
            unsafe_allow_html=True
        )

        chart_df = pd.DataFrame(
            chart_data
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
            title="Price Comparison"
        )

        fig.update_layout(
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            ),
            xaxis_title="Product",
            yaxis_title="Price"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # ITEMS TO REVIEW
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🚩 Items Worth Reviewing</div>',
        unsafe_allow_html=True
    )

    review_items = sorted(
        [
            item
            for item in items
            if item.get("reference_price") is not None
        ],
        key=lambda x: abs(
            safe_float(x.get("difference_percent"))
        ),
        reverse=True
    )

    if review_items:

        top_items = review_items[:5]

        for index, item in enumerate(top_items):

            difference = safe_float(
                item.get("difference")
            )

            percentage = safe_float(
                item.get("difference_percent")
            )

            if difference > 0:

                icon = "🔴"

                title = (
                    f"{icon} {item.get('product')} — "
                    f"{percentage:.1f}% above benchmark"
                )

            elif difference < 0:

                icon = "🟢"

                title = (
                    f"{icon} {item.get('product')} — "
                    f"{abs(percentage):.1f}% below benchmark"
                )

            else:

                title = (
                    f"🟡 {item.get('product')} — "
                    "At benchmark"
                )

            with st.expander(title):

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Charged",
                        f"{safe_float(item.get('charged_unit_price')):,.2f}"
                    )

                with col2:
                    st.metric(
                        "Reference",
                        f"{safe_float(item.get('reference_price')):,.2f}"
                    )

                with col3:
                    st.metric(
                        "Difference",
                        f"{difference:,.2f}"
                    )

                st.caption(
                    f"Product match confidence: "
                    f"{safe_float(item.get('match_score')) * 100:.0f}%"
                )

                if st.button(
                    "🧠 Explain This Difference",
                    key=f"explain_difference_{index}"
                ):

                    with st.spinner(
                        "Generating explanation..."
                    ):

                        explanation = explain_price_difference(
                            item
                        )

                    st.info(
                        explanation
                    )

    else:

        st.success(
            "No benchmarked items require special review."
        )

    # --------------------------------------------------------
    # MOST SIGNIFICANT DIFFERENCE
    # --------------------------------------------------------

    if review_items:

        most_significant = review_items[0]

        st.markdown(
            '<div class="section-title">🎯 Most Significant Difference</div>',
            unsafe_allow_html=True
        )

        difference = safe_float(
            most_significant.get("difference")
        )

        percentage = safe_float(
            most_significant.get("difference_percent")
        )

        if difference > 0:

            st.warning(
                f"**{most_significant.get('product')}** has the "
                f"largest benchmark difference: "
                f"{difference:,.2f} "
                f"({percentage:.1f}% above benchmark)."
            )

        elif difference < 0:

            st.success(
                f"**{most_significant.get('product')}** has the "
                f"largest negative difference: "
                f"{abs(difference):,.2f} "
                f"({abs(percentage):.1f}% below benchmark)."
            )

        else:

            st.info(
                f"**{most_significant.get('product')}** "
                "is at the benchmark price."
            )

    # --------------------------------------------------------
    # AI CHAT
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">💬 Ask About Your Receipt</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Ask any question about the uploaded receipt and its analysis."
    )

    if st.session_state.chat_history:

        for message in st.session_state.chat_history:

            with st.chat_message(
                message["role"]
            ):

                st.write(
                    message["content"]
                )

    question = st.chat_input(
        "Example: Which product has the highest price difference?"
    )

    if question:

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            with st.spinner(
                "Analyzing your question..."
            ):

                answer = ask_receipt_question(
                    question
                )

            st.write(answer)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }

        )

    # --------------------------------------------------------
    # MANUAL VERIFICATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">✍️ Manual Price Verification</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Enter a known/reference price to manually compare it with the charged price."
    )

    if items:

        product_options = [
            item.get("product")
            for item in items
        ]

        selected_product = st.selectbox(
            "Select product",
            product_options,
            key="manual_product_select"
        )

        selected_item = next(
            (
                item
                for item in items
                if item.get("product") == selected_product
            ),
            None
        )

        if selected_item:

            col1, col2 = st.columns(2)

            with col1:

                charged_price = st.number_input(
                    "Charged unit price",
                    min_value=0.0,
                    value=float(
                        safe_float(
                            selected_item.get(
                                "charged_unit_price"
                            )
                        )
                    ),
                    step=1.0,
                    key="manual_charged_price"
                )

            with col2:

                manual_reference = st.number_input(
                    "Reference price",
                    min_value=0.0,
                    value=float(
                        safe_float(
                            selected_item.get(
                                "reference_price"
                            )
                        )
                    ),
                    step=1.0,
                    key="manual_reference_price"
                )

            if st.button(
                "🔎 Verify Manually",
                key="manual_verify_button"
            ):

                difference = (
                    charged_price
                    - manual_reference
                )

                if manual_reference > 0:

                    percentage = (
                        difference
                        / manual_reference
                    ) * 100

                else:

                    percentage = 0

                st.session_state.manual_results[
                    selected_product
                ] = {
                    "difference": difference,
                    "percentage": percentage
                }

            result = st.session_state.manual_results.get(
                selected_product
            )

            if result:

                difference = result["difference"]

                percentage = result["percentage"]

                if difference > 0:

                    st.warning(
                        f"Charged price is "
                        f"{difference:,.2f} above the manual reference "
                        f"({percentage:.1f}%)."
                    )

                elif difference < 0:

                    st.success(
                        f"Charged price is "
                        f"{abs(difference):,.2f} below the manual reference "
                        f"({abs(percentage):.1f}%)."
                    )

                else:

                    st.info(
                        "Charged price matches the manual reference."
                    )

    # --------------------------------------------------------
    # DOWNLOADS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">📥 Download Results</div>',
        unsafe_allow_html=True
    )

    report = create_report()

    st.session_state.last_report = report

    export_rows = []

    for item in items:

        export_rows.append(
            {
                "Product": item.get("product"),
                "Brand": item.get("brand"),
                "Quantity": item.get("quantity"),
                "Charged Unit Price": item.get(
                    "charged_unit_price"
                ),
                "Line Total": item.get(
                    "line_total"
                ),
                "Reference Price": item.get(
                    "reference_price"
                ),
                "Difference": item.get(
                    "difference"
                ),
                "Difference %": item.get(
                    "difference_percent"
                ),
                "Match Confidence": item.get(
                    "match_score"
                ),
                "Status": item.get(
                    "status"
                )
            }
        )

    export_df = pd.DataFrame(
        export_rows
    )

    json_data = json.dumps(
        {
            "receipt": receipt,
            "analysis": items,
            "price_health_score": score,
            "verdict": verdict,
            "potential_extra": potential_extra,
            "potential_savings": potential_savings
        },
        indent=2,
        default=str
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.download_button(
            "📊 Download CSV",
            data=export_df.to_csv(
                index=False
            ),
            file_name="priceproof_analysis.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:

        st.download_button(
            "📦 Download JSON",
            data=json_data,
            file_name="priceproof_analysis.json",
            mime="application/json",
            use_container_width=True
        )

    with col3:

        st.download_button(
            "📄 Consumer Report",
            data=report,
            file_name="priceproof_consumer_report.txt",
            mime="text/plain",
            use_container_width=True
        )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="warning-box">
            <strong>⚠️ Important Disclaimer</strong><br><br>
            PriceProof AI uses benchmark/reference prices for comparison.
            These values may not represent official market prices.
            A difference between a charged price and a benchmark does not
            by itself prove illegal overcharging. Always consider factors
            such as location, promotions, taxes, timing, and product variation.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        PriceProof AI • AI-powered consumer price verification
    </div>
    """,
    unsafe_allow_html=True
)
```
