import streamlit as st
import pandas as pd
from PIL import Image
from groq import Groq
import base64
import json
import re
import os
import difflib
from datetime import datetime
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
# MODELS
# ============================================================

VISION_MODEL = "qwen/qwen3.6-27b"
TEXT_MODEL = "openai/gpt-oss-20b"

MAX_IMAGE_BYTES = 20 * 1024 * 1024
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]


# ============================================================
# PROFESSIONAL UI STYLE
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #f8fafc 0%,
            #eef2ff 50%,
            #f8fafc 100%
        );
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    /* HERO */

    .hero {
        background: linear-gradient(
            135deg,
            #111827 0%,
            #1e3a8a 55%,
            #312e81 100%
        );
        padding: 2.8rem;
        border-radius: 26px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 15px 40px rgba(15, 23, 42, 0.18);
    }

    .hero h1 {
        font-size: 3rem;
        margin: 0;
        font-weight: 800;
        letter-spacing: -1px;
    }

    .hero p {
        font-size: 1.15rem;
        opacity: 0.92;
        margin-top: 0.5rem;
        margin-bottom: 0.7rem;
    }

    .hero-tag {
        display: inline-block;
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.18);
        padding: 0.5rem 1rem;
        border-radius: 999px;
        margin-top: 0.7rem;
        font-size: 0.9rem;
    }

    /* SECTION */

    .section-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
        margin-bottom: 1.2rem;
    }

    /* METRICS */

    .metric-card {
        background: white;
        padding: 1.25rem;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
        min-height: 125px;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #111827;
    }

    .metric-sub {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 0.3rem;
    }

    /* STATUS */

    .status-good {
        background: #ecfdf5;
        border-left: 5px solid #10b981;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
    }

    .status-warning {
        background: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
    }

    .status-danger {
        background: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
    }

    .status-info {
        background: #eff6ff;
        border-left: 5px solid #3b82f6;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
    }

    /* FEATURE CARDS */

    .feature-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 1.25rem;
        height: 100%;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
    }

    .feature-icon {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }

    .feature-title {
        font-weight: 750;
        color: #111827;
        font-size: 1.05rem;
        margin-bottom: 0.35rem;
    }

    .feature-text {
        color: #64748b;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* STEP BADGE */

    .step-badge {
        display: inline-block;
        background: #eef2ff;
        color: #3730a3;
        border-radius: 999px;
        padding: 0.35rem 0.8rem;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    /* FOOTER */

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        padding: 2.5rem 0 1rem 0;
    }

    /* BUTTON */

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
    }

    /* SIDEBAR */

    [data-testid="stSidebar"] {
        background: #ffffff;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "receipt_data" not in st.session_state:
    st.session_state.receipt_data = None

if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = None

if "receipt_image_bytes" not in st.session_state:
    st.session_state.receipt_image_bytes = None

if "receipt_filename" not in st.session_state:
    st.session_state.receipt_filename = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

if "ai_explanation" not in st.session_state:
    st.session_state.ai_explanation = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_groq_client():
    api_key = None

    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


def safe_float(value, default=0.0):
    if value is None:
        return default

    if isinstance(value, (int, float)):
        return float(value)

    try:
        cleaned = str(value).replace(",", "")
        cleaned = re.sub(r"[^\d.\-]", "", cleaned)

        if cleaned in ["", "-", ".", "-."]:
            return default

        return float(cleaned)

    except Exception:
        return default


def clean_text(value):
    if value is None:
        return ""

    return str(value).strip()


def normalize_product_name(name):
    name = clean_text(name).lower()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name)

    return name.strip()


def encode_image_bytes(image_bytes, mime_type):
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def detect_mime_type(filename):
    extension = filename.lower().split(".")[-1]

    mapping = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp"
    }

    return mapping.get(extension, "image/jpeg")


def extract_json_from_text(text):
    if not text:
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    cleaned = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE
    )

    cleaned = cleaned.replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    object_match = re.search(
        r"\{.*\}",
        cleaned,
        flags=re.DOTALL
    )

    if object_match:
        try:
            return json.loads(object_match.group(0))
        except Exception:
            pass

    return None


def format_currency(value):
    value = safe_float(value)
    return f"PKR {value:,.2f}"


# ============================================================
# LOAD REFERENCE DATABASE
# ============================================================

@st.cache_data
def load_products():

    if not os.path.exists("products.csv"):
        return pd.DataFrame()

    try:
        df = pd.read_csv("products.csv")
        df.columns = [
            str(column).strip()
            for column in df.columns
        ]

        return df

    except Exception:
        return pd.DataFrame()


products_df = load_products()


def find_column(df, candidates):

    if df.empty:
        return None

    normalized = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    for column in df.columns:

        column_lower = str(column).lower()

        for candidate in candidates:

            if candidate.lower() in column_lower:
                return column

    return None


def prepare_reference_database(df):

    if df.empty:

        return pd.DataFrame(
            columns=[
                "product",
                "reference_price",
                "brand",
                "category",
                "size",
                "date"
            ]
        )

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

    price_col = find_column(
        df,
        [
            "reference_price",
            "reference price",
            "market_price",
            "market price",
            "average_price",
            "average price",
            "price",
            "unit_price"
        ]
    )

    if not product_col or not price_col:

        return pd.DataFrame(
            columns=[
                "product",
                "reference_price",
                "brand",
                "category",
                "size",
                "date"
            ]
        )

    result = pd.DataFrame()

    result["product"] = df[product_col].astype(str)

    result["reference_price"] = (
        df[price_col].apply(safe_float)
    )

    brand_col = find_column(df, ["brand"])

    category_col = find_column(
        df,
        ["category"]
    )

    size_col = find_column(
        df,
        ["size", "package_size", "pack_size"]
    )

    date_col = find_column(
        df,
        ["date", "price_date", "updated"]
    )

    result["brand"] = (
        df[brand_col].astype(str)
        if brand_col
        else ""
    )

    result["category"] = (
        df[category_col].astype(str)
        if category_col
        else ""
    )

    result["size"] = (
        df[size_col].astype(str)
        if size_col
        else ""
    )

    result["date"] = (
        df[date_col].astype(str)
        if date_col
        else ""
    )

    result = result[
        result["reference_price"] > 0
    ].copy()

    return result


reference_df = prepare_reference_database(
    products_df
)


# ============================================================
# PRODUCT MATCHING
# ============================================================

def match_product(product_name, reference_data):

    if reference_data.empty:

        return {
            "matched": False,
            "product": product_name,
            "reference_price": None,
            "similarity": 0,
            "brand": "",
            "category": "",
            "size": "",
            "date": ""
        }

    target = normalize_product_name(
        product_name
    )

    if not target:

        return {
            "matched": False,
            "product": product_name,
            "reference_price": None,
            "similarity": 0,
            "brand": "",
            "category": "",
            "size": "",
            "date": ""
        }

    best_index = None
    best_score = 0.0

    for index, row in reference_data.iterrows():

        reference_name = normalize_product_name(
            row["product"]
        )

        if not reference_name:
            continue

        score = difflib.SequenceMatcher(
            None,
            target,
            reference_name
        ).ratio()

        if target == reference_name:
            score = 1.0

        elif (
            target in reference_name
            or reference_name in target
        ):
            score = max(score, 0.86)

        if score > best_score:
            best_score = score
            best_index = index

    if (
        best_index is None
        or best_score < 0.55
    ):

        return {
            "matched": False,
            "product": product_name,
            "reference_price": None,
            "similarity": best_score,
            "brand": "",
            "category": "",
            "size": "",
            "date": ""
        }

    row = reference_data.loc[best_index]

    return {
        "matched": True,
        "product": product_name,
        "reference_product": row["product"],
        "reference_price": safe_float(
            row["reference_price"]
        ),
        "similarity": best_score,
        "brand": clean_text(row["brand"]),
        "category": clean_text(row["category"]),
        "size": clean_text(row["size"]),
        "date": clean_text(row["date"])
    }


# ============================================================
# AI RECEIPT EXTRACTION
# ============================================================

def extract_receipt_with_ai(
    client,
    image_bytes,
    mime_type
):

    image_url = encode_image_bytes(
        image_bytes,
        mime_type
    )

    prompt = """
Read this shopping receipt and return ONLY valid JSON.

Use exactly this structure:

{
 "store_name":"",
 "receipt_date":"",
 "currency":"PKR",
 "total":0,
 "items":[
   {
     "name":"",
     "quantity":1,
     "unit_price":0,
     "line_total":0
   }
 ]
}

Rules:

- Extract only information visible on the receipt.
- Do not invent products or prices.
- Preserve product names as closely as possible.
- If quantity is unclear, use 1.
- If only a line total is visible, use it as line_total.
- If unit price is missing, calculate it from line_total / quantity when possible.
- If the total is unclear, use 0.
- Return JSON only.
"""

    response = client.chat.completions.create(
        model=VISION_MODEL,
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
                            "url": image_url
                        }
                    }
                ]
            }
        ],
        temperature=0,
        max_completion_tokens=700,
        reasoning_effort="none",
        response_format={
            "type": "json_object"
        }
    )

    content = (
        response.choices[0]
        .message
        .content
    )

    data = extract_json_from_text(
        content
    )

    if not isinstance(data, dict):

        raise ValueError(
            "AI did not return valid receipt JSON."
        )

    return data


# ============================================================
# NORMALIZE RECEIPT
# ============================================================

def normalize_receipt_data(data):

    if not isinstance(data, dict):
        data = {}

    items = data.get("items", [])

    if not isinstance(items, list):
        items = []

    normalized_items = []

    for item in items:

        if not isinstance(item, dict):
            continue

        name = clean_text(
            item.get("name", "")
        )

        if not name:
            continue

        quantity = safe_float(
            item.get("quantity", 1),
            1
        )

        if quantity <= 0:
            quantity = 1

        unit_price = safe_float(
            item.get("unit_price", 0)
        )

        line_total = safe_float(
            item.get("line_total", 0)
        )

        if (
            line_total <= 0
            and unit_price > 0
        ):

            line_total = (
                unit_price * quantity
            )

        if (
            unit_price <= 0
            and line_total > 0
        ):

            unit_price = (
                line_total / quantity
            )

        normalized_items.append(
            {
                "name": name,
                "quantity": quantity,
                "unit_price": unit_price,
                "line_total": line_total
            }
        )

    return {
        "store_name": clean_text(
            data.get("store_name", "")
        ),
        "receipt_date": clean_text(
            data.get("receipt_date", "")
        ),
        "currency": clean_text(
            data.get("currency", "PKR")
        ) or "PKR",
        "total": safe_float(
            data.get("total", 0)
        ),
        "items": normalized_items
    }


# ============================================================
# ANALYZE RECEIPT ITEMS
# ============================================================

def analyze_receipt_items(
    receipt_data,
    reference_data
):

    rows = []

    for item in receipt_data.get(
        "items",
        []
    ):

        product_name = item.get(
            "name",
            ""
        )

        quantity = safe_float(
            item.get("quantity", 1),
            1
        )

        unit_price = safe_float(
            item.get("unit_price", 0)
        )

        line_total = safe_float(
            item.get("line_total", 0)
        )

        match = match_product(
            product_name,
            reference_data
        )

        reference_price = match.get(
            "reference_price"
        )

        if reference_price is not None:

            reference_total = (
                reference_price * quantity
            )

            difference = (
                line_total
                - reference_total
            )

            percentage = (
                difference
                / reference_total
                * 100
                if reference_total > 0
                else 0
            )

            potentially_overpriced = (
                percentage >= 10
                and difference > 0
            )

            status = (
                "Potentially High"
                if potentially_overpriced
                else "Within Reference Range"
            )

        else:

            reference_total = None
            difference = None
            percentage = None
            potentially_overpriced = False
            status = "No Reference Match"

        rows.append(
            {
                "Product": product_name,
                "Quantity": quantity,
                "Charged Unit Price": unit_price,
                "Charged Total": line_total,
                "Reference Unit Price": reference_price,
                "Reference Total": reference_total,
                "Difference": difference,
                "Difference %": percentage,
                "Status": status,
                "Potentially Overpriced": potentially_overpriced,
                "Match Score": match.get(
                    "similarity",
                    0
                ),
                "Reference Product": match.get(
                    "reference_product",
                    ""
                ),
                "Brand": match.get(
                    "brand",
                    ""
                ),
                "Category": match.get(
                    "category",
                    ""
                ),
                "Package Size": match.get(
                    "size",
                    ""
                ),
                "Reference Date": match.get(
                    "date",
                    ""
                )
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# PRICE HEALTH SCORE
# ============================================================

def calculate_price_health_score(
    analysis_df
):

    if (
        analysis_df is None
        or analysis_df.empty
    ):
        return None

    matched = analysis_df[
        analysis_df[
            "Reference Unit Price"
        ].notna()
    ]

    if matched.empty:
        return None

    scores = []

    for _, row in matched.iterrows():

        diff = safe_float(
            row["Difference %"]
        )

        if diff <= 0:
            score = 100

        elif diff <= 5:
            score = 95

        elif diff <= 10:
            score = 85

        elif diff <= 20:
            score = 70

        elif diff <= 35:
            score = 50

        else:
            score = 25

        scores.append(score)

    return round(
        sum(scores) / len(scores)
    )


def get_health_label(score):

    if score is None:
        return "Not Available"

    if score >= 90:
        return "Healthy"

    if score >= 75:
        return "Mostly Healthy"

    if score >= 60:
        return "Needs Review"

    return "Review Carefully"


# ============================================================
# AI EXPLANATION
# ============================================================

def generate_ai_explanation(
    client,
    analysis_df
):

    if analysis_df.empty:

        return (
            "No receipt items were available "
            "for analysis."
        )

    records = []

    for _, row in analysis_df.iterrows():

        records.append(
            {
                "product": row["Product"],
                "charged": safe_float(
                    row["Charged Total"]
                ),
                "reference": (
                    safe_float(
                        row["Reference Total"]
                    )
                    if pd.notna(
                        row["Reference Total"]
                    )
                    else None
                ),
                "difference_percent": (
                    safe_float(
                        row["Difference %"]
                    )
                    if pd.notna(
                        row["Difference %"]
                    )
                    else None
                ),
                "status": row["Status"]
            }
        )

    prompt = f"""
Analyze these PriceProof AI receipt comparison results:

{json.dumps(records, indent=2)}

Give a concise consumer-friendly summary.

Rules:
- Mention potentially high-priced items.
- Mention unmatched items.
- Explain important differences.
- Do not call a price illegal.
- Reference prices are benchmarks, not official legal prices.
- Suggest checking brand, package size, quantity, promotions and receipt date.
- Keep the response under 180 words.
"""

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_completion_tokens=300,
        reasoning_effort="low"
    )

    return (
        response.choices[0]
        .message
        .content
        .strip()
    )


# ============================================================
# AI USER QUESTION
# ============================================================

def answer_user_question(
    client,
    question,
    receipt_data,
    analysis_df
):

    if analysis_df is None:

        return (
            "Please analyze a receipt first."
        )

    records = []

    for _, row in analysis_df.iterrows():

        records.append(
            {
                "product": row["Product"],
                "quantity": safe_float(
                    row["Quantity"]
                ),
                "charged": safe_float(
                    row["Charged Total"]
                ),
                "reference": (
                    safe_float(
                        row["Reference Total"]
                    )
                    if pd.notna(
                        row["Reference Total"]
                    )
                    else None
                ),
                "difference": (
                    safe_float(
                        row["Difference"]
                    )
                    if pd.notna(
                        row["Difference"]
                    )
                    else None
                ),
                "difference_percent": (
                    safe_float(
                        row["Difference %"]
                    )
                    if pd.notna(
                        row["Difference %"]
                    )
                    else None
                ),
                "status": row["Status"]
            }
        )

    prompt = f"""
You are PriceProof AI.

Receipt:

{json.dumps(receipt_data, indent=2)}

Analysis:

{json.dumps(records, indent=2)}

User question:

{question}

Answer directly and concisely.

Rules:
- Use only the provided information.
- Do not invent missing information.
- Do not make legal claims.
- Reference prices are benchmarks.
- Keep the answer under 150 words.
"""

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_completion_tokens=250,
        reasoning_effort="low"
    )

    return (
        response.choices[0]
        .message
        .content
        .strip()
    )


# ============================================================
# TOTAL CONSISTENCY
# ============================================================

def calculate_total_consistency(
    receipt_data,
    analysis_df
):

    receipt_total = safe_float(
        receipt_data.get("total", 0)
    )

    if receipt_total <= 0:
        return None

    if (
        analysis_df is None
        or analysis_df.empty
    ):
        return None

    item_total = (
        analysis_df["Charged Total"]
        .apply(safe_float)
        .sum()
    )

    return {
        "receipt_total": receipt_total,
        "item_total": item_total,
        "difference": (
            receipt_total - item_total
        )
    }


# ============================================================
# CONSUMER REPORT
# ============================================================

def generate_consumer_report(
    analysis_df,
    receipt_data
):

    lines = []

    lines.append(
        "PRICEPROOF AI - CONSUMER PRICE VERIFICATION REPORT"
    )

    lines.append("=" * 55)

    lines.append(
        "Generated: "
        + datetime.now().strftime(
            "%Y-%m-%d %H:%M"
        )
    )

    lines.append("")

    if receipt_data.get("store_name"):

        lines.append(
            f"Store: {receipt_data['store_name']}"
        )

    if receipt_data.get("receipt_date"):

        lines.append(
            f"Receipt Date: {receipt_data['receipt_date']}"
        )

    lines.append(
        "Receipt Total: "
        + format_currency(
            receipt_data.get(
                "total",
                0
            )
        )
    )

    lines.append("")

    lines.append("ITEM ANALYSIS")
    lines.append("-" * 55)

    for _, row in analysis_df.iterrows():

        lines.append(
            f"Product: {row['Product']}"
        )

        lines.append(
            "Charged: "
            + format_currency(
                row["Charged Total"]
            )
        )

        if pd.notna(
            row["Reference Total"]
        ):

            lines.append(
                "Reference: "
                + format_currency(
                    row["Reference Total"]
                )
            )

            lines.append(
                "Difference: "
                + format_currency(
                    row["Difference"]
                )
                + f" ({safe_float(row['Difference %']):.1f}%)"
            )

        else:

            lines.append(
                "Reference: "
                "No matching benchmark"
            )

        lines.append(
            f"Status: {row['Status']}"
        )

        lines.append("")

    lines.append("-" * 55)

    lines.append(
        "Reference prices are benchmarks, "
        "not official government or legal prices."
    )

    lines.append(
        "A potentially high-price result "
        "does not by itself prove illegal overcharging."
    )

    return "\n".join(lines)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <h1>🧾 PriceProof AI</h1>

        <p>
            Detect overpricing. Verify the price. Know your rights.
        </p>

        <div class="hero-tag">
            🤖 AI-powered receipt & price verification
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🧾 PriceProof AI")

    st.caption(
        "Smart receipt and price verification"
    )

    st.divider()

    st.markdown("### ✨ What it does")

    st.markdown(
        """
        **1. 📷 Read Receipt**  
        AI extracts products and prices.

        **2. 🔎 Verify Prices**  
        Compare against reference market data.

        **3. ⚠️ Detect Potential Issues**  
        Identify prices that may deserve review.

        **4. 🤖 Explain Results**  
        Get a simple AI-generated explanation.

        **5. 📊 View Report**  
        Download your analysis.
        """
    )

    st.divider()

    st.markdown("### 🔒 Privacy")

    st.caption(
        "Your receipt is processed for analysis "
        "and is not treated as an official legal record."
    )

    st.divider()

    st.markdown("### ℹ️ Important")

    st.caption(
        "Reference prices are benchmarks. "
        "They are not official government prices "
        "and a flag does not prove illegal overcharging."
    )


# ============================================================
# ABOUT / HOW IT WORKS
# ============================================================

with st.expander(
    "💡 About PriceProof AI",
    expanded=False
):

    st.markdown(
        "### What is PriceProof AI?"
    )

    st.write(
        """
        PriceProof AI helps consumers understand whether
        the prices on a shopping receipt are noticeably
        higher than available reference market prices.
        """
    )

    st.markdown(
        "### How it works"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📷</div>
                <div class="feature-title">
                    Read Receipt
                </div>
                <div class="feature-text">
                    Upload a clear receipt and AI
                    extracts the visible products,
                    quantities and prices.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🔎</div>
                <div class="feature-title">
                    Compare
                </div>
                <div class="feature-text">
                    Extracted prices are compared
                    against reference market data.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">⚠️</div>
                <div class="feature-title">
                    Detect
                </div>
                <div class="feature-text">
                    Potentially high prices are
                    highlighted for review.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🤖</div>
                <div class="feature-title">
                    Explain
                </div>
                <div class="feature-text">
                    AI explains the results in
                    simple consumer-friendly language.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# START YOUR PRICE CHECK
# ============================================================

st.header("🧾 Start Your Price Check")

st.write(
    "Upload a clear shopping receipt to begin."
)

uploaded_file = st.file_uploader(
    "Upload receipt image",
    type=SUPPORTED_IMAGE_TYPES,
    help=(
        "Supported formats: JPG, JPEG, PNG and WEBP. "
        "Use a clear, readable receipt."
    )
)


# ============================================================
# UPLOAD PREVIEW
# ============================================================

if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    file_size = len(image_bytes)

    if file_size > MAX_IMAGE_BYTES:

        st.error(
            "The image is too large. "
            "Please upload an image smaller than 20 MB."
        )

    else:

        try:

            image = Image.open(
                io.BytesIO(image_bytes)
            )

            image.verify()

            image = Image.open(
                io.BytesIO(image_bytes)
            )

            st.session_state.receipt_image_bytes = (
                image_bytes
            )

            st.session_state.receipt_filename = (
                uploaded_file.name
            )

            preview_col1, preview_col2 = st.columns(
                [1, 2]
            )

            with preview_col1:

                st.image(
                    image,
                    caption="Receipt Preview",
                    use_container_width=True
                )

            with preview_col2:

                st.markdown(
                    "### 📄 Receipt Ready"
                )

                st.write(
                    f"**File:** {uploaded_file.name}"
                )

                st.write(
                    f"**Size:** "
                    f"{file_size / 1024:.1f} KB"
                )

                st.info(
                    "Make sure the receipt text is "
                    "clear and readable before analysis."
                )

        except Exception:

            st.error(
                "This image could not be read. "
                "Please upload a valid JPG, PNG or WEBP image."
            )


# ============================================================
# ANALYZE BUTTON
# ============================================================

if uploaded_file is not None:

    st.markdown("")

    analyze_button = st.button(
        "🔍 Analyze Receipt with AI",
        type="primary",
        use_container_width=True
    )

    if analyze_button:

        client = get_groq_client()

        if client is None:

            st.error(
                "GROQ_API_KEY is not configured. "
                "Please add it to Streamlit Secrets."
            )

        elif st.session_state.receipt_image_bytes is None:

            st.error(
                "Receipt image is not available."
            )

        else:

            try:

                with st.status(
                    "🤖 AI is analyzing your receipt...",
                    expanded=True
                ) as status:

                    st.write(
                        "📷 Reading receipt..."
                    )

                    mime_type = detect_mime_type(
                        st.session_state.receipt_filename
                    )

                    receipt_data = (
                        extract_receipt_with_ai(
                            client,
                            st.session_state.receipt_image_bytes,
                            mime_type
                        )
                    )

                    st.write(
                        "🧾 Extracting products and prices..."
                    )

                    receipt_data = (
                        normalize_receipt_data(
                            receipt_data
                        )
                    )

                    if not receipt_data.get(
                        "items"
                    ):

                        raise ValueError(
                            "No products could be "
                            "detected on this receipt."
                        )

                    st.write(
                        "🔎 Comparing prices..."
                    )

                    analysis_df = (
                        analyze_receipt_items(
                            receipt_data,
                            reference_df
                        )
                    )

                    st.session_state.receipt_data = (
                        receipt_data
                    )

                    st.session_state.analysis_df = (
                        analysis_df
                    )

                    st.session_state.analysis_complete = (
                        True
                    )

                    st.session_state.chat_history = []

                    st.session_state.ai_explanation = None

                    status.update(
                        label="✅ Receipt analysis complete!",
                        state="complete"
                    )

                st.rerun()

            except Exception as error:

                error_text = str(error)

                if (
                    "429" in error_text
                    or "rate_limit" in error_text
                ):

                    st.error(
                        "⚠️ AI request limit reached temporarily. "
                        "Please wait a few seconds and try again."
                    )

                elif (
                    "invalid_api_key"
                    in error_text.lower()
                ):

                    st.error(
                        "❌ Groq API key is invalid. "
                        "Please check your Streamlit Secrets."
                    )

                else:

                    st.error(
                        "❌ Receipt analysis failed."
                    )

                    st.caption(
                        "Technical details:"
                    )

                    st.code(
                        error_text
                    )


# ============================================================
# RESULTS
# ============================================================

if (
    st.session_state.analysis_complete
    and st.session_state.analysis_df is not None
):

    receipt_data = (
        st.session_state.receipt_data
    )

    analysis_df = (
        st.session_state.analysis_df
    )

    st.divider()

    st.header("📊 Price Verification Results")

    # --------------------------------------------------------
    # BASIC RECEIPT INFO
    # --------------------------------------------------------

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">
                    🏪 Store
                </div>
                <div class="metric-value">
                    {}
                </div>
            </div>
            """.format(
                receipt_data.get(
                    "store_name"
                )
                or "Not detected"
            ),
            unsafe_allow_html=True
        )

    with info_col2:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">
                    🛒 Items
                </div>
                <div class="metric-value">
                    {}
                </div>
            </div>
            """.format(
                len(
                    receipt_data.get(
                        "items",
                        []
                    )
                )
            ),
            unsafe_allow_html=True
        )

    with info_col3:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">
                    💰 Receipt Total
                </div>
                <div class="metric-value">
                    {}
                </div>
            </div>
            """.format(
                format_currency(
                    receipt_data.get(
                        "total",
                        0
                    )
                )
            ),
            unsafe_allow_html=True
        )

    st.markdown("")


    # --------------------------------------------------------
    # DASHBOARD METRICS
    # --------------------------------------------------------

    total_items = len(analysis_df)

    matched_items = int(
        analysis_df[
            "Reference Unit Price"
        ].notna().sum()
    )

    flagged_items = int(
        analysis_df[
            "Potentially Overpriced"
        ].sum()
    )

    unmatched_items = (
        total_items - matched_items
    )

    health_score = (
        calculate_price_health_score(
            analysis_df
        )
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:

        st.metric(
            "🧾 Total Items",
            total_items
        )

    with metric2:

        st.metric(
            "🔎 Matched",
            matched_items
        )

    with metric3:

        st.metric(
            "⚠️ Potentially High",
            flagged_items
        )

    with metric4:

        if health_score is not None:

            st.metric(
                "❤️ Price Health",
                f"{health_score}/100"
            )

        else:

            st.metric(
                "❤️ Price Health",
                "N/A"
            )


    # --------------------------------------------------------
    # OVERALL STATUS
    # --------------------------------------------------------

    st.markdown("### 🧠 Overall Assessment")

    if flagged_items > 0:

        st.markdown(
            f"""
            <div class="status-warning">

                <strong>
                    ⚠️ {flagged_items}
                    item(s) may deserve review
                </strong>

                <br><br>

                PriceProof AI found one or more
                receipt prices that are noticeably
                above the available reference
                benchmark.

                <br><br>

                Check the product brand, package size,
                quantity, promotion and receipt date
                before drawing a conclusion.

            </div>
            """,
            unsafe_allow_html=True
        )

    elif matched_items > 0:

        st.markdown(
            """
            <div class="status-good">

                <strong>
                    ✅ No potentially high-priced
                    matched items detected
                </strong>

                <br><br>

                The matched receipt prices are
                currently within the project's
                reference comparison range.

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="status-info">

                <strong>
                    ℹ️ More reference data is needed
                </strong>

                <br><br>

                The receipt was successfully read,
                but no matching benchmark prices
                were found.

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # PRICE HEALTH
    # --------------------------------------------------------

    if health_score is not None:

        st.markdown("### ❤️ Price Health Score")

        score_col1, score_col2 = st.columns(
            [1, 2]
        )

        with score_col1:

            st.metric(
                "Score",
                f"{health_score}/100"
            )

            st.caption(
                get_health_label(
                    health_score
                )
            )

        with score_col2:

            st.progress(
                health_score / 100
            )

            st.caption(
                "This score is a comparison indicator "
                "based on available reference prices. "
                "It is not a legal determination."
            )


    # --------------------------------------------------------
    # TOTAL CONSISTENCY
    # --------------------------------------------------------

    consistency = (
        calculate_total_consistency(
            receipt_data,
            analysis_df
        )
    )

    if consistency:

        st.markdown(
            "### 🧮 Receipt Total Check"
        )

        difference = consistency[
            "difference"
        ]

        if abs(difference) <= 0.05:

            st.success(
                "✅ Receipt total is consistent "
                "with the extracted line items."
            )

        else:

            st.warning(
                "⚠️ The receipt total differs from "
                "the sum of extracted line items by "
                f"{format_currency(abs(difference))}."
            )


    # --------------------------------------------------------
    # FLAGGED ITEMS
    # --------------------------------------------------------

    st.divider()

    st.header("⚠️ Items Worth Reviewing")

    flagged_df = analysis_df[
        analysis_df[
            "Potentially Overpriced"
        ]
    ].copy()

    if not flagged_df.empty:

        st.warning(
            f"{len(flagged_df)} item(s) "
            "are above the project's review threshold."
        )

        for _, row in flagged_df.iterrows():

            product_name = row["Product"]

            difference = safe_float(
                row["Difference"]
            )

            percentage = safe_float(
                row["Difference %"]
            )

            with st.expander(
                f"⚠️ {product_name} — "
                f"{percentage:.1f}% above reference"
            ):

                detail1, detail2, detail3 = st.columns(3)

                with detail1:

                    st.metric(
                        "Charged",
                        format_currency(
                            row["Charged Unit Price"]
                        )
                    )

                with detail2:

                    st.metric(
                        "Reference",
                        format_currency(
                            row[
                                "Reference Unit Price"
                            ]
                        )
                    )

                with detail3:

                    st.metric(
                        "Difference",
                        format_currency(
                            difference
                        )
                    )

                st.info(
                    "Why review this item? "
                    "The charged price is noticeably "
                    "higher than the available reference "
                    "benchmark."
                )

                st.caption(
                    "Before taking action, check "
                    "brand, package size, quantity, "
                    "promotion/discount and receipt date."
                )

    else:

        st.success(
            "✅ No items currently meet the "
            "potentially-high review threshold."
        )


    # --------------------------------------------------------
    # PRODUCT ANALYSIS TABLE
    # --------------------------------------------------------

    st.divider()

    st.header("🛍️ Product-by-Product Analysis")

    display_df = analysis_df[
        [
            "Product",
            "Quantity",
            "Charged Unit Price",
            "Reference Unit Price",
            "Difference %",
            "Status"
        ]
    ].copy()

    display_df.columns = [
        "Product",
        "Qty",
        "Charged",
        "Reference",
        "Difference %",
        "Status"
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    matched_chart_df = analysis_df[
        analysis_df[
            "Reference Unit Price"
        ].notna()
    ].copy()

    if not matched_chart_df.empty:

        st.divider()

        st.header(
            "📈 Charged Price vs Reference"
        )

        chart_df = matched_chart_df[
            [
                "Product",
                "Charged Unit Price",
                "Reference Unit Price"
            ]
        ].copy()

        chart_df = chart_df.rename(
            columns={
                "Charged Unit Price":
                    "Charged Price",
                "Reference Unit Price":
                    "Reference Price"
            }
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
            title="Receipt Price Comparison",
            labels={
                "Price": "Price (PKR)",
                "Product": "Product"
            }
        )

        fig.update_layout(
            height=450,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=80
            ),
            legend_title_text=""
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # MOST SIGNIFICANT DIFFERENCE
    # --------------------------------------------------------

    significant_df = analysis_df[
        analysis_df[
            "Difference"
        ].notna()
    ].copy()

    if not significant_df.empty:

        significant_df[
            "Abs Difference"
        ] = significant_df[
            "Difference"
        ].abs()

        biggest = significant_df.loc[
            significant_df[
                "Abs Difference"
            ].idxmax()
        ]

        st.divider()

        st.markdown(
            "### 🔎 Most Significant Difference"
        )

        st.info(
            f"**{biggest['Product']}** has the "
            f"largest absolute price difference: "
            f"**{format_currency(biggest['Difference'])}** "
            f"({safe_float(biggest['Difference %']):.1f}%)."
        )


    # --------------------------------------------------------
    # AI EXPLANATION
    # --------------------------------------------------------

    st.divider()

    st.header("🤖 AI Explanation")

    if st.session_state.ai_explanation is None:

        explanation_client = (
            get_groq_client()
        )

        if explanation_client:

            try:

                with st.spinner(
                    "Generating a concise explanation..."
                ):

                    st.session_state.ai_explanation = (
                        generate_ai_explanation(
                            explanation_client,
                            analysis_df
                        )
                    )

            except Exception:

                st.session_state.ai_explanation = (
                    "The AI explanation could not "
                    "be generated right now."
                )

    if st.session_state.ai_explanation:

        st.markdown(
            """
            <div class="section-card">
            """,
            unsafe_allow_html=True
        )

        st.write(
            st.session_state.ai_explanation
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # AI CHAT
    # --------------------------------------------------------

    st.divider()

    st.header(
        "💬 Ask PriceProof AI"
    )

    st.write(
        "Ask your own question about this receipt."
    )

    example_col1, example_col2, example_col3 = st.columns(3)

    with example_col1:

        st.caption(
            "💡 Example: Which item is most expensive?"
        )

    with example_col2:

        st.caption(
            "💡 Example: Which products need review?"
        )

    with example_col3:

        st.caption(
            "💡 Example: How much higher is my total?"
        )

    question = st.chat_input(
        "Ask a question about your receipt..."
    )

    if question:

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )

        chat_client = get_groq_client()

        if chat_client:

            try:

                answer = answer_user_question(
                    chat_client,
                    question,
                    receipt_data,
                    analysis_df
                )

            except Exception:

                answer = (
                    "Sorry, I could not answer "
                    "that question right now."
                )

        else:

            answer = (
                "GROQ_API_KEY is not configured."
            )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    for message in st.session_state.chat_history:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )


    # --------------------------------------------------------
    # MANUAL VERIFICATION
    # --------------------------------------------------------

    st.divider()

    st.header(
        "✏️ Manual Verification"
    )

    st.caption(
        "Use this section when you want to "
        "double-check a specific product yourself."
    )

    manual_col1, manual_col2, manual_col3 = st.columns(3)

    with manual_col1:

        manual_product = st.selectbox(
            "Select product",
            analysis_df[
                "Product"
            ].tolist()
        )

    selected_row = analysis_df[
        analysis_df["Product"]
        == manual_product
    ].iloc[0]

    with manual_col2:

        manual_price = st.number_input(
            "Your reference price",
            min_value=0.0,
            value=float(
                safe_float(
                    selected_row[
                        "Reference Unit Price"
                    ]
                )
            ),
            step=1.0
        )

    with manual_col3:

        charged_price = safe_float(
            selected_row[
                "Charged Unit Price"
            ]
        )

        if manual_price > 0:

            manual_difference = (
                charged_price
                - manual_price
            )

            manual_percent = (
                manual_difference
                / manual_price
                * 100
            )

            st.metric(
                "Difference",
                f"{manual_percent:.1f}%"
            )

    if manual_price > 0:

        if manual_percent > 10:

            st.warning(
                "⚠️ This manually entered "
                "reference suggests the charged "
                "price is noticeably higher."
            )

        else:

            st.success(
                "✅ This manually entered "
                "reference does not show a "
                "large difference."
            )


    # --------------------------------------------------------
    # DOWNLOADS
    # --------------------------------------------------------

    st.divider()

    st.header(
        "📥 Download Your Results"
    )

    download_col1, download_col2, download_col3 = st.columns(3)

    # CSV

    with download_col1:

        csv_data = analysis_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📊 Download CSV",
            data=csv_data,
            file_name="priceproof_analysis.csv",
            mime="text/csv",
            use_container_width=True
        )

    # JSON

    with download_col2:

        json_data = json.dumps(
            {
                "receipt": receipt_data,
                "analysis": analysis_df.fillna(
                    ""
                ).to_dict(
                    orient="records"
                )
            },
            indent=2
        )

        st.download_button(
            label="🧾 Download JSON",
            data=json_data,
            file_name="priceproof_analysis.json",
            mime="application/json",
            use_container_width=True
        )

    # REPORT

    with download_col3:

        report_data = generate_consumer_report(
            analysis_df,
            receipt_data
        )

        st.download_button(
            label="📄 Download Report",
            data=report_data,
            file_name="priceproof_report.txt",
            mime="text/plain",
            use_container_width=True
        )


    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        """
        <div class="status-info">

        <strong>ℹ️ Important Disclaimer</strong>

        <br><br>

        PriceProof AI compares receipt prices against
        available reference market data.

        <br><br>

        Reference prices are <strong>benchmarks</strong>,
        not official government or legal prices.

        <br><br>

        A potentially high-price result does
        <strong>not</strong> by itself prove illegal
        overcharging.

        <br><br>

        Always consider brand, package size,
        quantity, promotions, location and
        receipt date before making a conclusion.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.markdown("")

    st.markdown(
        """
        <div class="section-card">

        <h3>🚀 Ready to verify your receipt?</h3>

        <p>
        Upload a clear shopping receipt above.
        PriceProof AI will read the receipt,
        compare available prices and explain
        the results.
        </p>

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

        <strong>PriceProof AI</strong>
        <br>

        Detect overpricing. Verify the price.
        Know your rights.

        <br><br>

        Built for AI-powered consumer price transparency.

    </div>
    """,
    unsafe_allow_html=True
)
