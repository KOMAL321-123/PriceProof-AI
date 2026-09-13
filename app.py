import streamlit as st
import pandas as pd
from PIL import Image
from groq import Groq
import base64
import json
import re
import os
import io
import difflib
from datetime import datetime
import plotly.express as px


# ============================================================
# PRICEPROOF AI
# Step 27.1 - Stable + Clean UI + Token Fix
# ============================================================

st.set_page_config(
    page_title="PriceProof AI",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================

VISION_MODEL = "qwen/qwen3.6-27b"
TEXT_MODEL = "openai/gpt-oss-20b"

MAX_IMAGE_BYTES = 20 * 1024 * 1024

SUPPORTED_IMAGE_TYPES = [
    "jpg",
    "jpeg",
    "png",
    "webp"
]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #f8fafc 0%,
            #eef2ff 100%
        );
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .hero {
        background: linear-gradient(
            135deg,
            #111827 0%,
            #1e3a8a 55%,
            #312e81 100%
        );
        padding: 2.5rem;
        border-radius: 24px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.12);
    }

    .hero h1 {
        font-size: 3rem;
        margin-bottom: 0.3rem;
        font-weight: 800;
    }

    .hero p {
        font-size: 1.15rem;
        opacity: 0.92;
        margin-bottom: 0.3rem;
    }

    .hero-tag {
        display: inline-block;
        background: rgba(255,255,255,0.14);
        padding: 0.45rem 0.9rem;
        border-radius: 999px;
        margin-top: 0.8rem;
        font-size: 0.9rem;
    }

    .info-card {
        background: white;
        border-radius: 18px;
        padding: 1.3rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }

    .metric-card {
        background: white;
        padding: 1.25rem;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.06);
        min-height: 120px;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.88rem;
        margin-bottom: 0.4rem;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #111827;
    }

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

    .status-info {
        background: #eff6ff;
        border-left: 5px solid #3b82f6;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        padding: 2rem 0 1rem 0;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
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

if "analysis_running" not in st.session_state:
    st.session_state.analysis_running = False

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

    name = re.sub(
        r"[^a-z0-9\s]",
        " ",
        name
    )

    name = re.sub(
        r"\s+",
        " ",
        name
    )

    return name.strip()


def encode_image_bytes(image_bytes, mime_type):

    encoded = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def detect_mime_type(filename):

    extension = filename.lower().split(".")[-1]

    mapping = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp"
    }

    return mapping.get(
        extension,
        "image/jpeg"
    )


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

    cleaned = cleaned.replace(
        "```",
        ""
    ).strip()

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
            return json.loads(
                object_match.group(0)
            )
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

        df = pd.read_csv(
            "products.csv"
        )

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

            return normalized[
                candidate.lower()
            ]

    for column in df.columns:

        column_lower = str(
            column
        ).lower()

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

    result["product"] = df[
        product_col
    ].astype(str)

    result["reference_price"] = df[
        price_col
    ].apply(safe_float)

    brand_col = find_column(
        df,
        ["brand"]
    )

    category_col = find_column(
        df,
        ["category"]
    )

    size_col = find_column(
        df,
        [
            "size",
            "package_size",
            "pack_size"
        ]
    )

    date_col = find_column(
        df,
        [
            "date",
            "price_date",
            "updated"
        ]
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

def match_product(
    product_name,
    reference_data
):

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

            score = max(
                score,
                0.86
            )

        if score > best_score:

            best_score = score
            best_index = index

    if best_index is None or best_score < 0.55:

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

    row = reference_data.loc[
        best_index
    ]

    return {
        "matched": True,
        "product": product_name,
        "reference_product": row["product"],
        "reference_price": safe_float(
            row["reference_price"]
        ),
        "similarity": best_score,
        "brand": clean_text(
            row["brand"]
        ),
        "category": clean_text(
            row["category"]
        ),
        "size": clean_text(
            row["size"]
        ),
        "date": clean_text(
            row["date"]
        )
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
        response
        .choices[0]
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

    items = data.get(
        "items",
        []
    )

    if not isinstance(items, list):

        items = []

    normalized_items = []

    for item in items:

        if not isinstance(
            item,
            dict
        ):

            continue

        name = clean_text(
            item.get(
                "name",
                ""
            )
        )

        if not name:
            continue

        quantity = safe_float(
            item.get(
                "quantity",
                1
            ),
            1
        )

        if quantity <= 0:
            quantity = 1

        unit_price = safe_float(
            item.get(
                "unit_price",
                0
            )
        )

        line_total = safe_float(
            item.get(
                "line_total",
                0
            )
        )

        if (
            line_total <= 0
            and unit_price > 0
        ):

            line_total = (
                unit_price
                * quantity
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
            data.get(
                "store_name",
                ""
            )
        ),
        "receipt_date": clean_text(
            data.get(
                "receipt_date",
                ""
            )
        ),
        "currency": clean_text(
            data.get(
                "currency",
                "PKR"
            )
        ) or "PKR",
        "total": safe_float(
            data.get(
                "total",
                0
            )
        ),
        "items": normalized_items
    }


# ============================================================
# PRICE ANALYSIS
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
            item.get(
                "quantity",
                1
            ),
            1
        )

        unit_price = safe_float(
            item.get(
                "unit_price",
                0
            )
        )

        line_total = safe_float(
            item.get(
                "line_total",
                0
            )
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
                reference_price
                * quantity
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
# PRICE HEALTH
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
        response
        .choices[0]
        .message
        .content
        .strip()
    )


# ============================================================
# AI QUESTION ANSWERING
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
        response
        .choices[0]
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
        receipt_data.get(
            "total",
            0
        )
    )

    if receipt_total <= 0:
        return None

    if (
        analysis_df is None
        or analysis_df.empty
    ):

        return None

    item_total = analysis_df[
        "Charged Total"
    ].apply(
        safe_float
    ).sum()

    return {
        "receipt_total": receipt_total,
        "item_total": item_total,
        "difference": (
            receipt_total
            - item_total
        )
    }


# ============================================================
# REPORT
# ============================================================

def generate_consumer_report(
    analysis_df,
    receipt_data
):

    lines = []

    lines.append(
        "PRICEPROOF AI - CONSUMER PRICE VERIFICATION REPORT"
    )

    lines.append(
        "=" * 55
    )

    lines.append(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

    lines.append("")

    if receipt_data.get(
        "store_name"
    ):

        lines.append(
            f"Store: {receipt_data['store_name']}"
        )

    if receipt_data.get(
        "receipt_date"
    ):

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

    lines.append(
        "ITEM ANALYSIS"
    )

    lines.append(
        "-" * 55
    )

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
                "Reference: No matching benchmark"
            )

        lines.append(
            f"Status: {row['Status']}"
        )

        lines.append("")

    lines.append(
        "-" * 55
    )

    lines.append(
        "Reference prices are benchmarks, not official "
        "government or legal prices."
    )

    lines.append(
        "A potentially high-price result does not by itself "
        "prove illegal overcharging."
    )

    return "\n".join(lines)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🧾 PriceProof AI</h1>
        <p>
            Detect overpricing. Verify the price. Know your rights.
        </p>
        <div class="hero-tag">
            AI-powered receipt & price verification
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧾 PriceProof AI")

    st.write(
        "AI-powered receipt and price verification."
    )

    st.divider()

    st.subheader("📊 Reference Database")

    if reference_df.empty:

        st.warning(
            "products.csv is missing or invalid."
        )

    else:

        st.success(
            f"{len(reference_df)} products loaded"
        )

    st.divider()

    st.subheader("🔐 Privacy")

    st.caption(
        "Receipt images are processed for the analysis "
        "session and are not intentionally stored by this app."
    )

    st.divider()

    st.caption(
        "Reference prices are benchmarks, not official legal prices."
    )


# ============================================================
# ABOUT
# ============================================================

with st.expander(
    "ℹ️ About PriceProof AI"
):

    st.write(
        """
        **PriceProof AI** helps consumers understand whether
        prices on a shopping receipt appear reasonable compared
        with reference market-price data.

        AI reads the receipt, extracts products and prices,
        and compares them against available reference benchmarks.
        """
    )

    st.info(
        "A potentially high-price flag is a review signal, "
        "not proof of illegal overcharging."
    )


# ============================================================
# HOW IT WORKS
# ============================================================

with st.expander(
    "🚀 How PriceProof AI Works",
    expanded=True
):

    step1, step2, step3, step4, step5 = st.columns(5)

    with step1:
        st.markdown("### 1️⃣")
        st.markdown("**Upload Receipt**")
        st.caption(
            "Upload a shopping receipt."
        )

    with step2:
        st.markdown("### 2️⃣")
        st.markdown("**AI Reads It**")
        st.caption(
            "AI extracts products and prices."
        )

    with step3:
        st.markdown("### 3️⃣")
        st.markdown("**Verify Prices**")
        st.caption(
            "Prices are compared with reference data."
        )

    with step4:
        st.markdown("### 4️⃣")
        st.markdown("**AI Explains**")
        st.caption(
            "AI explains important differences."
        )

    with step5:
        st.markdown("### 5️⃣")
        st.markdown("**Generate Report**")
        st.caption(
            "Create a consumer verification report."
        )


# ============================================================
# GROQ CLIENT
# ============================================================

client = get_groq_client()

if client is None:

    st.error(
        "GROQ_API_KEY is not configured."
    )

    st.info(
        "Add GROQ_API_KEY in Streamlit Cloud → Settings → Secrets."
    )

    st.stop()


# ============================================================
# UPLOAD
# ============================================================

st.header("📤 Upload Your Receipt")

st.caption(
    "Upload a clear receipt image to begin the verification process."
)

uploaded_file = st.file_uploader(
    "Choose a receipt image",
    type=SUPPORTED_IMAGE_TYPES
)


if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    if len(image_bytes) > MAX_IMAGE_BYTES:

        st.error(
            "The image is larger than 20 MB. "
            "Please upload a smaller image."
        )

        st.stop()

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image.verify()

    except Exception:

        st.error(
            "The uploaded file is not a valid image."
        )

        st.stop()

    st.session_state.receipt_image_bytes = image_bytes
    st.session_state.receipt_filename = uploaded_file.name

    left, right = st.columns(
        [1, 1]
    )

    with left:

        st.image(
            image_bytes,
            caption=uploaded_file.name,
            use_container_width=True
        )

    with right:

        st.subheader(
            "🔍 Ready for Analysis"
        )

        st.write(
            "The receipt image is ready to be processed "
            "by the AI vision model."
        )

        analyze_button = st.button(
            "🔍 Analyze Receipt with AI",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.analysis_running
        )

        if analyze_button:

            st.session_state.analysis_running = True

            with st.spinner(
                "AI is reading your receipt..."
            ):

                try:

                    mime_type = detect_mime_type(
                        uploaded_file.name
                    )

                    extracted = extract_receipt_with_ai(
                        client,
                        image_bytes,
                        mime_type
                    )

                    normalized = normalize_receipt_data(
                        extracted
                    )

                    if not normalized["items"]:

                        st.warning(
                            "No readable receipt items were detected. "
                            "Please try a clearer receipt image."
                        )

                        st.session_state.analysis_complete = False

                    else:

                        comparison = analyze_receipt_items(
                            normalized,
                            reference_df
                        )

                        st.session_state.receipt_data = normalized

                        st.session_state.analysis_df = comparison

                        st.session_state.analysis_complete = True

                        st.session_state.ai_explanation = None

                        st.success(
                            "Receipt analyzed successfully."
                        )
                        st.session_state.analysis_running = False

                except Exception as error:

                    error_text = str(error)

                    st.session_state.analysis_complete = False
                    st.session_state.analysis_running = False

                    if "429" in error_text:

                        st.error(
                            "Groq rate limit reached while reading "
                            "the receipt."
                        )

                        st.info(
                            "The app has been optimized to use a small "
                            "receipt-extraction response. Please wait "
                            "a short moment and try Analyze again."
                        )

                    else:

                        st.error(
                            "Receipt analysis failed."
                        )

                        st.caption(
                            f"Technical detail: {error_text}"
                        )


# ============================================================
# RESULTS
# ============================================================

if st.session_state.analysis_complete:

    receipt_data = st.session_state.receipt_data

    analysis_df = st.session_state.analysis_df

    st.divider()

    st.header(
        "📊 Price Verification Dashboard"
    )

    total_items = len(
        analysis_df
    )

    matched_items = int(
        analysis_df[
            "Reference Unit Price"
        ].notna().sum()
    )

    potentially_overpriced = int(
        analysis_df[
            "Potentially Overpriced"
        ].sum()
    )

    receipt_total = safe_float(
        receipt_data.get(
            "total",
            0
        )
    )

    health_score = calculate_price_health_score(
        analysis_df
    )

    # ========================================================
    # METRICS
    # ========================================================

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "Receipt Items",
            total_items
        )

    with c2:
        st.metric(
            "Matched",
            matched_items
        )

    with c3:
        st.metric(
            "Potentially High",
            potentially_overpriced
        )

    with c4:
        st.metric(
            "Receipt Total",
            format_currency(
                receipt_total
            )
        )

    with c5:
        st.metric(
            "Price Health",
            (
                f"{health_score}/100"
                if health_score is not None
                else "N/A"
            )
        )


    # ========================================================
    # QUICK RESULT SUMMARY
    # ========================================================

    st.subheader("🎯 Price Check Complete")

    within_reference = int(
        (
            (analysis_df["Reference Unit Price"].notna())
            & (~analysis_df["Potentially Overpriced"])
        ).sum()
    )

    unmatched_items = int(
        analysis_df["Reference Unit Price"].isna().sum()
    )

    matched_positive_differences = analysis_df[
        analysis_df["Reference Unit Price"].notna()
    ]["Difference %"].dropna()

    highest_difference = (
        matched_positive_differences.max()
        if not matched_positive_differences.empty
        else None
    )

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.metric(
            "⚠️ Potentially High",
            potentially_overpriced
        )

    with s2:
        st.metric(
            "✅ Within Benchmark",
            within_reference
        )

    with s3:
        st.metric(
            "❓ No Reference Match",
            unmatched_items
        )

    with s4:
        st.metric(
            "📈 Highest Difference",
            (
                f"+{highest_difference:.1f}%"
                if highest_difference is not None
                else "N/A"
            )
        )

    if potentially_overpriced > 0:
        st.warning(
            f"⚠️ {potentially_overpriced} item(s) may deserve further review based on the available benchmark data."
        )
    elif unmatched_items > 0:
        st.info(
            "ℹ️ No potentially high-priced items were detected, but some items could not be matched with the reference database."
        )
    else:
        st.success(
            "✅ No potentially high-priced items were detected against the available benchmarks."
        )


    # ========================================================
    # RECEIPT INFORMATION
    # ========================================================

    st.subheader(
        "🧾 Receipt Information"
    )

    r1, r2, r3 = st.columns(3)

    with r1:
        st.info(
            f"🏪 Store\n\n"
            f"{receipt_data.get('store_name') or 'Not detected'}"
        )

    with r2:
        st.info(
            f"📅 Date\n\n"
            f"{receipt_data.get('receipt_date') or 'Not detected'}"
        )

    with r3:
        st.info(
            f"💰 Total\n\n"
            f"{format_currency(receipt_total)}"
        )


    # ========================================================
    # PRICE HEALTH
    # ========================================================

    if health_score is not None:

        st.subheader(
            "❤️ Price Health Score"
        )

        st.progress(
            health_score / 100
        )

        label = get_health_label(
            health_score
        )

        if health_score >= 90:

            st.success(
                f"{label} — {health_score}/100"
            )

        elif health_score >= 75:

            st.info(
                f"{label} — {health_score}/100"
            )

        elif health_score >= 60:

            st.warning(
                f"{label} — {health_score}/100"
            )

        else:

            st.warning(
                f"{label} — {health_score}/100"
            )


    # ========================================================
    # TOTAL CHECK
    # ========================================================

    consistency = calculate_total_consistency(
        receipt_data,
        analysis_df
    )

    if consistency:

        st.subheader(
            "🧮 Receipt Total Check"
        )

        receipt_total_check = consistency[
            "receipt_total"
        ]

        item_total_check = consistency[
            "item_total"
        ]

        difference_check = consistency[
            "difference"
        ]

        if abs(difference_check) <= 1:

            st.success(
                "✓ Receipt total is consistent with "
                "the extracted item totals."
            )

        else:

            st.warning(
                "⚠ The receipt total differs from the "
                "sum of extracted items."
            )

            st.caption(
                f"Receipt: {format_currency(receipt_total_check)} | "
                f"Items: {format_currency(item_total_check)} | "
                f"Difference: {format_currency(abs(difference_check))}"
            )


    # ========================================================
    # FLAGGED ITEMS
    # ========================================================

    st.subheader(
        "🔎 Items Worth Reviewing"
    )

    flagged_df = analysis_df[
        analysis_df[
            "Potentially Overpriced"
        ] == True
    ]

    if not flagged_df.empty:

        st.warning(
            f"{len(flagged_df)} item(s) may deserve further review."
        )

        for _, row in flagged_df.iterrows():

            percentage = safe_float(
                row["Difference %"]
            )

            with st.expander(
                f"⚠️ {row['Product']} — {percentage:.1f}% above benchmark"
            ):

                a, b, c = st.columns(3)

                with a:
                    st.metric(
                        "Charged",
                        format_currency(
                            row["Charged Total"]
                        )
                    )

                with b:
                    st.metric(
                        "Reference",
                        format_currency(
                            row["Reference Total"]
                        )
                    )

                with c:
                    st.metric(
                        "Difference",
                        format_currency(
                            row["Difference"]
                        )
                    )

                st.markdown(
                    "#### 🔎 What should you check?"
                )

                st.write(
                    """
                    • Brand and exact product  
                    • Package size or weight  
                    • Quantity purchased  
                    • Promotions or discounts  
                    • Receipt date  
                    • Local market price
                    """
                )

    else:

        st.success(
            "✓ No potentially high-priced items detected "
            "against the available benchmarks."
        )


    # ========================================================
    # PRODUCT TABLE
    # ========================================================

    st.subheader(
        "📋 Product Analysis"
    )

    display_df = analysis_df[
        [
            "Product",
            "Quantity",
            "Charged Unit Price",
            "Reference Unit Price",
            "Difference",
            "Difference %",
            "Status"
        ]
    ].copy()

    display_df = display_df.rename(
        columns={
            "Charged Unit Price": "Charged",
            "Reference Unit Price": "Reference"
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # CHART
    # ========================================================

    chart_df = analysis_df[
        analysis_df[
            "Reference Unit Price"
        ].notna()
    ].copy()

    if not chart_df.empty:

        st.subheader(
            "📈 Charged vs Reference Price"
        )

        chart_data = chart_df[
            [
                "Product",
                "Charged Unit Price",
                "Reference Unit Price"
            ]
        ].copy()

        chart_data = chart_data.rename(
            columns={
                "Charged Unit Price": "Charged",
                "Reference Unit Price": "Reference"
            }
        )

        chart_data = chart_data.melt(
            id_vars=["Product"],
            value_vars=[
                "Charged",
                "Reference"
            ],
            var_name="Price Type",
            value_name="Price"
        )

        fig = px.bar(
            chart_data,
            x="Product",
            y="Price",
            color="Price Type",
            barmode="group"
        )

        fig.update_layout(
            height=500,
            xaxis_title="Product",
            yaxis_title="Price (PKR)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ========================================================
    # AI EXPLANATION
    # ========================================================

    st.subheader(
        "🤖 AI Price Analysis"
    )

    if st.button(
        "✨ Generate AI Explanation",
        use_container_width=True
    ):

        with st.spinner(
            "Preparing a concise explanation..."
        ):

            try:

                st.session_state.ai_explanation = (
                    generate_ai_explanation(
                        client,
                        analysis_df
                    )
                )

            except Exception as error:

                if "429" in str(error):

                    st.warning(
                        "AI explanation rate limit reached. "
                        "Please wait briefly and try again."
                    )

                else:

                    st.error(
                        "Could not generate the explanation."
                    )

                    st.caption(
                        str(error)
                    )

    if st.session_state.ai_explanation:

        st.info(
            st.session_state.ai_explanation
        )


    # ========================================================
    # ASK AI
    # ========================================================

    st.subheader(
        "💬 Ask PriceProof AI"
    )

    st.caption(
        "Ask any question about your analyzed receipt."
    )

    example_questions = [
        "Which item should I check first?",
        "How much more did I pay than the benchmarks?",
        "Which products have no reference match?",
        "What is the most expensive item?",
        "Are there any suspicious price differences?"
    ]

    selected_question = st.selectbox(
        "Quick question",
        [
            "Choose a question..."
        ] + example_questions
    )

    custom_question = st.text_input(
        "Or type your own question",
        placeholder=(
            "Example: Which item has the biggest price difference?"
        )
    )

    if st.button(
        "💬 Ask AI",
        type="primary"
    ):

        question = (
            custom_question.strip()
            if custom_question.strip()
            else (
                selected_question
                if selected_question != "Choose a question..."
                else ""
            )
        )

        if not question:

            st.warning(
                "Please enter or select a question."
            )

        else:

            with st.spinner(
                "PriceProof AI is thinking..."
            ):

                try:

                    answer = answer_user_question(
                        client,
                        question,
                        receipt_data,
                        analysis_df
                    )

                    st.session_state.chat_history.append(
                        {
                            "question": question,
                            "answer": answer
                        }
                    )

                except Exception as error:

                    if "429" in str(error):

                        st.warning(
                            "AI rate limit reached. "
                            "Please wait briefly and try again."
                        )

                    else:

                        st.error(
                            "Could not answer the question."
                        )

                        st.caption(
                            str(error)
                        )

    if st.session_state.chat_history:

        for chat in reversed(
            st.session_state.chat_history
        ):

            st.markdown(
                f"**👤 You:** {chat['question']}"
            )

            st.markdown(
                f"**🤖 PriceProof AI:** {chat['answer']}"
            )

            st.divider()


    # ========================================================
    # MANUAL VERIFICATION
    # ========================================================

    st.subheader(
        "📝 Manual Verification"
    )

    if not reference_df.empty:

        manual_product = st.selectbox(
            "Select a reference product",
            reference_df[
                "product"
            ].tolist()
        )

        selected = reference_df[
            reference_df[
                "product"
            ] == manual_product
        ]

        if not selected.empty:

            row = selected.iloc[0]

            m1, m2, m3 = st.columns(3)

            with m1:

                st.metric(
                    "Reference Price",
                    format_currency(
                        row["reference_price"]
                    )
                )

            with m2:

                st.metric(
                    "Brand",
                    row["brand"]
                    or "N/A"
                )

            with m3:

                st.metric(
                    "Category",
                    row["category"]
                    or "N/A"
                )

    else:

        st.info(
            "Manual verification requires products.csv."
        )


    # ========================================================
    # DOWNLOADS
    # ========================================================

    st.subheader(
        "📥 Download Results"
    )

    report = generate_consumer_report(
        analysis_df,
        receipt_data
    )

    d1, d2, d3 = st.columns(3)

    with d1:

        st.download_button(
            "📊 Download CSV",
            data=analysis_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="priceproof_analysis.csv",
            mime="text/csv",
            use_container_width=True
        )

    with d2:

        json_output = json.dumps(
            {
                "receipt": receipt_data,
                "analysis": analysis_df.fillna(
                    ""
                ).to_dict(
                    orient="records"
                )
            },
            indent=2,
            default=str
        )

        st.download_button(
            "🗂️ Download JSON",
            data=json_output,
            file_name="priceproof_analysis.json",
            mime="application/json",
            use_container_width=True
        )

    with d3:

        st.download_button(
            "📄 Download Report",
            data=report,
            file_name="priceproof_consumer_report.txt",
            mime="text/plain",
            use_container_width=True
        )


    # ========================================================
    # REFERENCE DATABASE
    # ========================================================

    with st.expander(
        "📚 View Reference Database"
    ):

        if reference_df.empty:

            st.warning(
                "No reference database available."
            )

        else:

            st.dataframe(
                reference_df,
                use_container_width=True,
                hide_index=True
            )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.divider()

    st.info(
        "ℹ️ PriceProof AI uses reference market-price data "
        "as a benchmark. A potentially high-price result "
        "does not by itself prove illegal overcharging. "
        "Consumers should verify brand, package size, "
        "quantity, promotions, receipt date and local "
        "market conditions."
    )


# ============================================================
# CLEAN EMPTY STATE
# ============================================================

else:

    st.divider()

    st.header(
        "🧾 Start Your Price Check"
    )

    st.write(
        "Upload a clear shopping receipt above. "
        "PriceProof AI will extract the products, "
        "compare prices with reference benchmarks, "
        "and identify items that may deserve further review."
    )

    e1, e2, e3 = st.columns(3)

    with e1:

        st.info(
            "📷 Read Receipt\n\n"
            "AI vision extracts products, "
            "quantities and prices."
        )

    with e2:

        st.info(
            "🔎 Verify Prices\n\n"
            "Receipt prices are compared "
            "against reference benchmarks."
        )

    with e3:

        st.info(
            "💡 Understand Results\n\n"
            "Get clear explanations and "
            "review recommendations."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <strong>PriceProof AI</strong><br>
        Detect overpricing. Verify the price. Know your rights.
        <br><br>
        AI-generated analysis is a verification aid,
        not a legal determination.
    </div>
    """,
    unsafe_allow_html=True
)
