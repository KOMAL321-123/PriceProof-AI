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
# Step 27 - Stable Professional Version
# ============================================================


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PriceProof AI",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .stApp {
        background: linear-gradient(
            135deg,
            #f8fafc 0%,
            #eef2ff 100%
        );
    }

    /* Main content width */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Hero */
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

    /* Cards */
    .info-card {
        background: white;
        border-radius: 18px;
        padding: 1.3rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }

    .info-card h3 {
        margin-top: 0;
    }

    /* Metric cards */
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
        font-size: 2rem;
        font-weight: 800;
        color: #111827;
    }

    /* Status cards */
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

    /* Product card */
    .product-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 0.7rem;
    }

    /* Small text */
    .small-muted {
        color: #64748b;
        font-size: 0.85rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        padding: 2rem 0 1rem 0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #ffffff;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
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

if "last_error" not in st.session_state:
    st.session_state.last_error = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_groq_client():
    """
    Creates the Groq client from Streamlit secrets
    or environment variables.
    """

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
    """
    Convert a value safely to float.
    """

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
    """
    Convert arbitrary value to clean string.
    """

    if value is None:
        return ""

    return str(value).strip()


def normalize_product_name(name):
    """
    Normalize product names for matching.
    """

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
    """
    Convert image bytes into a base64 data URL.
    """

    encoded = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def detect_mime_type(filename):
    """
    Detect image MIME type.
    """

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
    """
    Attempts to extract JSON from an AI response.
    """

    if not text:
        return None

    text = text.strip()

    # Direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Remove markdown fences if model returns them
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

    # Find object
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
    """
    Format amount as PKR.
    """

    value = safe_float(value)

    return f"PKR {value:,.2f}"


def percentage_difference(charged, reference):
    """
    Calculate percentage difference.
    """

    charged = safe_float(charged)
    reference = safe_float(reference)

    if reference <= 0:
        return 0.0

    return ((charged - reference) / reference) * 100


# ============================================================
# PRODUCT DATABASE
# ============================================================

@st.cache_data
def load_products():
    """
    Loads products.csv.

    The application supports several reasonable column names.
    """

    possible_files = [
        "products.csv",
        "./products.csv"
    ]

    file_path = None

    for path in possible_files:
        if os.path.exists(path):
            file_path = path
            break

    if not file_path:
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)

        df.columns = [
            str(column).strip()
            for column in df.columns
        ]

        return df

    except Exception:
        return pd.DataFrame()


products_df = load_products()


def find_column(df, candidates):
    """
    Finds a matching dataframe column.
    """

    if df.empty:
        return None

    normalized = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        key = candidate.lower()

        if key in normalized:
            return normalized[key]

    # Fuzzy fallback
    for column in df.columns:
        column_lower = str(column).lower()

        for candidate in candidates:
            if candidate.lower() in column_lower:
                return column

    return None


def prepare_reference_database(df):
    """
    Converts products.csv into a standard structure.
    """

    if df.empty:
        return pd.DataFrame(
            columns=[
                "product",
                "reference_price"
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
                "reference_price"
            ]
        )

    result = pd.DataFrame()

    result["product"] = df[
        product_col
    ].astype(str)

    result["reference_price"] = df[
        price_col
    ].apply(safe_float)

    # Optional columns
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

    if brand_col:
        result["brand"] = df[
            brand_col
        ].astype(str)

    else:
        result["brand"] = ""

    if category_col:
        result["category"] = df[
            category_col
        ].astype(str)

    else:
        result["category"] = ""

    if size_col:
        result["size"] = df[
            size_col
        ].astype(str)

    else:
        result["size"] = ""

    if date_col:
        result["date"] = df[
            date_col
        ].astype(str)

    else:
        result["date"] = ""

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
    """
    Matches an extracted receipt item against products.csv.
    """

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

        # Exact/containment bonus
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

    # Minimum match threshold
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
            row.get("brand", "")
        ),
        "category": clean_text(
            row.get("category", "")
        ),
        "size": clean_text(
            row.get("size", "")
        ),
        "date": clean_text(
            row.get("date", "")
        )
    }


# ============================================================
# RECEIPT EXTRACTION
# ============================================================

def extract_receipt_with_ai(
    client,
    image_bytes,
    mime_type
):
    """
    Extract receipt information using Groq vision.
    """

    image_url = encode_image_bytes(
        image_bytes,
        mime_type
    )

    prompt = """
You are a receipt-analysis assistant.

Analyze the uploaded shopping receipt carefully.

Extract ONLY information that can reasonably be read from the receipt.

Return a valid JSON object.

Required JSON structure:

{
  "store_name": "",
  "receipt_date": "",
  "currency": "PKR",
  "subtotal": 0,
  "tax": 0,
  "discount": 0,
  "total": 0,
  "items": [
    {
      "name": "",
      "quantity": 1,
      "unit_price": 0,
      "line_total": 0
    }
  ]
}

Important rules:

1. Do not invent products.
2. Do not invent prices.
3. If something cannot be read, use an empty string or 0.
4. Preserve product names as closely as possible.
5. If quantity is unclear, use 1.
6. If a line has one total price, use that as line_total.
7. If unit price is unclear, use line_total when appropriate.
8. Return JSON only.
"""

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract structured information "
                    "from receipts accurately."
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
                            "url": image_url
                        }
                    }
                ]
            }
        ],
        temperature=0,
        max_completion_tokens=3000,
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
            "The AI did not return valid receipt data."
        )

    return data


# ============================================================
# NORMALIZE RECEIPT DATA
# ============================================================

def normalize_receipt_data(data):
    """
    Normalize extracted receipt JSON.
    """

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

        if not isinstance(item, dict):
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

        if line_total <= 0 and unit_price > 0:
            line_total = (
                unit_price * quantity
            )

        if unit_price <= 0 and line_total > 0:
            unit_price = (
                line_total / quantity
                if quantity > 0
                else line_total
            )

        normalized_items.append(
            {
                "name": name,
                "quantity": quantity,
                "unit_price": unit_price,
                "line_total": line_total
            }
        )

    normalized = {
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
        "subtotal": safe_float(
            data.get(
                "subtotal",
                0
            )
        ),
        "tax": safe_float(
            data.get(
                "tax",
                0
            )
        ),
        "discount": safe_float(
            data.get(
                "discount",
                0
            )
        ),
        "total": safe_float(
            data.get(
                "total",
                0
            )
        ),
        "items": normalized_items
    }

    return normalized


# ============================================================
# BUILD PRICE ANALYSIS
# ============================================================

def analyze_receipt_items(
    receipt_data,
    reference_data
):
    """
    Compare receipt items against reference prices.
    """

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

            # Flag only potentially high prices.
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
                "Potentially Overpriced": (
                    potentially_overpriced
                ),
                "Match Score": (
                    match.get(
                        "similarity",
                        0
                    )
                ),
                "Reference Product": (
                    match.get(
                        "reference_product",
                        ""
                    )
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
# AI EXPLANATION
# ============================================================

def generate_ai_explanation(
    client,
    analysis_df,
    receipt_data
):
    """
    Generates a concise consumer-friendly analysis.
    """

    if analysis_df.empty:
        return (
            "No receipt items were available for analysis."
        )

    records = []

    for _, row in analysis_df.iterrows():

        records.append(
            {
                "product": row["Product"],
                "charged_total": safe_float(
                    row["Charged Total"]
                ),
                "reference_total": (
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
You are PriceProof AI, a consumer price verification assistant.

Analyze the following receipt comparison data.

Receipt:
{json.dumps(receipt_data, indent=2)}

Comparison:
{json.dumps(records, indent=2)}

Give a concise explanation for the consumer.

Requirements:
- Clearly mention potentially high-priced items.
- Explain the price difference simply.
- Mention unmatched items separately.
- Do not claim that a price is legally illegal.
- Do not claim that the reference price is an official government price.
- Explain that reference prices are benchmarks.
- Suggest checking brand, package size, quantity, promotions, receipt date and local market conditions.
- Keep the answer concise.
- Do not unnecessarily repeat all receipt data.
"""

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You provide concise, accurate "
                    "consumer price explanations."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_completion_tokens=1200
    )

    return (
        response.choices[0]
        .message
        .content
        .strip()
    )


# ============================================================
# AI CHAT
# ============================================================

def answer_user_question(
    client,
    question,
    receipt_data,
    analysis_df
):
    """
    Answers arbitrary questions about the analyzed receipt.
    """

    if analysis_df is None:
        return (
            "Please analyze a receipt first."
        )

    analysis_records = []

    for _, row in analysis_df.iterrows():

        analysis_records.append(
            {
                "product": row["Product"],
                "quantity": safe_float(
                    row["Quantity"]
                ),
                "charged_unit_price": safe_float(
                    row["Charged Unit Price"]
                ),
                "charged_total": safe_float(
                    row["Charged Total"]
                ),
                "reference_unit_price": (
                    safe_float(
                        row["Reference Unit Price"]
                    )
                    if pd.notna(
                        row["Reference Unit Price"]
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

    context = {
        "receipt": receipt_data,
        "analysis": analysis_records
    }

    prompt = f"""
You are PriceProof AI.

Answer the user's question using ONLY the receipt
information and price comparison context provided below.

Context:
{json.dumps(context, indent=2)}

User question:
{question}

Rules:
- Be concise but complete.
- Answer the actual question directly.
- Do not invent missing information.
- If the answer cannot be determined from the receipt,
  say so clearly.
- Do not make legal claims.
- Reference prices are benchmarks, not guaranteed official prices.
- If useful, suggest what the consumer should verify.
"""

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful receipt and "
                    "consumer price assistant."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_completion_tokens=800
    )

    return (
        response.choices[0]
        .message
        .content
        .strip()
    )


# ============================================================
# CONSUMER REPORT
# ============================================================

def generate_consumer_report(
    analysis_df,
    receipt_data
):
    """
    Creates a plain-text verification report.
    """

    lines = []

    lines.append(
        "PRICEPROOF AI - CONSUMER PRICE VERIFICATION REPORT"
    )

    lines.append(
        "=" * 58
    )

    lines.append(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

    lines.append("")

    store = receipt_data.get(
        "store_name",
        ""
    )

    date = receipt_data.get(
        "receipt_date",
        ""
    )

    total = safe_float(
        receipt_data.get(
            "total",
            0
        )
    )

    if store:
        lines.append(
            f"Store: {store}"
        )

    if date:
        lines.append(
            f"Receipt Date: {date}"
        )

    lines.append(
        f"Receipt Total: {format_currency(total)}"
    )

    lines.append("")

    lines.append(
        "ITEM ANALYSIS"
    )

    lines.append(
        "-" * 58
    )

    if analysis_df.empty:

        lines.append(
            "No item analysis available."
        )

    else:

        for _, row in analysis_df.iterrows():

            lines.append(
                f"Product: {row['Product']}"
            )

            lines.append(
                f"Charged: {format_currency(row['Charged Total'])}"
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
                    + " "
                    + (
                        f"({row['Difference %']:.1f}%)"
                    )
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
        "-" * 58
    )

    lines.append(
        "IMPORTANT:"
    )

    lines.append(
        "Reference prices are benchmarks for comparison "
        "and are not official government/legal prices."
    )

    lines.append(
        "A potentially high-price flag does not by itself "
        "prove illegal overcharging."
    )

    return "\n".join(lines)


# ============================================================
# CSV EXPORT
# ============================================================

def dataframe_to_csv(df):
    """
    Convert dataframe to CSV bytes.
    """

    return df.to_csv(
        index=False
    ).encode("utf-8")


# ============================================================
# PRICE HEALTH SCORE
# ============================================================

def calculate_price_health_score(
    analysis_df
):
    """
    Calculates a simple 0-100 benchmark score.
    """

    if analysis_df is None or analysis_df.empty:
        return None

    matched = analysis_df[
        analysis_df["Reference Unit Price"].notna()
    ]

    if matched.empty:
        return None

    scores = []

    for _, row in matched.iterrows():

        diff = safe_float(
            row["Difference %"]
        )

        if diff <= 0:
            item_score = 100

        elif diff <= 5:
            item_score = 95

        elif diff <= 10:
            item_score = 85

        elif diff <= 20:
            item_score = 70

        elif diff <= 35:
            item_score = 50

        else:
            item_score = 25

        scores.append(
            item_score
        )

    if not scores:
        return None

    return round(
        sum(scores) / len(scores)
    )


def get_health_label(score):
    """
    Converts score to human-friendly label.
    """

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
# RECEIPT TOTAL CONSISTENCY
# ============================================================

def calculate_total_consistency(
    receipt_data,
    analysis_df
):
    """
    Compare extracted item totals against receipt total.
    """

    receipt_total = safe_float(
        receipt_data.get(
            "total",
            0
        )
    )

    if receipt_total <= 0:
        return None

    if analysis_df is None or analysis_df.empty:
        return None

    item_total = analysis_df[
        "Charged Total"
    ].apply(
        safe_float
    ).sum()

    difference = (
        receipt_total
        - item_total
    )

    return {
        "receipt_total": receipt_total,
        "item_total": item_total,
        "difference": difference
    }


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

    st.markdown(
        "## 🧾 PriceProof AI"
    )

    st.markdown(
        """
        Upload a receipt and let AI extract the products,
        compare prices with reference benchmarks, and
        highlight items that may need review.
        """
    )

    st.divider()

    st.markdown(
        "### 📊 Reference Database"
    )

    if reference_df.empty:

        st.warning(
            "products.csv not found or no valid price columns were detected."
        )

    else:

        st.success(
            f"{len(reference_df)} reference products loaded"
        )

    st.divider()

    st.markdown(
        "### 🔐 Privacy"
    )

    st.caption(
        "Receipt images are processed for the analysis "
        "session and are not intentionally stored by this app."
    )

    st.divider()

    st.caption(
        "PriceProof AI"
    )

    st.caption(
        "Reference prices are benchmarks, not official legal prices."
    )


# ============================================================
# ABOUT / HOW IT WORKS
# ============================================================

with st.expander(
    "ℹ️ About PriceProof AI"
):

    st.markdown(
        """
        **PriceProof AI** helps consumers understand whether
        prices on a shopping receipt appear reasonable compared
        with reference market-price data.

        The system uses AI to read receipt information,
        identifies products and prices, and compares them
        against benchmark data.
        """
    )

    st.info(
        "A potentially high-price flag is a review signal, "
        "not proof of illegal overcharging."
    )


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
# GROQ CLIENT CHECK
# ============================================================

client = get_groq_client()

if client is None:

    st.error(
        "⚠️ GROQ_API_KEY is not configured."
    )

    st.info(
        "Add GROQ_API_KEY in Streamlit Cloud → "
        "Settings → Secrets."
    )

    st.stop()


# ============================================================
# RECEIPT UPLOAD
# ============================================================

st.markdown(
    "## 📤 Upload Your Receipt"
)

uploaded_file = st.file_uploader(
    "Choose a receipt image",
    type=SUPPORTED_IMAGE_TYPES,
    help=(
        "Supported formats: JPG, JPEG, PNG and WEBP."
    )
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

        preview_image = Image.open(
            io.BytesIO(image_bytes)
        )

        preview_image.verify()

    except Exception:

        st.error(
            "The uploaded file could not be read as an image."
        )

        st.stop()

    st.session_state.receipt_image_bytes = (
        image_bytes
    )

    st.session_state.receipt_filename = (
        uploaded_file.name
    )

    col_preview, col_action = st.columns(
        [1, 1]
    )

    with col_preview:

        st.image(
            image_bytes,
            caption=uploaded_file.name,
            use_container_width=True
        )

    with col_action:

        st.markdown(
            "### 🔍 Ready for Analysis"
        )

        st.write(
            "The receipt image is ready to be "
            "processed by the AI vision model."
        )

        analyze_button = st.button(
            "🔍 Analyze Receipt with AI",
            type="primary",
            use_container_width=True
        )

        if analyze_button:

            with st.spinner(
                "AI is reading and analyzing your receipt..."
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
                            "Try a clearer image."
                        )

                        st.session_state.analysis_complete = False

                    else:

                        comparison = analyze_receipt_items(
                            normalized,
                            reference_df
                        )

                        st.session_state.receipt_data = (
                            normalized
                        )

                        st.session_state.analysis_df = (
                            comparison
                        )

                        st.session_state.analysis_complete = True

                        st.session_state.last_error = None

                        st.success(
                            "Receipt analyzed successfully."
                        )

                except Exception as error:

                    st.session_state.last_error = str(
                        error
                    )

                    st.session_state.analysis_complete = False

                    st.error(
                        "Receipt analysis failed."
                    )

                    st.caption(
                        f"Technical detail: {error}"
                    )


# ============================================================
# RESULTS
# ============================================================

if st.session_state.analysis_complete:

    receipt_data = (
        st.session_state.receipt_data
    )

    analysis_df = (
        st.session_state.analysis_df
    )

    st.divider()

    st.markdown(
        "## 📊 Price Verification Dashboard"
    )

    # --------------------------------------------------------
    # Summary values
    # --------------------------------------------------------

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

    unmatched_items = (
        total_items
        - matched_items
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

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metric1, metric2, metric3, metric4, metric5 = st.columns(5)

    with metric1:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">
                    Receipt Items
                </div>
                <div class="metric-value">
                    ITEM_COUNT
                </div>
            </div>
            """.replace(
                "ITEM_COUNT",
                str(total_items)
            ),
            unsafe_allow_html=True
        )

    with metric2:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">
                    Matched
                </div>
                <div class="metric-value">
                    MATCHED_COUNT
                </div>
            </div>
            """.replace(
                "MATCHED_COUNT",
                str(matched_items)
            ),
            unsafe_allow_html=True
        )

    with metric3:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">
                    Potentially High
                </div>
                <div class="metric-value">
                    HIGH_COUNT
                </div>
            </div>
            """.replace(
                "HIGH_COUNT",
                str(potentially_overpriced)
            ),
            unsafe_allow_html=True
        )

    with metric4:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">
                    Receipt Total
                </div>
                <div class="metric-value">
                    TOTAL_VALUE
                </div>
            </div>
            """.replace(
                "TOTAL_VALUE",
                format_currency(
                    receipt_total
                )
            ),
            unsafe_allow_html=True
        )

    with metric5:

        score_text = (
            f"{health_score}/100"
            if health_score is not None
            else "N/A"
        )

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">
                    Price Health
                </div>
                <div class="metric-value">
                    SCORE_VALUE
                </div>
            </div>
            """.replace(
                "SCORE_VALUE",
                score_text
            ),
            unsafe_allow_html=True
        )


    # ========================================================
    # RECEIPT INFORMATION
    # ========================================================

    st.markdown(
        "### 🧾 Receipt Information"
    )

    info1, info2, info3 = st.columns(3)

    with info1:

        st.markdown(
            f"""
            <div class="info-card">
                <h4>🏪 Store</h4>
                <p>{receipt_data.get("store_name") or "Not detected"}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with info2:

        st.markdown(
            f"""
            <div class="info-card">
                <h4>📅 Date</h4>
                <p>{receipt_data.get("receipt_date") or "Not detected"}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with info3:

        st.markdown(
            f"""
            <div class="info-card">
                <h4>💰 Total</h4>
                <p>{format_currency(receipt_total)}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # PRICE HEALTH
    # ========================================================

    if health_score is not None:

        health_label = get_health_label(
            health_score
        )

        st.markdown(
            "### ❤️ Price Health Score"
        )

        st.progress(
            health_score / 100
        )

        if health_score >= 90:

            st.success(
                f"**{health_label} — {health_score}/100**. "
                "Most matched prices are close to or below "
                "the reference benchmarks."
            )

        elif health_score >= 75:

            st.info(
                f"**{health_label} — {health_score}/100**. "
                "Most prices look reasonable, but some items "
                "may deserve a closer look."
            )

        elif health_score >= 60:

            st.warning(
                f"**{health_label} — {health_score}/100**. "
                "Several price differences should be reviewed."
            )

        else:

            st.warning(
                f"**{health_label} — {health_score}/100**. "
                "Multiple items show significant differences "
                "from the reference benchmarks."
            )


    # ========================================================
    # TOTAL CONSISTENCY
    # ========================================================

    consistency = calculate_total_consistency(
        receipt_data,
        analysis_df
    )

    if consistency:

        st.markdown(
            "### 🧮 Receipt Total Check"
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

            st.markdown(
                f"""
                <div class="status-good">
                    <strong>✓ Total looks consistent</strong><br>
                    Receipt total:
                    {format_currency(receipt_total_check)}
                    &nbsp; | &nbsp;
                    Sum of extracted items:
                    {format_currency(item_total_check)}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="status-warning">
                    <strong>⚠ Total difference detected</strong><br>
                    Receipt total:
                    {format_currency(receipt_total_check)}
                    &nbsp; | &nbsp;
                    Sum of extracted items:
                    {format_currency(item_total_check)}
                    &nbsp; | &nbsp;
                    Difference:
                    {format_currency(abs(difference_check))}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.caption(
                "The difference may be caused by tax, "
                "discounts, unreadable receipt lines, "
                "rounding or extraction limitations."
            )


    # ========================================================
    # ITEMS WORTH REVIEWING
    # ========================================================

    st.markdown(
        "### 🔎 Items Worth Reviewing"
    )

    flagged_df = analysis_df[
        analysis_df[
            "Potentially Overpriced"
        ] == True
    ].copy()

    if not flagged_df.empty:

        st.markdown(
            f"""
            <div class="status-warning">
                <strong>
                    {len(flagged_df)}
                    item(s) may deserve further review.
                </strong>
                <br>
                These items are more than 10% above their
                reference benchmark.
            </div>
            """,
            unsafe_allow_html=True
        )

        for index, row in flagged_df.iterrows():

            product = row["Product"]

            charged = safe_float(
                row["Charged Total"]
            )

            reference = safe_float(
                row["Reference Total"]
            )

            difference = safe_float(
                row["Difference"]
            )

            percentage = safe_float(
                row["Difference %"]
            )

            with st.expander(
                f"⚠️ {product} — {percentage:.1f}% above benchmark"
            ):

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.metric(
                        "Charged",
                        format_currency(
                            charged
                        )
                    )

                with c2:

                    st.metric(
                        "Reference",
                        format_currency(
                            reference
                        )
                    )

                with c3:

                    st.metric(
                        "Difference",
                        format_currency(
                            difference
                        )
                    )

                st.markdown(
                    "**Why was this flagged?**"
                )

                st.write(
                    f"The charged amount is approximately "
                    f"{percentage:.1f}% above the reference "
                    "benchmark used by PriceProof AI."
                )

                st.markdown(
                    "**What should you check?**"
                )

                st.write(
                    """
                    • Brand and exact product version  
                    • Package size or weight  
                    • Quantity purchased  
                    • Promotional discounts  
                    • Receipt date  
                    • Local market conditions  
                    • Whether the reference product is truly equivalent
                    """
                )

    else:

        st.markdown(
            """
            <div class="status-good">
                <strong>✓ No potentially high-priced items detected.</strong>
                <br>
                Based on the available reference data, no matched
                item is more than 10% above its benchmark.
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # MOST SIGNIFICANT DIFFERENCE
    # ========================================================

    matched_for_difference = analysis_df[
        analysis_df[
            "Reference Unit Price"
        ].notna()
    ].copy()

    if not matched_for_difference.empty:

        matched_for_difference[
            "Abs Difference"
        ] = matched_for_difference[
            "Difference"
        ].abs()

        largest = matched_for_difference.sort_values(
            "Abs Difference",
            ascending=False
        ).iloc[0]

        st.markdown(
            "### 📌 Most Significant Difference"
        )

        st.markdown(
            f"""
            <div class="info-card">

                <h3>
                    {largest["Product"]}
                </h3>

                <p>
                    Charged:
                    <strong>
                        {format_currency(largest["Charged Total"])}
                    </strong>
                </p>

                <p>
                    Reference:
                    <strong>
                        {format_currency(largest["Reference Total"])}
                    </strong>
                </p>

                <p>
                    Difference:
                    <strong>
                        {format_currency(largest["Difference"])}
                    </strong>
                    ({safe_float(largest["Difference %"]):.1f}%)
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # PRODUCT ANALYSIS TABLE
    # ========================================================

    st.markdown(
        "### 📋 Product Analysis"
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
            "Reference Unit Price": "Reference",
            "Difference %": "Difference %"
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # PRICE COMPARISON CHART
    # ========================================================

    chart_df = analysis_df[
        analysis_df[
            "Reference Unit Price"
        ].notna()
    ].copy()

    if not chart_df.empty:

        st.markdown(
            "### 📈 Charged vs Reference Price"
        )

        chart_long = chart_df[
            [
                "Product",
                "Charged Unit Price",
                "Reference Unit Price"
            ]
        ].copy()

        chart_long = chart_long.rename(
            columns={
                "Charged Unit Price": "Charged",
                "Reference Unit Price": "Reference"
            }
        )

        chart_long = chart_long.melt(
            id_vars=["Product"],
            value_vars=[
                "Charged",
                "Reference"
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
            legend_title="",
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ========================================================
    # AI EXPLANATION
    # ========================================================

    st.markdown(
        "### 🤖 AI Price Analysis"
    )

    if st.button(
        "✨ Generate AI Explanation",
        use_container_width=True
    ):

        with st.spinner(
            "AI is preparing a concise explanation..."
        ):

            try:

                explanation = generate_ai_explanation(
                    client,
                    analysis_df,
                    receipt_data
                )

                st.session_state[
                    "ai_explanation"
                ] = explanation

            except Exception as error:

                st.error(
                    "Could not generate the AI explanation."
                )

                st.caption(
                    str(error)
                )

    if st.session_state.get(
        "ai_explanation"
    ):

        st.markdown(
            """
            <div class="info-card">
                <h3>💡 What PriceProof AI Found</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write(
            st.session_state[
                "ai_explanation"
            ]
        )


    # ========================================================
    # ASK PRICEPROOF AI
    # ========================================================

    st.markdown(
        "### 💬 Ask PriceProof AI"

    )

    st.caption(
        "Ask any question about the analyzed receipt."
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

    ask_question = st.button(
        "💬 Ask AI",
        type="primary"
    )

    if ask_question:

        question = (
            custom_question.strip()
            if custom_question.strip()
            else (
                selected_question
                if selected_question
                != "Choose a question..."
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
                f"""
                <div class="info-card">

                    <strong>👤 You</strong>
                    <p>{chat["question"]}</p>

                    <strong>🤖 PriceProof AI</strong>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.write(
                chat["answer"]
            )


    # ========================================================
    # MANUAL VERIFICATION
    # ========================================================

    st.markdown(
        "### 📝 Manual Verification"
    )

    st.caption(
        "If an item was not matched correctly, you can "
        "review the reference database manually."
    )

    if not reference_df.empty:

        manual_product = st.selectbox(
            "Select a reference product",
            reference_df[
                "product"
            ].tolist()
        )

        selected_reference = reference_df[
            reference_df[
                "product"
            ] == manual_product
        ]

        if not selected_reference.empty:

            selected_row = selected_reference.iloc[0]

            m1, m2, m3 = st.columns(3)

            with m1:

                st.metric(
                    "Reference Price",
                    format_currency(
                        selected_row[
                            "reference_price"
                        ]
                    )
                )

            with m2:

                st.metric(
                    "Brand",
                    selected_row.get(
                        "brand",
                        ""
                    )
                    or "N/A"
                )

            with m3:

                st.metric(
                    "Category",
                    selected_row.get(
                        "category",
                        ""
                    )
                    or "N/A"
                )

    else:

        st.info(
            "Manual verification requires a valid products.csv."
        )


    # ========================================================
    # DOWNLOADS
    # ========================================================

    st.markdown(
        "### 📥 Download Results"
    )

    report_text = generate_consumer_report(
        analysis_df,
        receipt_data
    )

    download1, download2, download3 = st.columns(3)

    with download1:

        st.download_button(
            label="📊 Download CSV",
            data=dataframe_to_csv(
                analysis_df
            ),
            file_name="priceproof_analysis.csv",
            mime="text/csv",
            use_container_width=True
        )

    with download2:

        json_data = json.dumps(
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
            label="🗂️ Download JSON",
            data=json_data,
            file_name="priceproof_analysis.json",
            mime="application/json",
            use_container_width=True
        )

    with download3:

        st.download_button(
            label="📄 Download Report",
            data=report_text,
            file_name="priceproof_consumer_report.txt",
            mime="text/plain",
            use_container_width=True
        )


    # ========================================================
    # REFERENCE DATA DETAILS
    # ========================================================

    with st.expander(
        "📚 View Reference Database"
    ):

        if reference_df.empty:

            st.warning(
                "No reference data available."
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

    st.markdown(
        """
        <div class="status-info">

        <strong>ℹ️ Important Information</strong>

        <br><br>

        PriceProof AI compares receipt prices with reference
        benchmark data. These benchmarks are not guaranteed
        official government or legal prices.

        <br><br>

        A "Potentially High" result is a signal for the consumer
        to investigate further. It does not by itself prove
        illegal overcharging.

        <br><br>

        Consumers should consider product brand, package size,
        quantity, promotions, receipt date and local market
        conditions before reaching a conclusion.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.markdown(
        """
        <div class="info-card">

            <h2>🧾 Start Your Price Check</h2>

            <p>
                Upload a clear shopping receipt above.
                PriceProof AI will extract the products,
                compare available prices with reference
                benchmarks, and identify items that may
                deserve further review.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="info-card">
                <h3>📷 Read Receipt</h3>
                <p>
                    AI vision extracts products,
                    quantities and prices.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="info-card">
                <h3>🔎 Verify Prices</h3>
                <p>
                    Receipt prices are compared
                    against reference benchmarks.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="info-card">
                <h3>💡 Understand Results</h3>
                <p>
                    Get clear explanations and
                    review recommendations.
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
        Detect overpricing. Verify the price. Know your rights.
        <br><br>
        AI-generated analysis should be treated as a verification
        aid, not a legal determination.
    </div>
    """,
    unsafe_allow_html=True
)
