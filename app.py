import streamlit as st
import pandas as pd
from PIL import Image
from groq import Groq
import base64
import json
import re
import os
import difflib
import io
from datetime import datetime
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PriceProof AI",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL LIGHT STYLING
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(219, 234, 254, 0.65), transparent 35%),
                radial-gradient(circle at top right, rgba(224, 231, 255, 0.55), transparent 30%),
                linear-gradient(180deg, #f8fbff 0%, #ffffff 55%, #f7faff 100%);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f1f6ff 0%, #ffffff 100%);
            border-right: 1px solid #dbe5f1;
        }

        [data-testid="stSidebar"] * {
            color: #1e293b;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        h1, h2, h3 {
            color: #0f172a;
        }

        .stButton > button {
            border-radius: 10px;
            border: 1px solid #dbe5f1;
            font-weight: 600;
            transition: all 0.2s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 5px 15px rgba(15, 23, 42, 0.10);
        }

        [data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.85);
            border: 1px dashed #b8c7dc;
            border-radius: 14px;
            padding: 10px;
        }

        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.90);
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 12px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
        }

        .priceproof-card {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05);
        }

        .small-muted {
            color: #64748b;
            font-size: 0.9rem;
        }

        .success-box {
            padding: 16px;
            border-radius: 12px;
            background: #ecfdf5;
            border: 1px solid #a7f3d0;
            color: #065f46;
        }

        .warning-box {
            padding: 16px;
            border-radius: 12px;
            background: #fffbeb;
            border: 1px solid #fde68a;
            color: #92400e;
        }

        .danger-box {
            padding: 16px;
            border-radius: 12px;
            background: #fef2f2;
            border: 1px solid #fecaca;
            color: #991b1b;
        }

        .info-box {
            padding: 16px;
            border-radius: 12px;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #1e40af;
        }

        div[data-testid="stExpander"] {
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            background: rgba(255,255,255,0.75);
        }

        .stProgress > div > div {
            border-radius: 20px;
        }

        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

VISION_MODEL = "qwen/qwen3.6-27b"
TEXT_MODEL = "openai/gpt-oss-20b"

OVERPRICE_THRESHOLD = 10.0
MAX_FILE_SIZE_MB = 20


# ============================================================
# SESSION STATE
# ============================================================

if "receipt_data" not in st.session_state:
    st.session_state.receipt_data = None

if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = None

if "ai_explanation" not in st.session_state:
    st.session_state.ai_explanation = ""

if "ai_chat_history" not in st.session_state:
    st.session_state.ai_chat_history = []

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = ""

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


# ============================================================
# GROQ CLIENT
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


client = get_groq_client()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(value):
    if value is None:
        return ""

    return str(value).strip()


def normalize_text(text):
    text = clean_text(text).lower()

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        if isinstance(value, (int, float)):
            return float(value)

        cleaned = re.sub(r"[^\d.\-]", "", str(value))

        if cleaned == "":
            return default

        return float(cleaned)

    except Exception:
        return default


def format_currency(value, currency="PKR"):
    value = safe_float(value)

    if currency == "PKR":
        return f"PKR {value:,.2f}"

    return f"{currency} {value:,.2f}"


def image_to_base64(image):
    buffer = io.BytesIO()

    image.save(buffer, format="JPEG", quality=90)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def clean_json_text(text):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text)
        text = re.sub(r"```$", "", text)

    return text.strip()


def extract_json_from_text(text):
    text = clean_json_text(text)

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return None


# ============================================================
# REFERENCE DATABASE
# ============================================================

@st.cache_data
def load_reference_database():
    possible_paths = [
        "products.csv",
        "./products.csv",
        "/mount/src/priceproof-ai/products.csv",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)

                df.columns = [
                    str(col).strip().lower().replace(" ", "_")
                    for col in df.columns
                ]

                return df

            except Exception:
                continue

    return pd.DataFrame()


reference_df = load_reference_database()


# ============================================================
# REFERENCE COLUMN DETECTION
# ============================================================

def find_column(df, candidates):
    if df.empty:
        return None

    normalized_columns = {
        str(col).lower().strip().replace(" ", "_"): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.lower().strip().replace(" ", "_")

        if key in normalized_columns:
            return normalized_columns[key]

    for col in df.columns:
        col_normalized = str(col).lower().replace(" ", "_")

        for candidate in candidates:
            if candidate.lower().replace(" ", "_") in col_normalized:
                return col

    return None


REFERENCE_PRODUCT_COL = find_column(
    reference_df,
    [
        "product",
        "product_name",
        "name",
        "item",
        "item_name",
    ],
)

REFERENCE_PRICE_COL = find_column(
    reference_df,
    [
        "reference_price",
        "reference_price_pkr",
        "market_price",
        "price",
        "average_price",
        "avg_price",
    ],
)

REFERENCE_BRAND_COL = find_column(
    reference_df,
    [
        "brand",
        "brand_name",
    ],
)


# ============================================================
# PRODUCT MATCHING
# ============================================================

def match_product(product_name, brand="", weight=""):
    if reference_df.empty or REFERENCE_PRODUCT_COL is None:
        return None

    product_name = normalize_text(product_name)
    brand = normalize_text(brand)
    weight = normalize_text(weight)

    if not product_name:
        return None

    best_match = None
    best_score = 0

    for _, row in reference_df.iterrows():

        ref_product = normalize_text(
            row.get(REFERENCE_PRODUCT_COL, "")
        )

        ref_brand = ""

        if REFERENCE_BRAND_COL:
            ref_brand = normalize_text(
                row.get(REFERENCE_BRAND_COL, "")
            )

        if not ref_product:
            continue

        product_similarity = difflib.SequenceMatcher(
            None,
            product_name,
            ref_product,
        ).ratio()

        score = product_similarity

        if brand and ref_brand:
            brand_similarity = difflib.SequenceMatcher(
                None,
                brand,
                ref_brand,
            ).ratio()

            score = (
                product_similarity * 0.70
                + brand_similarity * 0.30
            )

        if score > best_score:
            best_score = score
            best_match = row

    if best_match is None:
        return None

    if best_score < 0.45:
        return None

    reference_price = safe_float(
        best_match.get(REFERENCE_PRICE_COL, 0)
        if REFERENCE_PRICE_COL
        else 0
    )

    if reference_price <= 0:
        return None

    return {
        "reference_product": str(
            best_match.get(REFERENCE_PRODUCT_COL, "")
        ),
        "reference_price": reference_price,
        "match_score": round(best_score * 100, 1),
        "reference_brand": (
            str(best_match.get(REFERENCE_BRAND_COL, ""))
            if REFERENCE_BRAND_COL
            else ""
        ),
    }


# ============================================================
# RECEIPT EXTRACTION
# ============================================================

def extract_receipt_with_ai(image):
    if client is None:
        return None, "Groq API key is missing."

    image_base64 = image_to_base64(image)

    prompt = """
You are an expert receipt OCR and structured data extraction assistant.

Read the uploaded receipt carefully.

Return ONLY valid JSON.

Use this exact structure:

{
  "store_name": "",
  "receipt_date": "",
  "currency": "PKR",
  "total": 0,
  "items": [
    {
      "name": "",
      "brand": "",
      "quantity": 1,
      "unit_price": 0,
      "total_price": 0,
      "weight_or_size": ""
    }
  ]
}

Rules:
- Extract every visible purchased item.
- Do not invent products.
- Use 0 when a numeric value cannot be read.
- Keep product names concise.
- Preserve the receipt's currency when visible.
- If the receipt total is visible, extract it.
- Return JSON only.
"""

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            temperature=0,
            max_tokens=4000,
        )

        content = response.choices[0].message.content

        data = extract_json_from_text(content)

        if data is None:
            return None, "The AI response could not be converted into valid JSON."

        return data, None

    except Exception as e:
        return None, f"Receipt extraction failed: {str(e)}"


# ============================================================
# NORMALIZE RECEIPT
# ============================================================

def normalize_receipt_data(data):
    if not isinstance(data, dict):
        return {
            "store_name": "",
            "receipt_date": "",
            "currency": "PKR",
            "total": 0,
            "items": [],
        }

    items = data.get("items", [])

    if not isinstance(items, list):
        items = []

    normalized_items = []

    for item in items:

        if not isinstance(item, dict):
            continue

        name = clean_text(item.get("name"))

        if not name:
            continue

        normalized_items.append(
            {
                "name": name,
                "brand": clean_text(item.get("brand")),
                "quantity": safe_float(
                    item.get("quantity"),
                    1
                ),
                "unit_price": safe_float(
                    item.get("unit_price")
                ),
                "total_price": safe_float(
                    item.get("total_price")
                ),
                "weight_or_size": clean_text(
                    item.get("weight_or_size")
                ),
            }
        )

    return {
        "store_name": clean_text(data.get("store_name")),
        "receipt_date": clean_text(data.get("receipt_date")),
        "currency": clean_text(
            data.get("currency") or "PKR"
        ),
        "total": safe_float(data.get("total")),
        "items": normalized_items,
    }


# ============================================================
# PRICE ANALYSIS
# ============================================================

def analyze_receipt(receipt_data):
    rows = []

    currency = receipt_data.get("currency", "PKR")

    for item in receipt_data.get("items", []):

        product_name = item.get("name", "")
        brand = item.get("brand", "")
        quantity = safe_float(item.get("quantity"), 1)
        unit_price = safe_float(item.get("unit_price"))
        total_price = safe_float(item.get("total_price"))

        if total_price <= 0 and unit_price > 0:
            total_price = unit_price * max(quantity, 1)

        if unit_price <= 0 and total_price > 0:
            unit_price = total_price / max(quantity, 1)

        match = match_product(
            product_name,
            brand,
            item.get("weight_or_size", ""),
        )

        reference_price = None
        difference = None
        percentage = None
        status = "No Reference"

        reference_product = ""

        match_score = 0

        if match:

            reference_price = match["reference_price"]
            reference_product = match["reference_product"]
            match_score = match["match_score"]

            difference = unit_price - reference_price

            percentage = (
                (difference / reference_price) * 100
                if reference_price > 0
                else 0
            )

            if percentage >= OVERPRICE_THRESHOLD and difference > 0:
                status = "Potentially High"

            elif difference > 0:
                status = "Slightly High"

            elif percentage <= -OVERPRICE_THRESHOLD:
                status = "Below Reference"

            else:
                status = "Within Range"

        rows.append(
            {
                "Product": product_name,
                "Brand": brand,
                "Qty": quantity,
                "Receipt Price": unit_price,
                "Reference Price": reference_price,
                "Difference": difference,
                "Difference %": percentage,
                "Status": status,
                "Reference Match": reference_product,
                "Match Score": match_score,
                "Total": total_price,
                "Currency": currency,
                "Weight/Size": item.get("weight_or_size", ""),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# PRICE HEALTH SCORE
# ============================================================

def calculate_price_health_score(df):
    if df is None or df.empty:
        return 100

    comparable = df[df["Reference Price"].notna()].copy()

    if comparable.empty:
        return 100

    penalty = 0

    for _, row in comparable.iterrows():

        pct = safe_float(row["Difference %"])

        if pct >= 10:
            penalty += min(pct * 0.7, 25)

        elif pct > 0:
            penalty += min(pct * 0.2, 5)

    score = 100 - penalty

    return max(0, min(100, round(score)))


# ============================================================
# AI EXPLANATION
# ============================================================

def generate_ai_explanation(df, receipt_data):
    if client is None:
        return "Groq API key is missing."

    if df is None or df.empty:
        return "There are no products available for analysis."

    analysis_records = []

    for _, row in df.iterrows():

        analysis_records.append(
            {
                "product": row["Product"],
                "brand": row["Brand"],
                "receipt_price": row["Receipt Price"],
                "reference_price": row["Reference Price"],
                "difference": row["Difference"],
                "difference_percent": row["Difference %"],
                "status": row["Status"],
            }
        )

    prompt = f"""
You are PriceProof AI, a consumer price verification assistant.

Analyze this receipt comparison.

Receipt:
{json.dumps(receipt_data, indent=2)}

Comparison:
{json.dumps(analysis_records, indent=2)}

Write a concise and useful explanation.

Requirements:
- Clearly identify potentially high-priced items.
- Explain the difference between receipt price and reference price.
- Do not call the reference price an official government/legal price.
- Say that the reference database is a prototype benchmark.
- Mention that prices can vary by city, shop, date, brand, package size, promotions and market conditions.
- Give practical next steps if an item appears potentially overpriced.
- Do not provide unnecessary legal claims.
- Keep the answer easy to understand.
- Use short headings and bullet points.
"""

    try:
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise consumer price verification assistant.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
            max_tokens=1200,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"AI explanation failed: {str(e)}"


# ============================================================
# ASK AI
# ============================================================

def ask_ai(question, receipt_data, df):
    if client is None:
        return "Groq API key is missing."

    comparison = []

    if df is not None and not df.empty:

        for _, row in df.iterrows():

            comparison.append(
                {
                    "product": row["Product"],
                    "brand": row["Brand"],
                    "receipt_price": row["Receipt Price"],
                    "reference_price": row["Reference Price"],
                    "difference": row["Difference"],
                    "difference_percent": row["Difference %"],
                    "status": row["Status"],
                }
            )

    prompt = f"""
You are PriceProof AI.

Answer the user's question about their receipt and price verification.

Receipt:
{json.dumps(receipt_data, indent=2)}

Price comparison:
{json.dumps(comparison, indent=2)}

User question:
{question}

Rules:
- Answer directly.
- Keep it concise.
- Use simple language.
- Base the answer only on the available receipt and comparison information.
- Do not call reference prices official government prices.
- Mention uncertainty when appropriate.
- Do not invent facts.
"""

    try:
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful and concise consumer price verification assistant.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
            max_tokens=800,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"AI response failed: {str(e)}"


# ============================================================
# REPORT GENERATION
# ============================================================

def create_text_report(receipt_data, df, explanation):
    lines = []

    lines.append("PRICEPROOF AI - PRICE VERIFICATION REPORT")
    lines.append("=" * 55)
    lines.append("")

    lines.append(
        f"Store: {receipt_data.get('store_name', '')}"
    )

    lines.append(
        f"Receipt Date: {receipt_data.get('receipt_date', '')}"
    )

    lines.append(
        f"Receipt Total: {format_currency(receipt_data.get('total', 0), receipt_data.get('currency', 'PKR'))}"
    )

    lines.append("")

    lines.append("PRODUCT ANALYSIS")
    lines.append("-" * 55)

    if df is not None and not df.empty:

        for _, row in df.iterrows():

            lines.append(
                f"Product: {row['Product']}"
            )

            lines.append(
                f"Receipt Price: {format_currency(row['Receipt Price'], row['Currency'])}"
            )

            if pd.notna(row["Reference Price"]):

                lines.append(
                    f"Reference Price: {format_currency(row['Reference Price'], row['Currency'])}"
                )

                lines.append(
                    f"Difference: {format_currency(row['Difference'], row['Currency'])}"
                )

                lines.append(
                    f"Difference %: {safe_float(row['Difference %']):.1f}%"
                )

                lines.append(
                    f"Status: {row['Status']}"
                )

            else:

                lines.append(
                    "Reference Price: Not available"
                )

                lines.append(
                    "Status: No Reference"
                )

            lines.append("")

    lines.append("")
    lines.append("AI EXPLANATION")
    lines.append("-" * 55)
    lines.append(explanation)

    lines.append("")
    lines.append("=" * 55)
    lines.append(
        "Disclaimer: Reference prices are prototype benchmark data, "
        "not official government prices or legal determinations."
    )

    return "\n".join(lines)


def create_json_report(receipt_data, df, explanation):
    products = []

    if df is not None and not df.empty:

        for _, row in df.iterrows():

            products.append(
                {
                    "product": row["Product"],
                    "brand": row["Brand"],
                    "quantity": row["Qty"],
                    "receipt_price": row["Receipt Price"],
                    "reference_price": (
                        None
                        if pd.isna(row["Reference Price"])
                        else row["Reference Price"]
                    ),
                    "difference": (
                        None
                        if pd.isna(row["Difference"])
                        else row["Difference"]
                    ),
                    "difference_percent": (
                        None
                        if pd.isna(row["Difference %"])
                        else row["Difference %"]
                    ),
                    "status": row["Status"],
                    "reference_match": row["Reference Match"],
                }
            )

    report = {
        "app": "PriceProof AI",
        "generated_at": datetime.now().isoformat(),
        "receipt": receipt_data,
        "analysis": products,
        "ai_explanation": explanation,
        "disclaimer": (
            "Reference prices are prototype benchmark data, "
            "not official government prices or legal determinations."
        ),
    }

    return json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧾 PriceProof AI")

    st.caption(
        "AI-powered receipt and price verification"
    )

    st.divider()

    st.subheader("What it does")

    st.write(
        "PriceProof AI reads a receipt, extracts products and prices, "
        "compares them with a prototype reference database, and highlights "
        "potentially suspicious overcharging."
    )

    st.divider()

    st.subheader("Verification Flow")

    st.write("1. 📷 Upload receipt")
    st.write("2. 🤖 AI reads receipt")
    st.write("3. 🔎 Match products")
    st.write("4. 📊 Compare prices")
    st.write("5. 🚨 Detect suspicious prices")
    st.write("6. 💡 Explain results")
    st.write("7. 📄 Generate report")

    st.divider()

    st.caption(
        "Prototype benchmark data is used for demonstration. "
        "It is not official government pricing data."
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🧾 PriceProof AI")

st.subheader(
    "Detect overpricing. Verify the price. Know your rights."
)

st.caption(
    "✨ AI-powered receipt & price verification"
)

st.divider()


# ============================================================
# ABOUT
# ============================================================

with st.expander("ℹ️ About PriceProof AI"):

    st.write(
        """
        **PriceProof AI** is an AI-powered consumer price verification
        prototype.

        It helps users understand whether products on a receipt appear
        significantly higher than prices in a reference market database.

        The system does not declare a price to be legally correct or
        officially fixed. It provides a comparison and highlights items
        that may deserve further verification.
        """
    )


# ============================================================
# HOW IT WORKS
# ============================================================

with st.expander("🚀 How PriceProof AI Works"):

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown("### 1️⃣")
        st.write("**Upload Receipt**")
        st.caption("Add a receipt image.")

    with col2:
        st.markdown("### 2️⃣")
        st.write("**AI Reads It**")
        st.caption("AI extracts products and prices.")

    with col3:
        st.markdown("### 3️⃣")
        st.write("**Verify Prices**")
        st.caption("Prices are compared with references.")

    with col4:
        st.markdown("### 4️⃣")
        st.write("**AI Explains**")
        st.caption("Suspicious differences are explained.")

    with col5:
        st.markdown("### 5️⃣")
        st.write("**Generate Report**")
        st.caption("Download verification evidence.")


# ============================================================
# STEP 1 — UPLOAD
# ============================================================

st.caption("STEP 1 • RECEIPT INPUT")

st.header("📷 Upload Your Receipt")

uploaded_file = st.file_uploader(
    "Upload a clear receipt image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Maximum recommended file size: 20 MB.",
)


if uploaded_file is not None:

    file_size_mb = uploaded_file.size / (1024 * 1024)

    if file_size_mb > MAX_FILE_SIZE_MB:

        st.error(
            f"File is too large. Please upload an image smaller than "
            f"{MAX_FILE_SIZE_MB} MB."
        )

    else:

        try:

            image = Image.open(uploaded_file)

            st.session_state.uploaded_file_name = uploaded_file.name

            col1, col2 = st.columns([1, 1])

            with col1:

                st.image(
                    image,
                    caption="Uploaded Receipt",
                    use_container_width=True,
                )

            with col2:

                st.info(
                    "Ready for AI analysis.\n\n"
                    "The AI will read the receipt, identify products "
                    "and prices, then compare them against the reference database."
                )

                analyze_button = st.button(
                    "🔍 Analyze Receipt",
                    type="primary",
                    use_container_width=True,
                )

            if analyze_button:

                if client is None:

                    st.error(
                        "Groq API key is not configured. "
                        "Please add GROQ_API_KEY to Streamlit Secrets."
                    )

                else:

                    with st.spinner(
                        "🤖 AI is reading and analyzing your receipt..."
                    ):

                        receipt_data, extraction_error = (
                            extract_receipt_with_ai(image)
                        )

                    if extraction_error:

                        st.error(extraction_error)

                    elif receipt_data:

                        receipt_data = normalize_receipt_data(
                            receipt_data
                        )

                        st.session_state.receipt_data = receipt_data

                        analysis_df = analyze_receipt(
                            receipt_data
                        )

                        st.session_state.analysis_df = analysis_df

                        with st.spinner(
                            "💡 Generating AI explanation..."
                        ):

                            explanation = generate_ai_explanation(
                                analysis_df,
                                receipt_data,
                            )

                        st.session_state.ai_explanation = explanation
                        st.session_state.analysis_done = True

                        st.success(
                            "Receipt analyzed successfully!"
                        )

        except Exception as e:

            st.error(
                f"Could not read the uploaded image: {str(e)}"
            )


# ============================================================
# RESULTS
# ============================================================

if (
    st.session_state.analysis_done
    and st.session_state.receipt_data is not None
):

    receipt_data = st.session_state.receipt_data
    analysis_df = st.session_state.analysis_df

    st.divider()

    st.caption("STEP 2 • VERIFICATION RESULTS")

    st.header("📊 Verification Results")

    # --------------------------------------------------------
    # RECEIPT SUMMARY
    # --------------------------------------------------------

    st.subheader("🧾 Receipt Summary")

    summary_col1, summary_col2, summary_col3, summary_col4 = (
        st.columns(4)
    )

    with summary_col1:
        st.metric(
            "Store",
            receipt_data.get("store_name") or "Unknown",
        )

    with summary_col2:
        st.metric(
            "Receipt Date",
            receipt_data.get("receipt_date") or "Unknown",
        )

    with summary_col3:
        st.metric(
            "Items",
            len(receipt_data.get("items", [])),
        )

    with summary_col4:
        st.metric(
            "Receipt Total",
            format_currency(
                receipt_data.get("total", 0),
                receipt_data.get("currency", "PKR"),
            ),
        )

    # --------------------------------------------------------
    # PRICE HEALTH
    # --------------------------------------------------------

    score = calculate_price_health_score(
        analysis_df
    )

    st.subheader("💚 Price Health Score")

    score_col1, score_col2 = st.columns([1, 3])

    with score_col1:

        st.metric(
            "Overall Score",
            f"{score}/100",
        )

    with score_col2:

        st.progress(
            score / 100
        )

        if score >= 80:

            st.success(
                "Most comparable prices appear to be within the reference range."
            )

        elif score >= 60:

            st.warning(
                "Some prices deserve additional verification."
            )

        else:

            st.error(
                "Several prices appear significantly higher than the reference benchmarks."
            )

    # --------------------------------------------------------
    # FLAGGED ITEMS
    # --------------------------------------------------------

    st.subheader("🚨 Potentially High-Priced Items")

    if analysis_df is not None and not analysis_df.empty:

        flagged_df = analysis_df[
            analysis_df["Status"] == "Potentially High"
        ].copy()

        if flagged_df.empty:

            st.success(
                "No potentially high-priced items were detected "
                "using the current reference database."
            )

        else:

            for _, row in flagged_df.iterrows():

                difference = safe_float(
                    row["Difference"]
                )

                percentage = safe_float(
                    row["Difference %"]
                )

                st.warning(
                    f"**{row['Product']}** — "
                    f"Receipt: {format_currency(row['Receipt Price'], row['Currency'])} | "
                    f"Reference: {format_currency(row['Reference Price'], row['Currency'])} | "
                    f"Difference: {format_currency(difference, row['Currency'])} "
                    f"({percentage:.1f}% higher)"
                )

    # --------------------------------------------------------
    # AI EXPLANATION
    # --------------------------------------------------------

    st.subheader("🤖 AI Explanation")

    if st.session_state.ai_explanation:

        st.info(
            st.session_state.ai_explanation
        )

    # --------------------------------------------------------
    # EVIDENCE SUMMARY
    # --------------------------------------------------------

    st.subheader("🔎 Evidence Summary")

    if analysis_df is not None and not analysis_df.empty:

        comparable_df = analysis_df[
            analysis_df["Reference Price"].notna()
        ]

        if comparable_df.empty:

            st.info(
                "No products could be matched with the current reference database."
            )

        else:

            evidence_cols = st.columns(3)

            with evidence_cols[0]:

                st.metric(
                    "Comparable Items",
                    len(comparable_df),
                )

            with evidence_cols[1]:

                flagged_count = len(
                    comparable_df[
                        comparable_df["Status"]
                        == "Potentially High"
                    ]
                )

                st.metric(
                    "Potentially High",
                    flagged_count,
                )

            with evidence_cols[2]:

                average_difference = comparable_df[
                    "Difference %"
                ].mean()

                if pd.isna(average_difference):
                    average_difference = 0

                st.metric(
                    "Average Difference",
                    f"{average_difference:.1f}%",
                )

    # --------------------------------------------------------
    # PRODUCT TABLE
    # --------------------------------------------------------

    st.subheader("📋 Product-by-Product Analysis")

    if analysis_df is not None and not analysis_df.empty:

        display_df = analysis_df[
            [
                "Product",
                "Brand",
                "Qty",
                "Receipt Price",
                "Reference Price",
                "Difference",
                "Difference %",
                "Status",
                "Match Score",
            ]
        ].copy()

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # PRICE COMPARISON CHART
    # --------------------------------------------------------

    st.subheader("📈 Price Comparison")

    if analysis_df is not None and not analysis_df.empty:

        chart_df = analysis_df[
            analysis_df["Reference Price"].notna()
        ].copy()

        if not chart_df.empty:

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=chart_df["Product"],
                    y=chart_df["Receipt Price"],
                    name="Receipt Price",
                )
            )

            fig.add_trace(
                go.Bar(
                    x=chart_df["Product"],
                    y=chart_df["Reference Price"],
                    name="Reference Price",
                )
            )

            fig.update_layout(
                barmode="group",
                title="Receipt Price vs Reference Price",
                xaxis_title="Product",
                yaxis_title="Price",
                height=500,
                template="plotly_white",
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=100,
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:

            st.info(
                "A comparison chart will appear when products match the reference database."
            )

    # --------------------------------------------------------
    # RECEIPT TOTAL CONSISTENCY
    # --------------------------------------------------------

    st.subheader("🧮 Receipt Total Check")

    calculated_total = 0

    if analysis_df is not None and not analysis_df.empty:

        calculated_total = analysis_df["Total"].sum()

    receipt_total = safe_float(
        receipt_data.get("total")
    )

    total_difference = calculated_total - receipt_total

    total_col1, total_col2, total_col3 = st.columns(3)

    with total_col1:

        st.metric(
            "Receipt Total",
            format_currency(
                receipt_total,
                receipt_data.get("currency", "PKR"),
            ),
        )

    with total_col2:

        st.metric(
            "Calculated Items Total",
            format_currency(
                calculated_total,
                receipt_data.get("currency", "PKR"),
            ),
        )

    with total_col3:

        st.metric(
            "Difference",
            format_currency(
                total_difference,
                receipt_data.get("currency", "PKR"),
            ),
        )

    if receipt_total > 0:

        if abs(total_difference) <= max(5, receipt_total * 0.02):

            st.success(
                "The item totals are reasonably consistent with the receipt total."
            )

        else:

            st.warning(
                "The extracted item totals differ from the receipt total. "
                "The receipt may contain tax, discounts, rounding, or extraction errors."
            )

    # --------------------------------------------------------
    # ASK AI
    # --------------------------------------------------------

    st.subheader("💬 Ask PriceProof AI")

    st.write(
        "Ask a question about this receipt or the price comparison."
    )

    user_question = st.text_input(
        "Your question",
        placeholder="Example: Which item should I verify first?",
        key="priceproof_question",
    )

    ask_button = st.button(
        "🤖 Ask AI",
        type="primary",
    )

    if ask_button:

        if not user_question.strip():

            st.warning(
                "Please enter a question first."
            )

        else:

            with st.spinner("Thinking..."):

                answer = ask_ai(
                    user_question,
                    receipt_data,
                    analysis_df,
                )

            st.session_state.ai_chat_history.append(
                {
                    "question": user_question,
                    "answer": answer,
                }
            )

    if st.session_state.ai_chat_history:

        st.markdown("### Previous Questions")

        for chat in reversed(
            st.session_state.ai_chat_history
        ):

            with st.expander(
                f"❓ {chat['question']}"
            ):

                st.write(
                    chat["answer"]
                )

    # --------------------------------------------------------
    # MANUAL VERIFICATION
    # --------------------------------------------------------

    st.subheader("✍️ Manual Verification")

    st.write(
        "If the AI could not confidently match an item, "
        "you can manually verify it against your own market information."
    )

    if analysis_df is not None and not analysis_df.empty:

        manual_product = st.selectbox(
            "Select a product",
            analysis_df["Product"].tolist(),
        )

        selected_row = analysis_df[
            analysis_df["Product"]
            == manual_product
        ].iloc[0]

        manual_col1, manual_col2 = st.columns(2)

        with manual_col1:

            manual_price = st.number_input(
                "Your verified market price",
                min_value=0.0,
                value=float(
                    selected_row["Reference Price"]
                    if pd.notna(
                        selected_row["Reference Price"]
                    )
                    else 0.0
                ),
                step=1.0,
            )

        with manual_col2:

            receipt_price = safe_float(
                selected_row["Receipt Price"]
            )

            manual_difference = (
                receipt_price - manual_price
            )

            manual_percentage = (
                manual_difference
                / manual_price
                * 100
                if manual_price > 0
                else 0
            )

            st.metric(
                "Manual Difference",
                f"{manual_percentage:.1f}%",
            )

        if manual_price > 0:

            if manual_percentage >= OVERPRICE_THRESHOLD:

                st.warning(
                    f"This product is {manual_percentage:.1f}% "
                    f"higher than your manually entered reference price."
                )

            elif manual_percentage > 0:

                st.info(
                    f"This product is {manual_percentage:.1f}% "
                    f"higher than your manually entered reference price."
                )

            else:

                st.success(
                    "The receipt price is not higher than your manually entered reference price."
                )

    # --------------------------------------------------------
    # DOWNLOAD REPORTS
    # --------------------------------------------------------

    st.subheader("📄 Generate Verification Report")

    text_report = create_text_report(
        receipt_data,
        analysis_df,
        st.session_state.ai_explanation,
    )

    json_report = create_json_report(
        receipt_data,
        analysis_df,
        st.session_state.ai_explanation,
    )

    download_col1, download_col2, download_col3 = st.columns(3)

    with download_col1:

        st.download_button(
            "📄 Download Text Report",
            data=text_report,
            file_name="priceproof_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with download_col2:

        st.download_button(
            "🧾 Download JSON Report",
            data=json_report,
            file_name="priceproof_report.json",
            mime="application/json",
            use_container_width=True,
        )

    with download_col3:

        if analysis_df is not None:

            csv_data = analysis_df.to_csv(
                index=False
            )

            st.download_button(
                "📊 Download CSV Analysis",
                data=csv_data,
                file_name="priceproof_analysis.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # --------------------------------------------------------
    # REFERENCE DATABASE
    # --------------------------------------------------------

    with st.expander("📚 View Reference Database"):

        if reference_df.empty:

            st.warning(
                "products.csv was not found or could not be loaded."
            )

        else:

            st.write(
                f"Reference database contains "
                f"**{len(reference_df)} products**."
            )

            st.dataframe(
                reference_df,
                use_container_width=True,
                hide_index=True,
            )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.divider()

    st.warning(
        """
        **Important Disclaimer**

        PriceProof AI uses a prototype reference-price database for
        demonstration and comparison purposes.

        Reference prices are **not official government prices and are not
        legal determinations**. Actual prices may vary based on city,
        shop, date, brand, package size, promotions, taxes, and market
        conditions.

        Always verify important pricing concerns with reliable local
        sources and keep your original receipt as evidence.
        """
    )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.divider()

    st.info(
        """
        ### 👋 Ready for Analysis

        Upload a clear receipt image above to begin.

        PriceProof AI will:

        **📷 Read your receipt → 🔎 Extract products → 📊 Compare prices → 🚨 Detect potentially high prices → 🤖 Explain the results → 📄 Generate a report**
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🧾 PriceProof AI • Detect overpricing. Verify the price. Know your rights."
)

st.caption(
    "AI-generated analysis is a verification aid, not a legal determination."
)
