```python
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
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PriceProof AI",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL UI STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL APP BACKGROUND
       ====================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 8% 5%,
                rgba(59, 130, 246, 0.10),
                transparent 25%
            ),
            radial-gradient(
                circle at 92% 8%,
                rgba(16, 185, 129, 0.09),
                transparent 24%
            ),
            radial-gradient(
                circle at 50% 100%,
                rgba(99, 102, 241, 0.07),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #f8fbff 0%,
                #f3f6fb 45%,
                #eef4ff 100%
            );
    }


    /* ======================================================
       MAIN CONTENT
       ====================================================== */

    .main .block-container {
        max-width: 1400px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }


    /* ======================================================
       TYPOGRAPHY
       ====================================================== */

    h1, h2, h3, h4 {
        letter-spacing: -0.025em;
        color: #111827;
    }

    h2 {
        margin-top: 1.5rem;
    }

    h3 {
        margin-top: 1.2rem;
    }

    p {
        color: #475569;
    }


    /* ======================================================
       HERO SECTION
       ====================================================== */

    .hero {
        position: relative;
        overflow: hidden;

        background:
            linear-gradient(
                135deg,
                #0f172a 0%,
                #172554 45%,
                #312e81 100%
            );

        padding: 2.7rem 3rem;
        border-radius: 26px;
        color: white;
        margin-bottom: 1.8rem;

        box-shadow:
            0 18px 45px rgba(15, 23, 42, 0.16);

        border: 1px solid rgba(255,255,255,0.08);
    }

    .hero::before {
        content: "";
        position: absolute;

        width: 280px;
        height: 280px;

        right: -100px;
        top: -120px;

        background: rgba(255,255,255,0.07);
        border-radius: 50%;
    }

    .hero::after {
        content: "";
        position: absolute;

        width: 180px;
        height: 180px;

        right: 160px;
        bottom: -120px;

        background: rgba(255,255,255,0.05);
        border-radius: 50%;
    }

    .hero-content {
        position: relative;
        z-index: 2;
    }

    .hero h1 {
        color: white;
        font-size: 3.15rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0 0 0.45rem 0;
    }

    .hero p {
        color: rgba(255,255,255,0.90);
        font-size: 1.15rem;
        margin: 0;
        max-width: 750px;
    }

    .hero-tag {
        display: inline-flex;
        align-items: center;

        margin-top: 1.1rem;

        padding: 0.48rem 1rem;

        border-radius: 999px;

        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.14);

        color: #f8fafc;
        font-size: 0.88rem;
        font-weight: 600;

        backdrop-filter: blur(8px);
    }


    /* ======================================================
       GLASS / INFORMATION CARDS
       ====================================================== */

    .info-card {
        background: rgba(255,255,255,0.84);

        border: 1px solid rgba(148,163,184,0.18);

        border-radius: 18px;

        padding: 1.35rem;

        box-shadow:
            0 8px 25px rgba(15,23,42,0.055);

        backdrop-filter: blur(10px);
    }


    /* ======================================================
       METRIC CONTAINERS
       ====================================================== */

    div[data-testid="metric-container"] {
        background:
            linear-gradient(
                145deg,
                rgba(255,255,255,0.96),
                rgba(248,250,252,0.90)
            );

        border: 1px solid rgba(148,163,184,0.18);

        border-radius: 18px;

        padding: 1rem 1.1rem;

        min-height: 105px;

        box-shadow:
            0 7px 22px rgba(15,23,42,0.055);

        transition:
            transform 0.18s ease,
            box-shadow 0.18s ease;
    }

    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);

        box-shadow:
            0 12px 28px rgba(15,23,42,0.09);
    }


    /* ======================================================
       CUSTOM METRIC CARD
       ====================================================== */

    .metric-card {
        background:
            linear-gradient(
                145deg,
                rgba(255,255,255,0.96),
                rgba(248,250,252,0.92)
            );

        padding: 1.25rem;

        border-radius: 18px;

        border: 1px solid rgba(148,163,184,0.18);

        box-shadow:
            0 7px 22px rgba(15,23,42,0.055);

        min-height: 120px;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.86rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #111827;
    }


    /* ======================================================
       SECTION LABEL
       ====================================================== */

    .section-label {
        display: inline-flex;
        align-items: center;

        background: rgba(239,246,255,0.9);

        color: #1d4ed8;

        border: 1px solid rgba(59,130,246,0.14);

        padding: 0.35rem 0.75rem;

        border-radius: 999px;

        font-size: 0.78rem;
        font-weight: 700;

        margin-bottom: 0.5rem;
    }


    /* ======================================================
       STATUS CARDS
       ====================================================== */

    .status-good {
        background: rgba(236,253,245,0.92);

        border-left: 5px solid #10b981;

        padding: 1rem 1.2rem;

        border-radius: 14px;

        margin: 0.8rem 0;

        box-shadow:
            0 5px 16px rgba(16,185,129,0.07);
    }

    .status-warning {
        background: rgba(255,251,235,0.94);

        border-left: 5px solid #f59e0b;

        padding: 1rem 1.2rem;

        border-radius: 14px;

        margin: 0.8rem 0;

        box-shadow:
            0 5px 16px rgba(245,158,11,0.07);
    }

    .status-info {
        background: rgba(239,246,255,0.94);

        border-left: 5px solid #3b82f6;

        padding: 1rem 1.2rem;

        border-radius: 14px;

        margin: 0.8rem 0;

        box-shadow:
            0 5px 16px rgba(59,130,246,0.07);
    }


    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton > button,
    .stDownloadButton > button {

        border-radius: 11px;

        min-height: 44px;

        font-weight: 700;

        border: 1px solid rgba(148,163,184,0.20);

        transition:
            transform 0.18s ease,
            box-shadow 0.18s ease,
            border-color 0.18s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {

        transform: translateY(-1px);

        box-shadow:
            0 8px 20px rgba(15,23,42,0.11);

        border-color:
            rgba(59,130,246,0.28);
    }

    .stButton > button[kind="primary"] {

        box-shadow:
            0 7px 18px rgba(37,99,235,0.16);
    }


    /* ======================================================
       FILE UPLOADER
       ====================================================== */

    div[data-testid="stFileUploader"] {

        background:
            linear-gradient(
                145deg,
                rgba(255,255,255,0.90),
                rgba(248,250,252,0.78)
            );

        border: 1px dashed #93a4b8;

        border-radius: 18px;

        padding: 0.7rem;

        box-shadow:
            0 6px 20px rgba(15,23,42,0.045);
    }

    div[data-testid="stFileUploader"]:hover {

        border-color: #3b82f6;

        box-shadow:
            0 8px 24px rgba(59,130,246,0.09);
    }


    /* ======================================================
       EXPANDERS
       ====================================================== */

    div[data-testid="stExpander"] {

        background:
            rgba(255,255,255,0.80);

        border:
            1px solid rgba(148,163,184,0.18);

        border-radius:
            16px;

        box-shadow:
            0 6px 20px rgba(15,23,42,0.045);

        overflow:
            hidden;
    }


    /* ======================================================
       ALERTS
       ====================================================== */

    div[data-testid="stAlert"] {

        border-radius:
            13px;

        border:
            1px solid rgba(148,163,184,0.12);
    }


    /* ======================================================
       DATAFRAME
       ====================================================== */

    div[data-testid="stDataFrame"] {

        border-radius:
            14px;

        overflow:
            hidden;

        border:
            1px solid rgba(148,163,184,0.18);

        box-shadow:
            0 6px 18px rgba(15,23,42,0.04);
    }


    /* ======================================================
       SELECT BOX / INPUTS
       ====================================================== */

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    textarea {

        border-radius:
            11px !important;
    }


    input,
    textarea {

        background:
            rgba(255,255,255,0.86) !important;
    }


    /* ======================================================
       PROGRESS BAR
       ====================================================== */

    div[data-testid="stProgressBar"] {

        margin-top: 0.6rem;
        margin-bottom: 0.8rem;
    }


    /* ======================================================
       IMAGE PREVIEW
       ====================================================== */

    img {

        border-radius:
            14px;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {

        background:
            linear-gradient(
                180deg,
                #0f172a 0%,
                #172554 58%,
                #1e1b4b 100%
            );

        border-right:
            1px solid rgba(255,255,255,0.06);
    }

    section
```
