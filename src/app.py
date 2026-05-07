from __future__ import annotations

import base64
import hashlib
import html
import io
import math
import os
import re
import textwrap
from collections import Counter
from pathlib import Path
from zipfile import BadZipFile, ZipFile
import nltk
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
from nltk.sentiment import SentimentIntensityAnalyzer
from pypdf import PdfReader

# Optional libraries
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from PIL import Image, ImageOps, ImageFilter
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

TESSERACT_AVAILABLE = False


# ----------------------------
# Setup
# ----------------------------
def ensure_vader_lexicon() -> None:
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


def configure_tesseract() -> None:
    global TESSERACT_AVAILABLE

    if not OCR_AVAILABLE:
        TESSERACT_AVAILABLE = False
        return

    candidates = [
        os.environ.get("TESSERACT_CMD", ""),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            break

    try:
        pytesseract.get_tesseract_version()
        TESSERACT_AVAILABLE = True
    except Exception:
        TESSERACT_AVAILABLE = False


ensure_vader_lexicon()
configure_tesseract()
sia = SentimentIntensityAnalyzer()

st.set_page_config(page_title="Narrative Emotion Analyzer", layout="wide")

APP_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = APP_ROOT / "assets"
HF_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
EMOTION_VALENCE = {
    "joy": 0.82,
    "surprise": 0.18,
    "neutral": 0.0,
    "sadness": -0.66,
    "disgust": -0.62,
    "anger": -0.72,
    "fear": -0.78,
    "wonder": 0.36,
    "mixed": 0.0,
}

STORY_SHAPES = {
    "Rags to Riches": {
        "points": [-1.0, -0.65, -0.25, 0.2, 0.62, 1.0],
        "description": "A steady emotional rise from difficulty into improvement.",
    },
    "Tragedy": {
        "points": [1.0, 0.62, 0.2, -0.25, -0.65, -1.0],
        "description": "A steady emotional fall from stability into loss or defeat.",
    },
    "Icarus": {
        "points": [-0.75, 0.25, 1.0, 0.35, -0.35, -1.0],
        "description": "A rise followed by a sharp fall.",
    },
    "Cinderella": {
        "points": [-0.85, 0.55, -0.25, 0.1, 0.55, 1.0],
        "description": "A rise, setback, and stronger recovery.",
    },
    "Oedipus": {
        "points": [0.75, -0.55, 0.35, 0.1, -0.45, -1.0],
        "description": "A fall, partial recovery, and final decline.",
    },
    "Man in a Hole": {
        "points": [0.3, -0.85, -0.5, 0.05, 0.58, 0.95],
        "description": "A drop into trouble followed by recovery.",
    },
}


@st.cache_data(show_spinner=False)
def image_data_uri(file_name: str) -> str:
    path = ASSET_DIR / file_name
    if not path.exists():
        return ""

    mime_type = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


@st.cache_resource(show_spinner=False)
def load_emotion_classifier():
    try:
        from transformers import pipeline
    except Exception:
        return None

    try:
        return pipeline(
            "text-classification",
            model=HF_EMOTION_MODEL,
            top_k=None,
            device=-1,
        )
    except Exception:
        return None


# ----------------------------
# Styling
# ----------------------------
st.markdown("""
<style>
:root {
    --ink: #091425;
    --muted: #475569;
    --panel: rgba(255, 255, 255, 0.94);
    --panel-strong: #ffffff;
    --line: rgba(9, 20, 37, 0.14);
    --teal: #14b8a6;
    --coral: #f9735b;
    --gold: #f5b84b;
    --green: #55b86f;
    --navy: #07111f;
    --shadow: 0 26px 80px rgba(4, 12, 24, 0.22);
}

.stApp {
    background:
        linear-gradient(135deg, #07111f 0%, #102b36 34%, #f6f1e8 70%, #e9f8f5 100%);
    color: var(--ink);
    font-family: "Inter", "Segoe UI", Arial, sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        linear-gradient(180deg, rgba(7, 17, 31, 0.96) 0%, rgba(10, 29, 42, 0.82) 23%, rgba(245, 248, 244, 0.93) 55%, rgba(235, 248, 244, 0.96) 100%),
        repeating-linear-gradient(90deg, rgba(255, 255, 255, 0.06) 0 1px, transparent 1px 86px),
        repeating-linear-gradient(0deg, rgba(20, 184, 166, 0.08) 0 1px, transparent 1px 86px),
        linear-gradient(135deg, #07111f 0%, #10313d 42%, #f7efe5 100%) !important;
}

[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background:
        linear-gradient(120deg, transparent 0 11%, rgba(20, 184, 166, 0.14) 11.2% 11.6%, transparent 11.8% 100%),
        linear-gradient(134deg, transparent 0 41%, rgba(249, 115, 91, 0.13) 41.2% 41.7%, transparent 41.9% 100%),
        linear-gradient(151deg, transparent 0 70%, rgba(245, 184, 75, 0.12) 70.2% 70.7%, transparent 70.9% 100%);
    mix-blend-mode: screen;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    background: transparent !important;
    visibility: hidden;
    height: 0;
}

#MainMenu, footer {
    visibility: hidden;
}

.block-container {
    max-width: 1220px;
    padding: 2.25rem 2rem 4rem 2rem;
    position: relative;
    z-index: 1;
}

h1, h2, h3 {
    color: var(--ink) !important;
    font-family: "Inter", "Segoe UI", Arial, sans-serif;
    letter-spacing: 0;
}

p, label, div, span {
    color: var(--ink);
}

.hero {
    position: relative;
    min-height: 430px;
    border-radius: 30px;
    overflow: hidden;
    background-size: cover;
    background-position: 58% center;
    border: 1px solid rgba(255, 255, 255, 0.16);
    box-shadow: 0 34px 100px rgba(3, 10, 20, 0.42);
    isolation: isolate;
    display: flex;
    align-items: flex-end;
    margin-bottom: 2.15rem;
    animation: fadeUp 700ms ease both;
}

.hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
        radial-gradient(circle at 21% 59%, rgba(20, 184, 166, 0.2), transparent 18rem),
        linear-gradient(90deg, rgba(1, 6, 16, 0.98) 0%, rgba(3, 11, 23, 0.96) 35%, rgba(3, 11, 23, 0.58) 64%, rgba(3, 11, 23, 0.2) 100%),
        linear-gradient(180deg, rgba(7, 17, 31, 0.12) 0%, rgba(7, 17, 31, 0.78) 100%);
    z-index: 0;
    pointer-events: none;
    animation: atmosphereShift 12s ease-in-out infinite alternate;
}

.hero::after {
    content: "";
    position: absolute;
    inset: 1.2rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    pointer-events: none;
    z-index: 1;
}

.hero-content {
    position: relative;
    z-index: 2;
    max-width: 680px;
    padding: 3.25rem;
}

.hero-content::before {
    content: "";
    position: absolute;
    inset: 2rem 1.7rem 1.8rem 1.7rem;
    z-index: -1;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(2, 8, 18, 0.74), rgba(3, 17, 29, 0.42));
    box-shadow: 0 24px 70px rgba(0, 0, 0, 0.32);
    backdrop-filter: none;
}

.hero-kicker,
.section-kicker,
.result-kicker {
    color: #21e6d1;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.hero h1,
.hero .hero-title {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: clamp(2.7rem, 7vw, 5.6rem);
    line-height: 0.95;
    margin: 0.45rem 0 1rem 0;
    max-width: 10ch;
    opacity: 1 !important;
    text-shadow: 0 5px 28px rgba(0, 0, 0, 0.82), 0 0 24px rgba(20, 184, 166, 0.18);
}

.hero h1 *,
.hero .hero-title *,
.hero-content *,
.hero-content span {
    opacity: 1 !important;
}

.hero p,
.hero .hero-copy {
    color: rgba(255, 255, 255, 0.94) !important;
    -webkit-text-fill-color: rgba(255, 255, 255, 0.94) !important;
    font-size: 1.1rem;
    line-height: 1.65;
    max-width: 570px;
    margin-bottom: 1.35rem;
    text-shadow: 0 2px 18px rgba(0, 0, 0, 0.48);
}

.hero-chips,
.micro-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.7rem;
}

.hero-chip {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.34);
    background: rgba(255, 255, 255, 0.17);
    backdrop-filter: none;
    border-radius: 999px;
    padding: 0.55rem 0.85rem;
    font-weight: 750;
    font-size: 0.88rem;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.18), 0 10px 24px rgba(0, 0, 0, 0.22);
    animation: chipFloat 4.8s ease-in-out infinite;
}

.hero-chip:nth-child(2) {
    animation-delay: 350ms;
}

.hero-chip:nth-child(3) {
    animation-delay: 700ms;
}

.section-heading {
    display: flex;
    align-items: end;
    justify-content: space-between;
    gap: 1rem;
    margin: 2.25rem 0 1.1rem;
    padding: 1.35rem 1.45rem;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.86), rgba(237, 250, 247, 0.78));
    border: 1px solid rgba(255, 255, 255, 0.72);
    border-radius: 22px;
    box-shadow: 0 20px 55px rgba(4, 12, 24, 0.12);
    backdrop-filter: none;
}

.section-heading h2 {
    margin: 0.2rem 0 0 0;
    font-size: clamp(1.7rem, 3vw, 2.55rem);
}

.section-heading p {
    max-width: 520px;
    color: var(--muted);
    margin: 0;
    line-height: 1.55;
}

.visual-grid,
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1.15rem;
    margin: 1.5rem 0 2.25rem;
}

.visual-tile,
.metric-card,
.peak-card,
.summary-card,
.input-info-card {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(248, 252, 251, 0.94));
    border: 1px solid rgba(255, 255, 255, 0.76);
    border-radius: 20px;
    box-shadow: 0 22px 58px rgba(4, 12, 24, 0.15);
}

.section-heading,
.visual-tile,
.input-info-card,
.metric-card,
.peak-card,
.summary-card,
.stPlotlyChart,
[data-testid="stDataFrame"] {
    transform-origin: center bottom;
    animation: revealUp 780ms cubic-bezier(0.2, 0.75, 0.2, 1) both;
}

.metric-card:nth-child(2),
.visual-tile:nth-child(2) {
    animation-delay: 90ms;
}

.metric-card:nth-child(3),
.visual-tile:nth-child(3) {
    animation-delay: 180ms;
}

.metric-card:nth-child(4) {
    animation-delay: 270ms;
}

.visual-tile {
    overflow: hidden;
    outline: 1px solid rgba(9, 20, 37, 0.08);
    transition: transform 320ms ease, box-shadow 320ms ease, border-color 320ms ease;
}

.visual-tile:nth-child(2) {
    animation-delay: 120ms;
}

.visual-tile:nth-child(3) {
    animation-delay: 240ms;
}

.visual-tile img {
    width: 100%;
    aspect-ratio: 16 / 10;
    object-fit: cover;
    display: block;
    transition: transform 420ms ease;
}

.visual-tile:hover img {
    transform: scale(1.04);
}

.visual-tile:hover,
.metric-card:hover,
.peak-card:hover,
.summary-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 30px 72px rgba(4, 12, 24, 0.22);
    border-color: rgba(20, 184, 166, 0.34);
}

.visual-tile figcaption {
    padding: 1.05rem 1.1rem 1.2rem;
}

.visual-tile strong,
.metric-card strong {
    display: block;
    color: var(--ink);
    font-size: 1.02rem;
}

.visual-tile span,
.metric-card span {
    display: block;
    color: #475569;
    font-size: 0.88rem;
    margin-top: 0.25rem;
}

.stRadio > div {
    gap: 0.65rem;
}

[data-testid="stFileUploader"],
.stTextArea textarea,
[data-testid="stDataFrame"] {
    background: rgba(255, 255, 255, 0.98) !important;
    border: 1px solid rgba(9, 20, 37, 0.15) !important;
    border-radius: 18px !important;
    box-shadow: 0 18px 50px rgba(4, 12, 24, 0.14);
}

.stTextArea textarea {
    color: #07111f !important;
    -webkit-text-fill-color: #07111f !important;
    font-family: "Inter", "Segoe UI", Arial, sans-serif !important;
    min-height: 260px;
    font-size: 1rem !important;
    line-height: 1.55 !important;
}

.stTextArea textarea::placeholder {
    color: #64748b !important;
    -webkit-text-fill-color: #64748b !important;
    opacity: 1 !important;
}

.stTextArea textarea:focus,
[data-testid="stFileUploader"]:focus-within {
    border-color: rgba(20, 184, 166, 0.55) !important;
    box-shadow: 0 0 0 4px rgba(20, 184, 166, 0.13), 0 16px 40px rgba(21, 32, 51, 0.08);
}

div.stButton > button {
    background: linear-gradient(135deg, #19c7b5, #0f8f83) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem 1.2rem !important;
    font-weight: 850 !important;
    box-shadow: 0 16px 34px rgba(20, 184, 166, 0.28);
    position: relative;
    overflow: hidden;
    transition: transform 180ms ease, box-shadow 180ms ease, filter 180ms ease;
}

[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"],
[data-testid="stDownloadButton"] button {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

[data-testid="stBaseButton-secondary"] *,
[data-testid="stBaseButton-primary"] *,
[data-testid="stDownloadButton"] button *,
[data-testid="stDownloadButton"] button p,
[data-testid="stDownloadButton"] button span {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    opacity: 1 !important;
}

div.stButton > button::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(115deg, transparent 0%, rgba(255, 255, 255, 0.3) 42%, transparent 62%);
    transform: translateX(-125%);
    transition: transform 620ms ease;
}

div.stButton > button:hover {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    transform: translateY(-2px);
    filter: saturate(1.05);
    box-shadow: 0 20px 42px rgba(20, 184, 166, 0.34);
}

div.stButton > button:hover::before {
    transform: translateX(125%);
}

div.stButton > button *,
div.stButton > button p,
div.stButton > button span {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    opacity: 1 !important;
    position: relative;
    z-index: 1;
    font-weight: 850 !important;
}

div.stButton > button:active {
    transform: translateY(0);
}

[data-testid="stFileUploader"] {
    padding: 0.7rem;
}

[data-testid="stFileUploader"] section {
    border: 1px dashed rgba(20, 184, 166, 0.56);
    background: linear-gradient(135deg, rgba(20, 184, 166, 0.08), rgba(245, 184, 75, 0.08));
    border-radius: 14px;
}

[data-testid="stFileUploader"] button {
    background: #07111f !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 10px !important;
    box-shadow: 0 10px 24px rgba(4, 12, 24, 0.18) !important;
}

[data-testid="stFileUploader"] button:hover {
    background: #0f2438 !important;
    box-shadow: 0 12px 28px rgba(4, 12, 24, 0.26) !important;
}

[data-testid="stFileUploader"] button *,
[data-testid="stFileUploader"] button span,
[data-testid="stFileUploader"] button p,
[data-testid="stFileUploader"] button svg {
    color: #ffffff !important;
    fill: #ffffff !important;
    stroke: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    opacity: 1 !important;
}

[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small {
    color: #334155 !important;
    -webkit-text-fill-color: #334155 !important;
    opacity: 1 !important;
}

[data-testid="stRadio"] label,
[data-testid="stFileUploader"] label,
[data-testid="stTextArea"] label,
[data-testid="stCheckbox"] label {
    font-weight: 750;
    color: var(--ink) !important;
}

[data-testid="stRadio"] p,
[data-testid="stRadio"] span,
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] p,
[data-testid="stCheckbox"] span,
[data-testid="stCheckbox"] label {
    color: #07111f !important;
    -webkit-text-fill-color: #07111f !important;
    opacity: 1 !important;
}

[data-testid="stCheckbox"] {
    background: rgba(255, 255, 255, 0.96);
    border: 1px solid rgba(9, 20, 37, 0.15);
    border-radius: 14px;
    padding: 0.6rem 0.75rem;
    box-shadow: 0 10px 28px rgba(4, 12, 24, 0.08);
}

[data-testid="stExpander"] {
    border: 1px solid var(--line);
    border-radius: 16px;
    box-shadow: 0 14px 34px rgba(21, 32, 51, 0.07);
    overflow: hidden;
}

[data-testid="stExpander"] details,
[data-testid="stExpander"] details summary {
    background: #07111f !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] details > summary,
[data-testid="stExpander"] [role="button"] {
    background: #07111f !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border-radius: 14px 14px 0 0;
}

[data-testid="stExpander"] summary div,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary label,
[data-testid="stExpander"] summary [data-testid="stMarkdownContainer"],
[data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] *,
[data-testid="stExpander"] details > summary *,
[data-testid="stExpander"] [role="button"] *,
[data-testid="stExpander"] details > summary p,
[data-testid="stExpander"] details > summary span {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    opacity: 1 !important;
    font-weight: 750 !important;
}

[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] details > summary svg,
[data-testid="stExpander"] [role="button"] svg {
    color: #ffffff !important;
    fill: #ffffff !important;
    stroke: #ffffff !important;
    opacity: 1 !important;
}

[data-testid="stExpander"] details[open] > summary {
    border-bottom: 1px solid rgba(255, 255, 255, 0.16);
}

[data-testid="stExpander"] details > div {
    background: rgba(240, 253, 250, 0.98) !important;
}

[data-testid="stExpander"] [data-testid="stExpanderDetails"],
[data-testid="stExpander"] [data-testid="stExpanderDetails"] > div {
    background: rgba(240, 253, 250, 0.98) !important;
    color: #07111f !important;
    -webkit-text-fill-color: #07111f !important;
}

.input-info-card,
.peak-card,
.summary-card {
    padding: 1.15rem 1.25rem;
    margin: 1rem 0;
}

.input-info-card {
    border-left: 5px solid var(--gold);
}

.metric-grid {
    grid-template-columns: repeat(4, minmax(0, 1fr));
    margin-top: 0.9rem;
}

.metric-card {
    position: relative;
    overflow: hidden;
    padding: 1rem;
    transition: transform 320ms ease, box-shadow 320ms ease, border-color 320ms ease;
}

.metric-card::after {
    content: "";
    position: absolute;
    inset: auto 1rem 0.8rem 1rem;
    height: 3px;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--accent, var(--teal)), transparent);
    transform-origin: left;
    animation: lineGrow 900ms ease both 240ms;
}

.metric-card::before {
    content: "";
    position: absolute;
    inset: 0 auto 0 0;
    width: 5px;
    background: var(--accent, var(--teal));
}

.metric-card strong {
    font-size: clamp(1.45rem, 3vw, 2.15rem);
    margin-top: 0.2rem;
}

.result-kicker {
    margin-top: 1.65rem;
}

.peak-card {
    border-left: 5px solid var(--coral);
}

.summary-card {
    border-left: 5px solid var(--teal);
}

.peak-card h3,
.summary-card h3,
.input-info-card h3 {
    margin: 0 0 0.5rem 0;
    font-size: 1.05rem;
}

.peak-card p,
.summary-card p,
.input-info-card p {
    margin: 0.25rem 0;
    color: var(--muted);
    line-height: 1.62;
    overflow-wrap: anywhere;
}

.peak-card b,
.input-info-card b {
    color: var(--ink);
}

.chart-title {
    margin-top: 1.5rem;
    font-size: 1.25rem !important;
}

.stPlotlyChart {
    background: rgba(255, 255, 255, 0.96);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 0.65rem;
    box-shadow: 0 18px 44px rgba(21, 32, 51, 0.08);
    position: relative;
    overflow: hidden;
}

.stPlotlyChart::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: inherit;
    pointer-events: none;
    background: linear-gradient(90deg, rgba(20, 184, 166, 0.16), transparent 28%, rgba(249, 115, 91, 0.14));
    opacity: 0;
    animation: chartGlow 1600ms ease 320ms both;
}

.section-report-panel {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(239, 252, 249, 0.96));
    border: 1px solid rgba(20, 184, 166, 0.26);
    border-radius: 22px;
    box-shadow: 0 28px 80px rgba(4, 12, 24, 0.2);
    padding: 1.2rem;
    margin: 1rem 0 1.1rem;
    animation: none !important;
    opacity: 1 !important;
    transform: none !important;
    filter: none !important;
}

.section-report-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 1rem;
}

.section-report-title h3 {
    margin: 0;
    color: #07111f !important;
    font-size: 1.25rem;
}

.section-report-title span {
    color: #0f766e !important;
    font-size: 0.78rem;
    font-weight: 850;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.summary-report-card {
    background: #ffffff;
    border: 1px solid rgba(9, 20, 37, 0.12);
    border-left: 6px solid #14b8a6;
    border-radius: 18px;
    box-shadow: 0 18px 44px rgba(4, 12, 24, 0.1);
    padding: 1rem;
    margin-bottom: 1rem;
}

.summary-report-header {
    display: grid;
    grid-template-columns: 1.2fr repeat(3, minmax(8rem, 0.8fr));
    gap: 0.75rem;
    margin-bottom: 0.9rem;
}

.summary-report-header div {
    background: #f8fafc;
    border: 1px solid rgba(9, 20, 37, 0.1);
    border-radius: 14px;
    padding: 0.75rem 0.85rem;
}

.summary-report-header span,
.summary-copy-grid span {
    display: block;
    color: #64748b !important;
    font-size: 0.72rem;
    font-weight: 850;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.summary-report-header strong {
    display: block;
    color: #07111f !important;
    font-size: 1rem;
    margin-top: 0.18rem;
    overflow-wrap: anywhere;
}

.summary-copy-grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 0.8rem;
}

.summary-copy-grid section {
    background: linear-gradient(180deg, #ffffff, #f8fafc);
    border: 1px solid rgba(9, 20, 37, 0.1);
    border-radius: 14px;
    padding: 0.85rem;
}

.summary-copy-grid p {
    color: #0f172a !important;
    line-height: 1.65;
    margin: 0.4rem 0 0;
    overflow-wrap: anywhere;
}

.summary-table-wrap {
    overflow-x: auto;
    border: 1px solid rgba(20, 184, 166, 0.28);
    border-radius: 16px;
    background: #f0fdfa;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.9);
}

.summary-table {
    width: 100%;
    min-width: 980px;
    border-collapse: separate;
    border-spacing: 0;
    table-layout: fixed;
}

.summary-table th {
    background: #ccfbf1;
    color: #063b35 !important;
    padding: 0.85rem;
    text-align: left;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    position: sticky;
    top: 0;
    z-index: 1;
}

.summary-table td {
    background: #ffffff;
    color: #0f172a !important;
    border-top: 1px solid rgba(20, 184, 166, 0.18);
    padding: 0.85rem;
    vertical-align: top;
    line-height: 1.58;
    white-space: normal;
    overflow-wrap: anywhere;
}

.summary-table tbody tr:nth-child(even) td {
    background: #ecfdf5;
}

.summary-table .number-cell {
    color: #0f766e !important;
    font-weight: 850;
}

.summary-table .emotion-cell {
    color: #b45309 !important;
    font-weight: 850;
}

.summary-table .long-cell {
    width: 28%;
}

.summary-table-note {
    color: #475569 !important;
    font-size: 0.88rem;
    margin: 0.85rem 0 0;
}

.summary-row-list {
    display: grid;
    gap: 0.85rem;
    margin-top: 1rem;
}

.summary-row-card {
    display: grid;
    grid-template-columns: 12rem minmax(0, 1fr);
    gap: 0.85rem;
    background: #ffffff;
    border: 1px solid rgba(20, 184, 166, 0.22);
    border-left: 6px solid #2dd4bf;
    border-radius: 16px;
    padding: 0.95rem;
    box-shadow: 0 14px 34px rgba(4, 12, 24, 0.1);
}

.summary-row-meta {
    background: #ecfdf5;
    border: 1px solid rgba(20, 184, 166, 0.2);
    border-radius: 13px;
    padding: 0.8rem;
}

.summary-row-meta span,
.summary-row-body span {
    display: block;
    color: #0f766e !important;
    font-size: 0.72rem;
    font-weight: 850;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.summary-row-meta strong {
    display: block;
    color: #07111f !important;
    margin: 0.25rem 0 0.65rem;
}

.summary-row-body {
    display: grid;
    gap: 0.8rem;
}

.summary-row-body section {
    background: linear-gradient(180deg, #ffffff, #f8fafc);
    border: 1px solid rgba(9, 20, 37, 0.08);
    border-radius: 13px;
    padding: 0.85rem;
}

.summary-row-body p {
    color: #0f172a !important;
    line-height: 1.68;
    margin: 0.35rem 0 0;
    overflow-wrap: anywhere;
}

.paragraph-report-panel {
    background: linear-gradient(180deg, #f0fdfa, #ffffff);
    border: 1px solid rgba(20, 184, 166, 0.28);
    border-radius: 18px;
    box-shadow: 0 18px 48px rgba(4, 12, 24, 0.12);
    padding: 1rem;
    margin: 0.7rem 0 1rem;
}

.paragraph-report-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.9rem;
}

.paragraph-report-title h3 {
    color: #07111f !important;
    margin: 0;
    font-size: 1.12rem;
}

.paragraph-report-title span {
    color: #0f766e !important;
    font-size: 0.72rem;
    font-weight: 850;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.paragraph-row-card {
    background: #ffffff;
    border: 1px solid rgba(9, 20, 37, 0.1);
    border-left: 5px solid #14b8a6;
    border-radius: 15px;
    box-shadow: 0 12px 30px rgba(4, 12, 24, 0.09);
    padding: 0.9rem;
    margin-bottom: 0.8rem;
}

.paragraph-row-meta {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 0.65rem;
    margin-bottom: 0.75rem;
}

.paragraph-row-meta div {
    background: #ecfdf5;
    border: 1px solid rgba(20, 184, 166, 0.18);
    border-radius: 12px;
    padding: 0.65rem;
}

.paragraph-row-meta span,
.paragraph-row-text span {
    display: block;
    color: #0f766e !important;
    font-size: 0.68rem;
    font-weight: 850;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.paragraph-row-meta strong {
    display: block;
    color: #07111f !important;
    margin-top: 0.2rem;
    overflow-wrap: anywhere;
}

.paragraph-row-text {
    background: #f8fafc;
    border: 1px solid rgba(9, 20, 37, 0.08);
    border-radius: 12px;
    padding: 0.8rem;
}

.paragraph-row-text p {
    color: #0f172a !important;
    line-height: 1.68;
    margin: 0.35rem 0 0;
    overflow-wrap: anywhere;
}

.story-shape-card {
    background: linear-gradient(135deg, #ffffff, #ecfdf5);
    border: 1px solid rgba(20, 184, 166, 0.28);
    border-left: 6px solid #14b8a6;
    border-radius: 18px;
    box-shadow: 0 18px 48px rgba(4, 12, 24, 0.12);
    padding: 1rem;
    margin: 1rem 0;
}

.story-shape-card h3 {
    color: #07111f !important;
    margin: 0 0 0.35rem;
    font-size: 1.2rem;
}

.story-shape-card p {
    color: #1e293b !important;
    line-height: 1.62;
    margin: 0.25rem 0;
}

.story-shape-meta {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.65rem;
    margin: 0.85rem 0;
}

.story-shape-meta div {
    background: #ffffff;
    border: 1px solid rgba(20, 184, 166, 0.2);
    border-radius: 12px;
    padding: 0.72rem;
}

.story-shape-meta span,
.story-shape-alternatives span {
    display: block;
    color: #0f766e !important;
    font-size: 0.68rem;
    font-weight: 850;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.story-shape-meta strong {
    display: block;
    color: #07111f !important;
    margin-top: 0.2rem;
    overflow-wrap: anywhere;
}

.story-shape-alternatives {
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid rgba(9, 20, 37, 0.08);
    border-radius: 12px;
    padding: 0.75rem;
}

.story-shape-alternatives p {
    color: #334155 !important;
    font-size: 0.92rem;
    margin-top: 0.25rem;
}

.analysis-chat-intro {
    background: linear-gradient(135deg, #ffffff, #ecfdf5);
    border: 1px solid rgba(20, 184, 166, 0.28);
    border-left: 6px solid #f5b84b;
    border-radius: 18px;
    box-shadow: 0 18px 48px rgba(4, 12, 24, 0.12);
    padding: 1rem;
    margin: 1.25rem 0 0.85rem;
}

.analysis-chat-intro span {
    display: block;
    color: #0f766e !important;
    font-size: 0.72rem;
    font-weight: 850;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.analysis-chat-intro h3 {
    color: #07111f !important;
    margin: 0.22rem 0 0.35rem;
    font-size: 1.35rem;
}

.analysis-chat-intro p {
    color: #334155 !important;
    margin: 0;
    line-height: 1.6;
}

.st-key-story_bot_floating {
    position: fixed;
    right: 1.35rem;
    bottom: 1.25rem;
    z-index: 9999;
    width: 4.9rem !important;
    height: 4.9rem !important;
    pointer-events: auto;
    cursor: grab;
    user-select: none;
    touch-action: none;
    filter: drop-shadow(0 18px 34px rgba(4, 12, 24, 0.32));
}

.st-key-story_bot_floating::after {
    content: "Story Guide";
    position: absolute;
    right: 0;
    bottom: calc(100% + 0.76rem);
    z-index: 3;
    padding: 0.48rem 0.68rem;
    border-radius: 999px;
    background: rgba(7, 17, 31, 0.94);
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 12px 30px rgba(4, 12, 24, 0.28);
    font-size: 0.76rem;
    font-weight: 850;
    letter-spacing: 0.01em;
    line-height: 1;
    opacity: 0;
    pointer-events: none;
    transform: translateY(0.28rem) scale(0.96);
    transform-origin: right bottom;
    transition: opacity 150ms ease, transform 150ms ease;
    white-space: nowrap;
}

.st-key-story_bot_floating:hover::after,
.st-key-story_bot_floating.story-bot-open::after {
    opacity: 1;
    transform: translateY(0) scale(1);
}

.st-key-story_bot_floating.story-bot-open::after {
    content: "Story Guide";
}

.st-key-story_bot_floating.story-bot-opening::after {
    content: "Opening pages...";
}

.st-key-story_bot_floating.story-bot-closing::after {
    content: "Closing pages...";
}

.st-key-story_bot_floating [data-testid="stPopover"] {
    width: 4.9rem !important;
    height: 4.9rem !important;
}

.st-key-story_bot_floating,
.st-key-story_bot_floating > div,
.st-key-story_bot_floating [data-testid="stPopover"],
.st-key-story_bot_floating [data-testid="stPopover"] > div {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
}

.st-key-story_bot_launcher button,
.st-key-story_bot_floating button,
.st-key-story_bot_floating [data-testid="stPopover"] button,
.st-key-story_bot_floating [data-testid="baseButton-secondary"],
.st-key-story_bot_floating [data-testid="baseButton-secondaryFormSubmit"] {
    position: relative !important;
    width: 4.9rem !important;
    min-width: 4.9rem !important;
    max-width: 4.9rem !important;
    height: 4.9rem !important;
    min-height: 4.9rem !important;
    padding: 0 !important;
    display: grid !important;
    place-items: center !important;
    overflow: hidden !important;
    border-radius: 999px !important;
    background:
        radial-gradient(circle at 30% 22%, rgba(255, 255, 255, 0.55), transparent 28%),
        linear-gradient(145deg, rgba(255, 255, 255, 0.18), transparent 42%),
        linear-gradient(135deg, #14b8a6 0%, #0f766e 58%, #063b35 100%) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.62) !important;
    outline: 1px solid rgba(20, 184, 166, 0.16) !important;
    outline-offset: 6px !important;
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.45),
        inset 0 -12px 22px rgba(4, 12, 24, 0.2),
        0 14px 36px rgba(4, 12, 24, 0.28),
        0 0 0 9px rgba(20, 184, 166, 0.1) !important;
    font-size: 0 !important;
    line-height: 1 !important;
    perspective: 180px;
    isolation: isolate;
    transition: transform 160ms ease, box-shadow 160ms ease, background 180ms ease, outline-color 160ms ease !important;
}

.st-key-story_bot_launcher button:hover,
.st-key-story_bot_floating button:hover,
.st-key-story_bot_floating [data-testid="stPopover"] button:hover {
    transform: translateY(-3px) scale(1.035);
    outline-color: rgba(245, 158, 11, 0.22) !important;
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.5),
        inset 0 -12px 22px rgba(4, 12, 24, 0.18),
        0 22px 48px rgba(4, 12, 24, 0.34),
        0 0 0 10px rgba(20, 184, 166, 0.16) !important;
}

.st-key-story_bot_launcher button *,
.st-key-story_bot_launcher button p,
.st-key-story_bot_launcher button span,
.st-key-story_bot_floating button *,
.st-key-story_bot_floating button p,
.st-key-story_bot_floating button span,
.st-key-story_bot_floating button svg,
.st-key-story_bot_floating [data-testid="stPopover"] button *,
.st-key-story_bot_floating [data-testid="stPopover"] button p,
.st-key-story_bot_floating [data-testid="stPopover"] button span,
.st-key-story_bot_floating [data-testid="stPopover"] button svg {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 0 !important;
    width: 0 !important;
    height: 0 !important;
    opacity: 0 !important;
    margin: 0 !important;
}

.st-key-story_bot_floating button::before,
.st-key-story_bot_floating [data-testid="stPopover"] button::before {
    content: "";
    position: absolute;
    left: 50%;
    top: 50%;
    z-index: 1;
    display: block;
    width: 2.12rem;
    height: 2.72rem;
    border-radius: 0.42rem 0.78rem 0.78rem 0.42rem;
    background:
        linear-gradient(90deg, rgba(7, 17, 31, 0.28) 0 0.36rem, transparent 0.36rem),
        linear-gradient(120deg, rgba(255, 255, 255, 0.38), transparent 34%),
        linear-gradient(135deg, #fffaf0 0 0.14rem, transparent 0.14rem),
        linear-gradient(135deg, #0d9488 0%, #14b8a6 58%, #0f766e 100%);
    border: 1px solid rgba(255, 255, 255, 0.78);
    box-shadow:
        inset -0.2rem 0 0 rgba(255, 255, 255, 0.26),
        inset 0.34rem 0 0 rgba(7, 17, 31, 0.18),
        0 0.56rem 1.1rem rgba(4, 12, 24, 0.28);
    transform: translate(-50%, -50%) rotate(-8deg);
    transform-origin: center;
    transition: width 180ms ease, height 180ms ease, border-radius 180ms ease, background 180ms ease, transform 180ms ease, box-shadow 180ms ease;
}

.st-key-story_bot_floating button::after,
.st-key-story_bot_floating [data-testid="stPopover"] button::after {
    content: "";
    position: absolute;
    left: calc(50% + 0.24rem);
    top: calc(50% - 0.66rem);
    z-index: 2;
    width: 0.82rem;
    height: 0.11rem;
    border-radius: 999px;
    background: rgba(255, 250, 240, 0.92);
    box-shadow:
        0 0.34rem 0 rgba(255, 250, 240, 0.72),
        0 0.68rem 0 rgba(255, 250, 240, 0.52);
    transform: translate(-50%, -50%) rotate(-8deg);
    transition: width 180ms ease, height 180ms ease, top 180ms ease, left 180ms ease, background 180ms ease, box-shadow 180ms ease, transform 180ms ease;
}

.st-key-story_bot_floating.story-bot-open button,
.st-key-story_bot_floating button[aria-expanded="true"],
.st-key-story_bot_floating [data-testid="stPopover"] button[aria-expanded="true"] {
    background:
        radial-gradient(circle at 32% 22%, rgba(255, 255, 255, 0.62), transparent 28%),
        linear-gradient(145deg, rgba(255, 255, 255, 0.18), transparent 42%),
        linear-gradient(135deg, #f59e0b 0%, #14b8a6 48%, #0f766e 100%) !important;
}

.st-key-story_bot_floating.story-bot-open button::before,
.st-key-story_bot_floating button[aria-expanded="true"]::before,
.st-key-story_bot_floating [data-testid="stPopover"] button[aria-expanded="true"]::before {
    width: 3.12rem;
    height: 2.24rem;
    border-radius: 0.48rem 0.48rem 0.86rem 0.86rem;
    background:
        linear-gradient(90deg, transparent 0 46.5%, rgba(15, 118, 110, 0.5) 46.5% 53.5%, transparent 53.5% 100%),
        repeating-linear-gradient(180deg, transparent 0 0.42rem, rgba(15, 23, 42, 0.16) 0.42rem 0.48rem),
        linear-gradient(90deg, #fffaf0 0 50%, #fef3c7 50% 100%);
    border: 1px solid rgba(255, 255, 255, 0.9);
    box-shadow:
        inset 0 0 0 0.22rem rgba(255, 255, 255, 0.26),
        0 0.58rem 1.05rem rgba(4, 12, 24, 0.2);
    transform: translate(-50%, -50%) rotate(0deg) rotateX(8deg);
}

.st-key-story_bot_floating.story-bot-open button::after,
.st-key-story_bot_floating button[aria-expanded="true"]::after,
.st-key-story_bot_floating [data-testid="stPopover"] button[aria-expanded="true"]::after {
    left: 50%;
    top: 50%;
    width: 2.26rem;
    height: 0.08rem;
    background: rgba(15, 118, 110, 0.35);
    box-shadow:
        -0.46rem -0.42rem 0 rgba(15, 118, 110, 0.2),
        0.46rem -0.42rem 0 rgba(15, 118, 110, 0.2),
        -0.46rem -0.78rem 0 rgba(15, 118, 110, 0.18),
        0.46rem -0.78rem 0 rgba(15, 118, 110, 0.18);
    transform: translate(-50%, -10%) rotate(0deg);
}

.st-key-story_bot_floating.story-bot-opening button::before,
.st-key-story_bot_floating.story-bot-opening [data-testid="stPopover"] button::before {
    animation: storyBookOpenCover 520ms cubic-bezier(0.2, 0.85, 0.2, 1) both;
}

.st-key-story_bot_floating.story-bot-opening button::after,
.st-key-story_bot_floating.story-bot-opening [data-testid="stPopover"] button::after {
    animation: storyBookOpenLines 520ms cubic-bezier(0.2, 0.85, 0.2, 1) both;
}

.st-key-story_bot_floating.story-bot-closing button::before,
.st-key-story_bot_floating.story-bot-closing [data-testid="stPopover"] button::before {
    animation: storyBookCloseCover 460ms cubic-bezier(0.34, 0.02, 0.2, 1) both;
}

.st-key-story_bot_floating.story-bot-closing button::after,
.st-key-story_bot_floating.story-bot-closing [data-testid="stPopover"] button::after {
    animation: storyBookCloseLines 460ms cubic-bezier(0.34, 0.02, 0.2, 1) both;
}

@keyframes storyBookOpenCover {
    0% {
        width: 2.12rem;
        height: 2.72rem;
        border-radius: 0.42rem 0.78rem 0.78rem 0.42rem;
        transform: translate(-50%, -50%) rotate(-8deg) rotateY(0deg) scale(1);
    }
    42% {
        width: 2.48rem;
        height: 2.62rem;
        transform: translate(-50%, -50%) rotate(-3deg) rotateY(-42deg) scaleX(0.86);
    }
    100% {
        width: 3.12rem;
        height: 2.24rem;
        border-radius: 0.48rem 0.48rem 0.86rem 0.86rem;
        transform: translate(-50%, -50%) rotate(0deg) rotateX(8deg);
    }
}

@keyframes storyBookOpenLines {
    0% {
        left: calc(50% + 0.24rem);
        top: calc(50% - 0.66rem);
        width: 0.82rem;
        transform: translate(-50%, -50%) rotate(-8deg);
    }
    44% {
        left: calc(50% + 0.05rem);
        top: calc(50% - 0.36rem);
        width: 1.18rem;
        transform: translate(-50%, -50%) rotate(-4deg);
    }
    100% {
        left: 50%;
        top: 50%;
        width: 2.26rem;
        transform: translate(-50%, -10%) rotate(0deg);
    }
}

@keyframes storyBookCloseCover {
    0% {
        width: 3.12rem;
        height: 2.24rem;
        border-radius: 0.48rem 0.48rem 0.86rem 0.86rem;
        transform: translate(-50%, -50%) rotate(0deg) rotateX(8deg);
    }
    46% {
        width: 2.42rem;
        height: 2.58rem;
        transform: translate(-50%, -50%) rotate(-2deg) rotateY(38deg) scaleX(0.88);
    }
    100% {
        width: 2.12rem;
        height: 2.72rem;
        border-radius: 0.42rem 0.78rem 0.78rem 0.42rem;
        transform: translate(-50%, -50%) rotate(-8deg) rotateY(0deg) scale(1);
    }
}

@keyframes storyBookCloseLines {
    0% {
        left: 50%;
        top: 50%;
        width: 2.26rem;
        transform: translate(-50%, -10%) rotate(0deg);
    }
    46% {
        left: calc(50% + 0.05rem);
        top: calc(50% - 0.36rem);
        width: 1.18rem;
        transform: translate(-50%, -50%) rotate(-4deg);
    }
    100% {
        left: calc(50% + 0.24rem);
        top: calc(50% - 0.66rem);
        width: 0.82rem;
        transform: translate(-50%, -50%) rotate(-8deg);
    }
}

.st-key-story_bot_launcher button p,
.st-key-story_bot_floating button p,
.st-key-story_bot_floating [data-testid="stPopover"] button p {
    display: none !important;
}

.st-key-story_bot_launcher button::before {
    content: "";
}

[data-baseweb="tooltip"],
[data-baseweb="tooltip"] > div,
[role="tooltip"],
[data-testid="stTooltipContent"] {
    background: rgba(7, 17, 31, 0.96) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    border-radius: 9px !important;
    box-shadow: 0 12px 30px rgba(4, 12, 24, 0.32) !important;
}

[data-baseweb="tooltip"] *,
[role="tooltip"] *,
[data-testid="stTooltipContent"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

.stPopover,
[data-testid="stPopover"],
[data-testid="stPopoverBody"],
[data-testid="stPopoverContent"],
[data-testid="stPopover"] [role="dialog"],
[data-testid="stPopoverBody"] [role="dialog"],
[data-testid="stPopoverContent"] [role="dialog"] {
    background: #fffaf0 !important;
    color: #07111f !important;
}

[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="popover"] [role="dialog"],
[data-baseweb="popover"] [data-testid="stVerticalBlock"],
[data-baseweb="popover"] section,
[data-baseweb="popover"] article {
    background: #fffaf0 !important;
    color: #07111f !important;
    border-color: rgba(101, 67, 33, 0.18) !important;
    max-width: calc(100vw - 1rem) !important;
}

.stPopover *,
[data-testid="stPopover"] *,
[data-testid="stPopoverBody"] *,
[data-testid="stPopoverContent"] *,
[data-baseweb="popover"] * {
    color: #07111f !important;
    -webkit-text-fill-color: #07111f !important;
}

[data-baseweb="popover"] textarea,
[data-baseweb="popover"] input {
    background: #ffffff !important;
    color: #07111f !important;
    -webkit-text-fill-color: #07111f !important;
    caret-color: #0f766e !important;
}

[data-baseweb="popover"] textarea::placeholder,
[data-baseweb="popover"] input::placeholder {
    color: #64748b !important;
    -webkit-text-fill-color: #64748b !important;
}

.st-key-story_bot_panel {
    position: relative;
    background:
        radial-gradient(ellipse at center, rgba(101, 67, 33, 0.2), transparent 34%),
        linear-gradient(90deg, #efe0c9 0%, #fff8e8 8%, #fffdf5 49%, #d4b98f 50%, #fffdf5 51%, #fff8e8 92%, #ead8bc 100%);
    border: 1px solid rgba(101, 67, 33, 0.28);
    border-radius: 18px;
    box-shadow:
        inset 0 0 0 1px rgba(255, 255, 255, 0.55),
        inset 0 0 48px rgba(101, 67, 33, 0.1),
        0 28px 80px rgba(4, 12, 24, 0.32);
    padding: 1rem;
    width: min(860px, calc(100vw - 2rem));
    max-height: min(84vh, 720px);
    overflow: hidden;
}

.st-key-story_bot_panel::before {
    content: "";
    position: absolute;
    inset: 1rem calc(50% - 0.08rem);
    width: 0.16rem;
    border-radius: 999px;
    background:
        linear-gradient(180deg, transparent, rgba(101, 67, 33, 0.38) 12%, rgba(15, 118, 110, 0.22) 50%, rgba(101, 67, 33, 0.32) 88%, transparent);
    box-shadow:
        -0.7rem 0 1.4rem rgba(101, 67, 33, 0.12),
        0.7rem 0 1.4rem rgba(101, 67, 33, 0.12);
    pointer-events: none;
    z-index: 0;
}

.st-key-story_bot_book {
    position: relative;
    z-index: 1;
}

.st-key-story_bot_book [data-testid="column"] {
    background: transparent !important;
}

.st-key-story_bot_left_page,
.st-key-story_bot_right_page {
    min-height: 560px;
    max-height: calc(min(84vh, 720px) - 2rem);
    overflow: hidden;
    background:
        linear-gradient(90deg, rgba(20, 184, 166, 0.18) 0 0.18rem, transparent 0.18rem 0.55rem),
        repeating-linear-gradient(180deg, rgba(15, 23, 42, 0.035) 0 1px, transparent 1px 1.64rem),
        linear-gradient(135deg, #fffaf0, #fffdf7 52%, #f8efd9);
    border: 1px solid rgba(101, 67, 33, 0.18);
    box-shadow:
        inset 0 0 0 1px rgba(255, 255, 255, 0.65),
        inset 0 18px 42px rgba(255, 255, 255, 0.32);
    padding: 1rem 1rem 1rem 1.18rem;
}

.st-key-story_bot_left_page {
    border-radius: 14px 8px 8px 18px;
}

.st-key-story_bot_right_page {
    border-radius: 8px 14px 18px 8px;
    display: flex;
    flex-direction: column;
}

.story-book-page-heading {
    border-bottom: 1px solid rgba(101, 67, 33, 0.18);
    margin-bottom: 0.75rem;
    padding-bottom: 0.55rem;
}

.story-book-page-heading span {
    display: block;
    color: #0f766e !important;
    font-size: 0.64rem;
    font-weight: 900;
    letter-spacing: 0.12em;
    line-height: 1.2;
    text-transform: uppercase;
}

.story-book-page-heading h3 {
    color: #07111f !important;
    font-family: Georgia, "Times New Roman", serif;
    font-size: 1.25rem;
    line-height: 1.15;
    margin: 0.22rem 0 0;
}

.story-book-note {
    background: rgba(255, 255, 255, 0.64);
    border: 1px solid rgba(20, 184, 166, 0.16);
    border-radius: 12px;
    color: #334155 !important;
    font-family: Georgia, "Times New Roman", serif;
    font-size: 0.92rem;
    line-height: 1.55;
    margin: 0.45rem 0 0.75rem;
    padding: 0.72rem;
}

.story-book-stat-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.45rem;
    margin: 0.65rem 0 0.8rem;
}

.story-book-stat {
    background: rgba(255, 255, 255, 0.66);
    border: 1px solid rgba(101, 67, 33, 0.14);
    border-radius: 10px;
    padding: 0.5rem;
}

.story-book-stat span {
    display: block;
    color: #64748b !important;
    font-size: 0.62rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.story-book-stat strong {
    display: block;
    color: #07111f !important;
    font-size: 0.95rem;
    margin-top: 0.16rem;
    overflow-wrap: anywhere;
}

.st-key-story_bot_panel textarea {
    background: rgba(255, 255, 255, 0.92) !important;
    color: #07111f !important;
    -webkit-text-fill-color: #07111f !important;
    border: 1px solid rgba(20, 184, 166, 0.28) !important;
    border-radius: 12px !important;
}

.st-key-story_bot_panel button,
.st-key-story_bot_floating .st-key-story_bot_panel button {
    position: static !important;
    width: auto !important;
    min-width: 0 !important;
    max-width: none !important;
    height: auto !important;
    min-height: 2.45rem !important;
    padding: 0.45rem 0.72rem !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: visible !important;
    background: #ffffff !important;
    border: 1px solid rgba(20, 184, 166, 0.22) !important;
    border-radius: 12px !important;
    color: #063b35 !important;
    -webkit-text-fill-color: #063b35 !important;
    box-shadow: 0 8px 22px rgba(4, 12, 24, 0.07) !important;
    font-size: 0.88rem !important;
    line-height: 1.2 !important;
    text-transform: none !important;
}

.st-key-story_bot_panel button *,
.st-key-story_bot_panel button p,
.st-key-story_bot_panel button span,
.st-key-story_bot_floating .st-key-story_bot_panel button *,
.st-key-story_bot_floating .st-key-story_bot_panel button p,
.st-key-story_bot_floating .st-key-story_bot_panel button span {
    color: #063b35 !important;
    -webkit-text-fill-color: #063b35 !important;
    font-size: inherit !important;
    width: auto !important;
    height: auto !important;
    opacity: 1 !important;
    margin: 0 !important;
}

.st-key-story_bot_panel button::before,
.st-key-story_bot_panel button::after,
.st-key-story_bot_floating .st-key-story_bot_panel button::before,
.st-key-story_bot_floating .st-key-story_bot_panel button::after {
    content: none !important;
}

.st-key-story_bot_panel [data-testid="stForm"] {
    background:
        linear-gradient(135deg, rgba(255, 255, 255, 0.7), rgba(255, 250, 240, 0.7)) !important;
    border: 1px solid rgba(101, 67, 33, 0.16) !important;
    border-radius: 13px !important;
    padding: 0.65rem !important;
    margin-top: 0.72rem !important;
}

.st-key-story_bot_panel [data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #14b8a6, #0f766e) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 1px solid rgba(15, 118, 110, 0.34) !important;
}

.st-key-story_bot_panel [data-testid="stFormSubmitButton"] button *,
.st-key-story_bot_panel [data-testid="stFormSubmitButton"] button p,
.st-key-story_bot_panel [data-testid="stFormSubmitButton"] button span {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

.st-key-story_bot_panel p,
.st-key-story_bot_panel li,
.st-key-story_bot_panel div,
.st-key-story_bot_panel span,
.st-key-story_bot_panel label,
.st-key-story_bot_panel strong {
    color: #07111f !important;
    -webkit-text-fill-color: #07111f !important;
}

.story-bot-title {
    display: none;
}

.story-bot-title span {
    display: block;
    color: #0f766e !important;
    font-size: 0.68rem;
    font-weight: 850;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.story-bot-title h3 {
    color: #07111f !important;
    font-size: 1.24rem;
    margin: 0.2rem 0 0;
}

.story-bot-tip {
    display: none;
}

.story-bot-quick-title {
    color: #0f766e !important;
    font-size: 0.68rem;
    font-weight: 850;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 0.5rem 0 0.25rem;
}

.st-key-story_bot_quick_questions button {
    background: linear-gradient(135deg, #ffffff, #ecfdf5) !important;
    border: 1px solid rgba(20, 184, 166, 0.28) !important;
    color: #063b35 !important;
    -webkit-text-fill-color: #063b35 !important;
    box-shadow: 0 8px 18px rgba(4, 12, 24, 0.06) !important;
    font-weight: 850 !important;
}

.st-key-story_bot_quick_questions button *,
.st-key-story_bot_quick_questions button p,
.st-key-story_bot_quick_questions button span {
    color: #063b35 !important;
    -webkit-text-fill-color: #063b35 !important;
}

.st-key-story_bot_messages {
    background:
        linear-gradient(90deg, rgba(20, 184, 166, 0.16) 0 0.16rem, transparent 0.16rem 0.48rem),
        repeating-linear-gradient(180deg, rgba(15, 23, 42, 0.034) 0 1px, transparent 1px 1.55rem),
        linear-gradient(135deg, rgba(255, 255, 255, 0.72), rgba(255, 250, 240, 0.66));
    border: 1px solid rgba(101, 67, 33, 0.14);
    border-radius: 12px;
    box-shadow: inset 3px 0 0 rgba(20, 184, 166, 0.12);
    padding: 0.78rem 0.82rem 0.78rem 1rem;
    color: #07111f !important;
    -webkit-text-fill-color: #07111f !important;
    margin-top: 0.25rem;
}

.st-key-story_bot_messages p,
.st-key-story_bot_messages li,
.st-key-story_bot_messages div,
.st-key-story_bot_messages span,
.st-key-story_bot_messages strong {
    color: #07111f !important;
    -webkit-text-fill-color: #07111f !important;
}

.st-key-story_bot_messages ul,
.st-key-story_bot_messages ol {
    padding-left: 1.1rem;
}

.st-key-story_bot_messages hr {
    border-color: rgba(101, 67, 33, 0.14) !important;
}

.story-bot-empty {
    color: #475569 !important;
    font-size: 0.92rem;
    line-height: 1.55;
    margin: 0;
}

.story-bot-speaker {
    color: #0f766e !important;
    font-size: 0.72rem;
    font-weight: 850;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0.4rem 0 0.12rem;
}

.story-bot-user {
    color: #b45309 !important;
}

@media (max-width: 700px) {
    .st-key-story_bot_floating {
        right: 0.8rem;
        bottom: 0.8rem;
    }

    .st-key-story_bot_panel {
        width: min(96vw, 480px);
        max-height: 82vh;
        overflow-y: auto;
        background: linear-gradient(135deg, #fffaf0, #f0fdfa);
    }

    .st-key-story_bot_panel::before {
        display: none;
    }

    .st-key-story_bot_left_page,
    .st-key-story_bot_right_page {
        min-height: auto;
        max-height: none;
        border-radius: 14px;
        margin-bottom: 0.8rem;
    }
}

pre,
code,
[data-testid="stCodeBlock"],
[data-testid="stCodeBlock"] * {
    background: #ecfdf5 !important;
    color: #07111f !important;
    -webkit-text-fill-color: #07111f !important;
    border-color: rgba(20, 184, 166, 0.24) !important;
}

[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #d1fae5, #99f6e4) !important;
    color: #063b35 !important;
    -webkit-text-fill-color: #063b35 !important;
    border: 1px solid rgba(20, 184, 166, 0.36) !important;
    box-shadow: 0 12px 30px rgba(20, 184, 166, 0.16) !important;
}

[data-testid="stDownloadButton"] button *,
[data-testid="stDownloadButton"] button p,
[data-testid="stDownloadButton"] button span {
    color: #063b35 !important;
    -webkit-text-fill-color: #063b35 !important;
}

[data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderFile"] *,
[data-testid="stFileUploaderFileName"],
[data-testid="stFileUploaderFileName"] * {
    background: #ecfdf5 !important;
    color: #07111f !important;
    -webkit-text-fill-color: #07111f !important;
    border-color: rgba(20, 184, 166, 0.28) !important;
}

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(18px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes revealUp {
    from {
        opacity: 0.08;
        transform: translateY(22px) scale(0.992);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

@keyframes lineGrow {
    from {
        transform: scaleX(0);
        opacity: 0;
    }
    to {
        transform: scaleX(1);
        opacity: 1;
    }
}

@keyframes chartGlow {
    0% {
        opacity: 0;
        transform: translateX(-18%);
    }
    45% {
        opacity: 1;
    }
    100% {
        opacity: 0;
        transform: translateX(18%);
    }
}

@supports (animation-timeline: view()) {
    .section-heading,
    .visual-tile,
    .input-info-card,
    .peak-card,
    .summary-card,
    .story-shape-card,
    .stPlotlyChart,
    [data-testid="stDataFrame"] {
        animation-name: revealUp;
        animation-duration: 1ms;
        animation-timing-function: linear;
        animation-fill-mode: both;
        animation-timeline: view();
        animation-range: entry 0% cover 24%;
    }

    .section-report-panel,
    .section-report-panel * {
        animation-timeline: auto !important;
        animation-name: none !important;
        opacity: 1 !important;
        transform: none !important;
        filter: none !important;
    }
}

@keyframes chipFloat {
    0%, 100% {
        transform: translateY(0);
    }
    50% {
        transform: translateY(-6px);
    }
}

@keyframes atmosphereShift {
    from {
        opacity: 0.86;
        transform: scale(1);
    }
    to {
        opacity: 1;
        transform: scale(1.03);
    }
}

@media (max-width: 900px) {
    .block-container {
        padding: 1rem 1rem 3rem 1rem;
    }

    .hero {
        min-height: 440px;
        border-radius: 22px;
        background-position: 62% center;
    }

    .hero-content {
        padding: 2rem;
    }

    .visual-grid,
    .metric-grid {
        grid-template-columns: 1fr;
    }

    .section-heading {
        display: block;
    }

    .summary-report-header,
    .summary-copy-grid,
    .summary-row-card,
    .story-shape-meta,
    .paragraph-row-meta {
        grid-template-columns: 1fr;
    }
}

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 1ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 1ms !important;
        scroll-behavior: auto !important;
    }
}
</style>
""", unsafe_allow_html=True)

page_backdrop_uri = image_data_uri("emotion-dashboard.png")
if page_backdrop_uri:
    st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background:
        linear-gradient(180deg, rgba(7, 17, 31, 0.95) 0%, rgba(9, 27, 40, 0.83) 23%, rgba(245, 248, 244, 0.93) 55%, rgba(235, 248, 244, 0.96) 100%),
        url("__PAGE_BACKDROP__"),
        repeating-linear-gradient(90deg, rgba(255, 255, 255, 0.055) 0 1px, transparent 1px 86px),
        repeating-linear-gradient(0deg, rgba(20, 184, 166, 0.075) 0 1px, transparent 1px 86px),
        linear-gradient(135deg, #07111f 0%, #10313d 42%, #f7efe5 100%) !important;
    background-attachment: fixed, fixed, fixed, fixed, fixed !important;
    background-position: center, right -14rem top -7rem, center, center, center !important;
    background-repeat: no-repeat, no-repeat, repeat, repeat, no-repeat !important;
    background-size: cover, min(82rem, 92vw) auto, auto, auto, cover !important;
}
</style>
""".replace("__PAGE_BACKDROP__", page_backdrop_uri), unsafe_allow_html=True)


# ----------------------------
# Page visuals
# ----------------------------
def render_hero() -> None:
    hero_uri = image_data_uri("hero-emotion.png")
    background = (
        f"background-image: url('{hero_uri}');"
        if hero_uri
        else "background: linear-gradient(135deg, #071629, #0f766e);"
    )

    st.markdown(f"""
<section class="hero" style="{background}">
    <div class="hero-content">
        <div class="hero-kicker" style="color:#21e6d1 !important;-webkit-text-fill-color:#21e6d1 !important;">AI-assisted literary analytics</div>
        <h1 class="hero-title" style="color:#ffffff !important;-webkit-text-fill-color:#ffffff !important;opacity:1 !important;">Narrative Emotion Analyzer</h1>
        <p class="hero-copy" style="color:rgba(255,255,255,0.95) !important;-webkit-text-fill-color:rgba(255,255,255,0.95) !important;">Turn chapters, letters, PDFs, scanned pages, and pasted passages into a clear emotional arc with interactive scores and section-level insight.</p>
        <div class="hero-chips">
            <span class="hero-chip" style="color:#ffffff !important;-webkit-text-fill-color:#ffffff !important;">Emotion flow</span>
            <span class="hero-chip" style="color:#ffffff !important;-webkit-text-fill-color:#ffffff !important;">OCR-ready</span>
            <span class="hero-chip" style="color:#ffffff !important;-webkit-text-fill-color:#ffffff !important;">Section insights</span>
        </div>
    </div>
</section>
""", unsafe_allow_html=True)


def render_visual_tiles() -> None:
    tiles = [
        ("manuscript-scan.png", "Manuscript Intake", "TXT, PDF, DOCX, image, and ZIP sources"),
        ("emotion-dashboard.png", "Emotion Signals", "Paragraph scores, peaks, and dominant moods"),
        ("hero-emotion.png", "Story Arc", "A visual path from text to emotional movement"),
    ]

    tile_markup = []
    for file_name, title, caption in tiles:
        uri = image_data_uri(file_name)
        if not uri:
            continue

        tile_markup.append(f"""
<figure class="visual-tile">
    <img src="{uri}" alt="{html.escape(title)}">
    <figcaption>
        <strong>{html.escape(title)}</strong>
        <span>{html.escape(caption)}</span>
    </figcaption>
</figure>
""")

    if tile_markup:
        st.markdown(f'<div class="visual-grid">{"".join(tile_markup)}</div>', unsafe_allow_html=True)


def render_section_heading(kicker: str, title: str, copy: str = "") -> None:
    copy_html = f"<p>{html.escape(copy)}</p>" if copy else ""
    st.markdown(f"""
<div class="section-heading">
    <div>
        <div class="section-kicker">{html.escape(kicker)}</div>
        <h2>{html.escape(title)}</h2>
    </div>
    {copy_html}
</div>
""", unsafe_allow_html=True)


def render_input_info(input_type: str, character_count: int) -> None:
    st.markdown(f"""
<div class="input-info-card">
    <h3>Input Information</h3>
    <p><b>Input Type:</b> {html.escape(input_type)}</p>
    <p><b>Extracted Character Count:</b> {character_count:,}</p>
</div>
""", unsafe_allow_html=True)


def render_metric_grid(analysis: dict) -> None:
    metrics = [
        ("Overall Valence", analysis["overall_score"], "#14b8a6"),
        ("Sentiment", analysis["overall_sentiment"], "#f5b84b"),
        ("Paragraphs", len(analysis["paragraphs"]), "#55b86f"),
        ("Dominant Emotion", f"{analysis['dominant_emotion']} {analysis['dominant_emoji']}", "#f9735b"),
    ]

    cards = []
    for label, value, accent in metrics:
        cards.append(f"""
<div class="metric-card" style="--accent: {accent};">
    <span>{html.escape(str(label))}</span>
    <strong>{html.escape(str(value))}</strong>
</div>
""")

    st.markdown(f'<div class="metric-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_distribution_grid(analysis: dict) -> None:
    metrics = [
        ("Positive", analysis["positive_count"], "#55b86f"),
        ("Negative", analysis["negative_count"], "#f9735b"),
        ("Neutral", analysis["neutral_count"], "#637082"),
        ("Peak Score", analysis["peak_score"], "#14b8a6"),
    ]

    cards = []
    for label, value, accent in metrics:
        cards.append(f"""
<div class="metric-card" style="--accent: {accent};">
    <span>{html.escape(str(label))}</span>
    <strong>{html.escape(str(value))}</strong>
</div>
""")

    st.markdown(f'<div class="metric-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_story_shape_card(analysis: dict) -> None:
    story_shape = analysis.get("story_shape") or {}
    if not story_shape:
        return

    candidates = story_shape.get("candidates", [])
    alternative_text = ", ".join(
        f"{item['name']} ({item['similarity']:.2f})" for item in candidates[1:4]
    )
    if not alternative_text:
        alternative_text = "Not enough variation for meaningful alternatives."

    st.markdown(f"""
<div class="story-shape-card">
    <h3>Story Shape Match</h3>
    <p>{html.escape(str(story_shape.get("description", "")))}</p>
    <div class="story-shape-meta">
        <div>
            <span>Canonical Shape</span>
            <strong>{html.escape(str(story_shape.get("name", "Unknown")))}</strong>
        </div>
        <div>
            <span>Similarity</span>
            <strong>{html.escape(f"{float(story_shape.get('similarity', 0.0)):.2f}")}</strong>
        </div>
        <div>
            <span>Engine</span>
            <strong>{html.escape(str(analysis.get("analysis_engine", "Emotion model")))}</strong>
        </div>
    </div>
    <div class="story-shape-alternatives">
        <span>Nearest Alternatives</span>
        <p>{html.escape(alternative_text)}</p>
    </div>
</div>
""", unsafe_allow_html=True)


def render_section_summary_report(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return

    def cell(value) -> str:
        if pd.isna(value):
            return ""
        return html.escape(str(value))

    detail_rows = []

    for _, row in summary_df.iterrows():
        section = cell(row.get("Section", ""))
        peak_score = cell(row.get("PeakScore", ""))
        peak_emotion = cell(row.get("PeakEmotion", ""))
        peak_emoji = cell(row.get("PeakEmoji", ""))
        peak_paragraph = cell(row.get("PeakParagraphNumber", ""))
        story_shape = cell(row.get("StoryShape", ""))
        story_shape_similarity = cell(row.get("StoryShapeSimilarity", ""))
        peak_text = cell(row.get("PeakText", ""))
        summary = cell(row.get("Summary", ""))
        emotion_label = f"{peak_emotion} {peak_emoji}".strip()

        detail_rows.append(f"""
<article class="summary-row-card">
<div class="summary-row-meta">
<span>Section</span>
<strong>{section}</strong>
<span>Peak Score</span>
<strong>{peak_score}</strong>
<span>Peak Emotion</span>
<strong>{emotion_label}</strong>
<span>Peak Paragraph</span>
<strong>{peak_paragraph}</strong>
<span>Story Shape</span>
<strong>{story_shape} ({story_shape_similarity})</strong>
</div>
<div class="summary-row-body">
<section>
<span>Peak Text</span>
<p>{peak_text}</p>
</section>
<section>
<span>Summary</span>
<p>{summary}</p>
</section>
</div>
</article>
""")

    report_html = (
        '<div class="section-report-panel">'
        '<div class="section-report-title"><div>'
        '<span>Readable summary</span>'
        '<h3>Detailed Section Report</h3>'
        '</div></div>'
        + '<div class="summary-row-list">'
        + "".join(detail_rows)
        + '</div>'
        + '<p class="summary-table-note">This section uses light green and white cards so the peak text and summaries stay readable on the page background.</p>'
        + '</div>'
    )
    st.markdown(report_html, unsafe_allow_html=True)


def render_paragraph_detail_report(df: pd.DataFrame, title: str) -> None:
    if df.empty:
        st.info("No paragraph details are available.")
        return

    cards = []
    for _, row in df.iterrows():
        paragraph = html.escape(str(row.get("Paragraph", "")))
        sentence_count = html.escape(str(row.get("SentenceCount", "")))
        score = html.escape(str(row.get("Score", "")))
        sentiment = html.escape(str(row.get("Sentiment", "")))
        emotion = html.escape(str(row.get("Emotion", "")))
        confidence = html.escape(str(row.get("EmotionConfidence", "")))
        emoji = html.escape(str(row.get("Emoji", "")))
        text = html.escape(str(row.get("Text", "")))

        cards.append(f"""
<article class="paragraph-row-card">
    <div class="paragraph-row-meta">
        <div>
            <span>Paragraph</span>
            <strong>{paragraph}</strong>
        </div>
        <div>
            <span>Sentences</span>
            <strong>{sentence_count}</strong>
        </div>
        <div>
            <span>Score</span>
            <strong>{score}</strong>
        </div>
        <div>
            <span>Sentiment</span>
            <strong>{sentiment}</strong>
        </div>
        <div>
            <span>Emotion</span>
            <strong>{emotion} {emoji}</strong>
        </div>
        <div>
            <span>Confidence</span>
            <strong>{confidence}</strong>
        </div>
    </div>
    <div class="paragraph-row-text">
        <span>Full Context</span>
        <p>{text}</p>
    </div>
</article>
""")

    st.markdown(f"""
<div class="paragraph-report-panel">
    <div class="paragraph-report-title">
        <div>
            <span>Paragraph-wise detail</span>
            <h3>{html.escape(title)}</h3>
        </div>
    </div>
    {"".join(cards)}
</div>
""", unsafe_allow_html=True)


CHAT_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "did", "do",
    "does", "find", "for", "from", "go", "goes", "happen", "happens", "happened",
    "how", "i", "in", "is", "it", "me", "of", "on", "or", "part", "plot", "show",
    "story", "tell", "that", "the", "there", "this", "to", "was", "what", "when",
    "where", "which", "who", "why", "with", "you",
}

QUICK_CHAT_QUESTIONS = [
    ("Why read it?", "Why should I read this book?"),
    ("Most interesting", "What is the most interesting part of this book?"),
    ("Peak emotion", "What is the peak emotion?"),
    ("Peak scene", "What happens in the peak scene?"),
    ("Most negative", "Which part is most emotionally negative?"),
    ("Most positive", "Which part is most emotionally positive?"),
    ("Story arc", "What is the story arc?"),
    ("Clear summary", "Give me a clear story summary."),
    ("Main themes", "What are the main themes?"),
    ("Love thread", "Is there a love concept in this story?"),
    ("Fear moments", "Where does fear appear?"),
    ("Sad moments", "Where does sadness appear?"),
]

THEME_KEYWORDS = {
    "love": [
        "love", "loved", "loving", "beloved", "affection", "affectionate",
        "attachment", "heart", "dear", "marriage", "mistress", "companion",
        "friend", "friendship", "union",
    ],
    "loneliness": [
        "alone", "lonely", "solitude", "solitary", "isolated", "isolation",
        "companionless", "friendless", "separation", "forsaken",
    ],
    "ambition": [
        "ambition", "ambitious", "enterprise", "undertaking", "success",
        "glory", "discovery", "pursuit", "knowledge", "endeavour",
    ],
    "fear": ["fear", "afraid", "terror", "horror", "dread", "anxiety", "panic", "alarm"],
    "family": ["father", "mother", "sister", "brother", "family", "parent", "child", "children"],
    "death": ["death", "dead", "die", "died", "grave", "murder", "corpse", "funeral"],
    "nature": ["nature", "mountain", "lake", "forest", "sky", "sea", "river", "wood", "landscape"],
    "science": ["science", "scientific", "experiment", "chemistry", "philosophy", "laboratory", "creation"],
    "guilt": ["guilt", "guilty", "remorse", "regret", "blame", "confession", "shame"],
    "revenge": ["revenge", "vengeance", "enemy", "hate", "hatred", "destroy", "murderer"],
    "hope": ["hope", "confidence", "promise", "comfort", "relief", "joy", "rejoice"],
}

THEME_INTERPRETATIONS = {
    "love": "It appears less like a simple romance label and more as attachment, affection, longing for companionship, and relationship tension.",
    "loneliness": "It shows up as emotional isolation: characters wanting connection, safety, or understanding.",
    "ambition": "It is tied to pursuit, success, discovery, and the emotional cost of wanting something intensely.",
    "fear": "It is visible as anxiety, dread, danger, or emotional pressure around uncertain events.",
    "family": "It appears through duties, letters, care, obligation, and emotional bonds between relatives.",
    "death": "It marks the darker pressure points of the story: loss, danger, grief, or irreversible consequence.",
    "nature": "It works as emotional scenery, often reflecting awe, comfort, or inner disturbance.",
    "science": "It connects knowledge and experiment with risk, responsibility, and consequence.",
    "guilt": "It points to responsibility, regret, and the emotional aftermath of choices.",
    "revenge": "It appears as conflict hardened into anger, pursuit, or retaliation.",
    "hope": "It appears where the language turns toward confidence, comfort, recovery, or expectation.",
}

GENERIC_THEME_TRIGGERS = {
    "concept", "theme", "themes", "idea", "ideas", "motif", "motifs", "meaning",
    "message", "symbol", "symbols", "topic", "topics", "about", "present",
}

READER_RECOMMENDATION_TERMS = [
    "why should i read",
    "should i read",
    "worth reading",
    "recommend",
    "suggest this book",
    "sell me",
    "why read",
]

READER_INTEREST_TERMS = [
    "most interesting",
    "more interesting",
    "interesting part",
    "best part",
    "standout part",
    "compelling part",
    "exciting part",
    "what makes",
]

QUESTION_STARTERS = (
    "is ", "are ", "does ", "do ", "did ", "has ", "have ", "can ", "could ",
    "would ", "should ", "was ", "were ",
)


def compact_excerpt(text: str, limit: int = 520) -> str:
    compact = re.sub(r"\s+", " ", clean_extracted_text(str(text))).strip()
    if len(compact) <= limit:
        return compact
    return compact[:limit].rsplit(" ", 1)[0].strip() + "..."


def chat_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", text.lower())
    return [token for token in tokens if token not in CHAT_STOP_WORDS]


def format_score(value) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def make_chat_section(title: str, analysis: dict) -> dict:
    df = analysis.get("df", pd.DataFrame()).copy()
    paragraph_rows = df.to_dict("records") if not df.empty else []
    story_shape = analysis.get("story_shape") or {}

    return {
        "title": title,
        "analysis": analysis,
        "paragraphs": paragraph_rows,
        "summary": str(analysis.get("summary", "")),
        "dominant_emotion": str(analysis.get("dominant_emotion", "neutral")),
        "dominant_emoji": str(analysis.get("dominant_emoji", "")),
        "overall_score": analysis.get("overall_score", 0.0),
        "overall_sentiment": str(analysis.get("overall_sentiment", "neutral")),
        "peak_paragraph_number": analysis.get("peak_paragraph_number", ""),
        "peak_score": analysis.get("peak_score", 0.0),
        "peak_emotion": str(analysis.get("peak_emotion", "neutral")),
        "peak_emoji": str(analysis.get("peak_emoji", "")),
        "peak_text": str(analysis.get("peak_text", "")),
        "story_shape": story_shape,
    }


def build_analysis_chat_context(
    sections: list[tuple[str, dict]],
    full_analysis: dict | None = None,
    mode: str = "single",
    input_type: str = "",
    character_count: int = 0,
) -> dict:
    chat_sections = [make_chat_section(title, analysis) for title, analysis in sections]
    full_section = make_chat_section("Entire Text", full_analysis) if full_analysis is not None else None
    fingerprint_parts = [
        section["title"] + str(section["overall_score"]) + str(len(section["paragraphs"]))
        for section in chat_sections
    ]
    if full_section:
        fingerprint_parts.append("Entire Text" + str(full_section["overall_score"]))

    context_id = hashlib.sha1("|".join(fingerprint_parts).encode("utf-8")).hexdigest()
    return {
        "id": context_id,
        "mode": mode,
        "input_type": input_type,
        "character_count": character_count,
        "sections": chat_sections,
        "full": full_section,
    }


def set_analysis_chat_context(context: dict) -> None:
    previous_id = st.session_state.get("analysis_chat_context_id")
    st.session_state.analysis_chat_context = context
    st.session_state.analysis_chat_context_id = context.get("id")
    if previous_id != context.get("id"):
        st.session_state.analysis_chat_messages = []


def find_section_for_question(question: str, context: dict) -> dict | None:
    q = question.lower()
    if any(word in q for word in ["entire", "overall", "whole", "full text", "complete"]):
        return context.get("full")

    sections = context.get("sections", [])
    for section in sections:
        title = section["title"].lower()
        if title and title in q:
            return section

    section_match = re.search(r"\b(chapter|letter|section)\s+([a-z0-9ivxlcdm]+)\b", q)
    if section_match:
        needle = f"{section_match.group(1)} {section_match.group(2)}"
        for section in sections:
            if needle in section["title"].lower():
                return section

    return None


def all_chat_sections(context: dict) -> list[dict]:
    sections = list(context.get("sections", []))
    full = context.get("full")
    if full is not None:
        sections.append(full)
    return sections


def strongest_peak_section(context: dict, selected_section: dict | None = None) -> dict | None:
    if selected_section is not None:
        return selected_section

    candidates = all_chat_sections(context)
    if not candidates:
        return None
    return max(candidates, key=lambda section: abs(float(section.get("peak_score", 0.0) or 0.0)))


def top_theme_signals(context: dict, selected_section: dict | None = None, limit: int = 3) -> list[dict]:
    sections = analysis_scope_sections(context, selected_section)
    signals = []

    for theme, terms in THEME_KEYWORDS.items():
        hits = 0
        emotions = Counter()
        for section in sections:
            for row in section.get("paragraphs", []):
                row_hits = count_theme_hits(str(row.get("Text", "")), terms)
                if not row_hits:
                    continue
                hits += row_hits
                emotions[str(row.get("Emotion", "neutral"))] += row_hits

        if hits:
            dominant_emotion = emotions.most_common(1)[0][0] if emotions else "neutral"
            signals.append({
                "theme": theme,
                "hits": hits,
                "dominant_emotion": dominant_emotion,
            })

    signals.sort(key=lambda item: item["hits"], reverse=True)
    return signals[:limit]


def strongest_paragraph_moments(context: dict, selected_section: dict | None = None, limit: int = 3) -> list[dict]:
    sections = [selected_section] if selected_section else all_chat_sections(context)
    rows = []

    for section in sections:
        if not section:
            continue
        for row in section.get("paragraphs", []):
            try:
                score = float(row.get("Score", 0.0) or 0.0)
            except Exception:
                score = 0.0
            rows.append({
                "section": section,
                "row": row,
                "score": score,
                "intensity": abs(score),
            })

    rows.sort(key=lambda item: item["intensity"], reverse=True)
    return rows[:limit]


def has_reader_recommendation_intent(question: str) -> bool:
    q = question.lower()
    return any(term in q for term in READER_RECOMMENDATION_TERMS + READER_INTEREST_TERMS)


def describe_reader_recommendation(question: str, context: dict, selected_section: dict | None, default_section: dict) -> str:
    q = question.lower()
    wants_recommendation = any(term in q for term in READER_RECOMMENDATION_TERMS)
    wants_interest = any(term in q for term in READER_INTEREST_TERMS)
    themes = top_theme_signals(context, selected_section, limit=3)
    theme_text = ", ".join(theme["theme"] for theme in themes) if themes else "its emotional movement"
    peak_section = strongest_peak_section(context, selected_section) or default_section
    peak_text = compact_excerpt(peak_section.get("peak_text", ""), limit=320)
    moments = strongest_paragraph_moments(context, selected_section, limit=2)

    reason = (
        f"Read it if you want a story driven by **{theme_text}**, with an emotional tone that leans "
        f"**{default_section['overall_sentiment']}** overall and centers on "
        f"**{default_section['dominant_emotion']} {default_section['dominant_emoji']}**."
    )

    interest_line = (
        f"**Most interesting part:** The strongest moment is in **{peak_section['title']}**, "
        f"paragraph **{peak_section['peak_paragraph_number']}**, where the emotion peaks as "
        f"**{peak_section['peak_emotion']} {peak_section['peak_emoji']}** "
        f"(score **{format_score(peak_section['peak_score'])}**)."
    )

    if wants_interest and not wants_recommendation:
        lines = [
            interest_line,
            f"**What happens there:** {peak_text}",
            "",
            f"**Why it stands out:** It is the clearest emotional pressure point in the text, and it best shows the story's pull toward **{theme_text}**.",
        ]
    else:
        lines = [
            "**Short recommendation:** Yes, it is worth reading if you enjoy emotionally layered storytelling.",
            "",
            f"**Why it is worth your time:** {reason}",
            "",
            interest_line,
            f"**What happens there:** {peak_text}",
        ]

    if themes:
        theme_names = ", ".join(f"**{theme['theme']}**" for theme in themes)
        lines.extend([
            "",
            f"**What to watch for while reading:** {theme_names}. These are the strongest recurring signals I found in the text.",
        ])

    if len(moments) > 1:
        supporting = moments[1]
        row = supporting["row"]
        lines.extend([
            "",
            (
                f"**Another standout moment:** **{supporting['section']['title']}**, paragraph "
                f"**{row.get('Paragraph')}** carries **{row.get('Emotion')} {row.get('Emoji', '')}** "
                f"with score **{format_score(row.get('Score'))}**."
            ),
        ])

    lines.append("")
    lines.append("**Bottom line:** The appeal is not just what happens, but how strongly the emotional pressure shifts across the text.")
    return "\n".join(lines)


def describe_peak(section: dict) -> str:
    return (
        f"**Peak emotion in {section['title']}**\n\n"
        f"- Emotion: **{section['peak_emotion']} {section['peak_emoji']}**\n"
        f"- Score: **{format_score(section['peak_score'])}**\n"
        f"- Paragraph: **{section['peak_paragraph_number']}**\n\n"
        f"**What happens there:** {compact_excerpt(section['peak_text'])}"
    )


def describe_story_shape(section: dict) -> str:
    shape = section.get("story_shape") or {}
    if not shape:
        return f"I do not have a story-shape match for **{section['title']}** yet."

    return (
        f"**Story shape for {section['title']}**\n\n"
        f"- Match: **{shape.get('name', 'Unknown')}**\n"
        f"- Similarity: **{format_score(shape.get('similarity', 0.0))}**\n"
        f"- Meaning: {shape.get('description', '')}\n\n"
        f"The dominant emotion is **{section['dominant_emotion']} {section['dominant_emoji']}**, "
        f"with overall valence **{format_score(section['overall_score'])}**."
    )


def describe_overview(section: dict) -> str:
    return (
        f"**Overview for {section['title']}**\n\n"
        f"{section['summary']}\n\n"
        f"- Dominant emotion: **{section['dominant_emotion']} {section['dominant_emoji']}**\n"
        f"- Overall sentiment: **{section['overall_sentiment']}**\n"
        f"- Overall valence: **{format_score(section['overall_score'])}**\n"
        f"- Peak: **{section['peak_emotion']} {section['peak_emoji']}**, paragraph "
        f"**{section['peak_paragraph_number']}**"
    )


def find_relevant_passages(question: str, context: dict, selected_section: dict | None = None, limit: int = 3) -> list[dict]:
    q = question.lower()
    paragraph_match = re.search(r"\bparagraph\s+(\d+)\b", q)
    sections = [selected_section] if selected_section else all_chat_sections(context)
    sections = [section for section in sections if section is not None]

    if paragraph_match:
        paragraph_number = int(paragraph_match.group(1))
        for section in sections:
            for row in section.get("paragraphs", []):
                if int(row.get("Paragraph", -1)) == paragraph_number:
                    return [{"section": section, "row": row, "score": 100}]

    tokens = chat_tokens(question)
    if not tokens:
        return []

    scored_rows = []
    for section in sections:
        title_lower = section["title"].lower()
        for row in section.get("paragraphs", []):
            text = str(row.get("Text", ""))
            haystack = f"{title_lower} {str(row.get('Emotion', '')).lower()} {text.lower()}"
            score = 0
            for token in tokens:
                if token in title_lower:
                    score += 3
                if token == str(row.get("Emotion", "")).lower():
                    score += 4
                if token in haystack:
                    score += 1
                score += min(3, haystack.count(token))

            if score > 0:
                scored_rows.append({"section": section, "row": row, "score": score})

    scored_rows.sort(key=lambda item: item["score"], reverse=True)
    return scored_rows[:limit]


def describe_passages(matches: list[dict], fallback_title: str = "Closest matching passages") -> str:
    if not matches:
        return ""

    parts = [f"**{fallback_title}**"]
    for item in matches:
        section = item["section"]
        row = item["row"]
        parts.append(
            "\n"
            f"- **{section['title']}, paragraph {row.get('Paragraph')}**: "
            f"{row.get('Emotion')} {row.get('Emoji', '')}, score **{format_score(row.get('Score'))}**. "
            f"{compact_excerpt(row.get('Text', ''), limit=360)}"
        )
    return "\n".join(parts)


def analysis_scope_sections(context: dict, selected_section: dict | None = None) -> list[dict]:
    if selected_section is not None:
        return [selected_section]

    sections = list(context.get("sections", []))
    if sections:
        return sections

    full = context.get("full")
    return [full] if full is not None else []


def infer_theme_terms(question: str) -> tuple[str, list[str]]:
    q = question.lower()
    tokens = chat_tokens(q)

    for theme, keywords in THEME_KEYWORDS.items():
        if theme in tokens or any(re.search(rf"\b{re.escape(keyword)}\b", q) for keyword in keywords):
            return theme, keywords

    theme_noise = GENERIC_THEME_TRIGGERS | {
        "story", "book", "novel", "text", "chapter", "letter", "part", "anything",
        "something", "any", "there", "this", "it",
    }
    concept_terms = [token for token in tokens if token not in theme_noise]
    if concept_terms:
        return " ".join(concept_terms[:3]), concept_terms[:5]

    return "", []


def is_generic_theme_question(question: str) -> bool:
    q = question.lower().strip()
    tokens = set(chat_tokens(q))
    has_theme_trigger = bool(tokens & GENERIC_THEME_TRIGGERS)
    has_known_theme = any(
        theme in tokens or any(re.search(rf"\b{re.escape(keyword)}\b", q) for keyword in keywords)
        for theme, keywords in THEME_KEYWORDS.items()
    )
    starts_like_question = q.startswith(QUESTION_STARTERS)
    return has_theme_trigger or (starts_like_question and has_known_theme) or " in this story" in q or " in it" in q


def count_theme_hits(text: str, terms: list[str]) -> int:
    text_lower = text.lower()
    hits = 0
    for term in terms:
        term_lower = term.lower().strip()
        if not term_lower:
            continue
        if " " in term_lower:
            hits += text_lower.count(term_lower)
        else:
            hits += len(re.findall(rf"\b{re.escape(term_lower)}\w*\b", text_lower))
    return hits


def collect_theme_evidence(
    question: str,
    context: dict,
    selected_section: dict | None = None,
    limit: int = 5,
) -> tuple[str, list[str], list[dict]]:
    theme, terms = infer_theme_terms(question)
    if not terms:
        return theme, terms, []

    evidence = []
    for section in analysis_scope_sections(context, selected_section):
        for row in section.get("paragraphs", []):
            text = str(row.get("Text", ""))
            hits = count_theme_hits(text, terms)
            if hits <= 0:
                continue
            try:
                score = float(row.get("Score", 0.0))
            except Exception:
                score = 0.0
            evidence.append({
                "section": section,
                "row": row,
                "hits": hits,
                "score": score,
                "abs_score": abs(score),
            })

    evidence.sort(key=lambda item: (item["hits"], item["abs_score"]), reverse=True)
    return theme, terms, evidence[:limit]


def confidence_label(total_hits: int, evidence_count: int, section_count: int) -> str:
    if total_hits >= 8 or evidence_count >= 5 or section_count >= 3:
        return "strong"
    if total_hits >= 3 or evidence_count >= 2:
        return "moderate"
    return "light"


def describe_theme_evidence(evidence: list[dict], title: str = "Good places to inspect") -> str:
    if not evidence:
        return ""

    lines = [f"**{title}:**"]
    for item in evidence[:3]:
        row = item["row"]
        section = item["section"]
        lines.append(
            f"- **{section['title']}, paragraph {row.get('Paragraph')}**: "
            f"{row.get('Emotion')} {row.get('Emoji', '')}, score **{format_score(row.get('Score'))}**. "
            f"{compact_excerpt(row.get('Text', ''), limit=260)}"
        )
    return "\n".join(lines)


def describe_detected_themes(context: dict, selected_section: dict | None, default_section: dict) -> str:
    sections = analysis_scope_sections(context, selected_section)
    theme_scores = []

    for theme, terms in THEME_KEYWORDS.items():
        hits = 0
        emotions = Counter()
        for section in sections:
            for row in section.get("paragraphs", []):
                row_hits = count_theme_hits(str(row.get("Text", "")), terms)
                if row_hits:
                    hits += row_hits
                    emotions[str(row.get("Emotion", "neutral"))] += 1
        if hits:
            dominant_emotion = emotions.most_common(1)[0][0] if emotions else "neutral"
            theme_scores.append((theme, hits, dominant_emotion))

    theme_scores.sort(key=lambda item: item[1], reverse=True)
    if not theme_scores:
        return describe_overview(default_section)

    lines = [
        f"**Main themes I can detect in {default_section['title']}**",
        "",
        "Here are the strongest theme signals from the analyzed text, based on repeated wording and the emotion scores around those passages:",
    ]
    for theme, hits, dominant_emotion in theme_scores[:5]:
        lines.append(
            f"- **{theme.title()}**: {hits} textual signal(s), usually near **{dominant_emotion}** moments. "
            f"{THEME_INTERPRETATIONS.get(theme, '')}"
        )

    lines.append("")
    lines.append(
        "Try asking one theme directly, for example: **Is ambition dangerous here?** or **Where does loneliness appear?**"
    )
    return "\n".join(lines)


def answer_theme_question(question: str, context: dict, selected_section: dict | None, default_section: dict) -> str:
    theme, terms, evidence = collect_theme_evidence(question, context, selected_section, limit=5)
    if not theme:
        return describe_detected_themes(context, selected_section, default_section)

    if not evidence:
        related = find_relevant_passages(question, context, selected_section, limit=2)
        response = [
            f"**Short answer:** I do not see a strong, direct **{theme}** pattern in the current analysis.",
            "",
            "That does not mean the idea is impossible; it means the analyzed text did not give me enough clear signals for it. "
            f"The story’s broader emotional center is **{default_section['dominant_emotion']} {default_section['dominant_emoji']}** "
            f"with overall valence **{format_score(default_section['overall_score'])}**.",
        ]
        if related:
            response.extend(["", describe_passages(related, "Closest related moments")])
        response.extend([
            "",
            f"You can make this sharper by asking: **Where is {theme} hinted?** or **Compare {theme} with fear.**",
        ])
        return "\n".join(response)

    total_hits = sum(item["hits"] for item in evidence)
    section_count = len({item["section"]["title"] for item in evidence})
    emotions = Counter(str(item["row"].get("Emotion", "neutral")) for item in evidence)
    dominant_emotion = emotions.most_common(1)[0][0] if emotions else "neutral"
    avg_score = float(np.mean([item["score"] for item in evidence])) if evidence else 0.0
    confidence = confidence_label(total_hits, len(evidence), section_count)
    direction = classify_sentiment(avg_score)
    interpretation = THEME_INTERPRETATIONS.get(
        theme,
        "It appears as a recurring idea in the language, but its meaning depends on the local scene.",
    )

    response = [
        f"**Short answer:** Yes, I see a **{confidence}** signal for **{theme}** in the analyzed story.",
        "",
        f"**How it appears:** {interpretation}",
        "",
        f"**Emotional pattern:** The matching moments lean **{direction}** overall "
        f"(average valence **{format_score(avg_score)}**) and most often carry **{dominant_emotion}**.",
        "",
        describe_theme_evidence(evidence),
        "",
        f"**My read:** This looks like a theme worth discussing, not just an isolated word match. "
        f"The chatbot found {total_hits} signal(s) across {section_count} section(s), so a good next question is: "
        f"**How does {theme} change from the beginning to the end?**",
    ]
    return "\n".join(part for part in response if part)


def answer_open_question_from_matches(question: str, matches: list[dict], default_section: dict) -> str:
    emotions = Counter(str(item["row"].get("Emotion", "neutral")) for item in matches)
    dominant_emotion = emotions.most_common(1)[0][0] if emotions else default_section["dominant_emotion"]
    scores = []
    for item in matches:
        try:
            scores.append(float(item["row"].get("Score", 0.0) or 0.0))
        except Exception:
            continue
    avg_score = float(np.mean(scores)) if scores else float(default_section.get("overall_score", 0.0) or 0.0)
    direction = classify_sentiment(avg_score)
    focus = ", ".join(chat_tokens(question)[:3]) or "your question"

    response = [
        f"**Here is the useful read:** I found a few moments connected to **{focus}**.",
        "",
        f"Across those matches, the emotional tone leans **{direction}** "
        f"(average valence **{format_score(avg_score)}**) and the most common detected emotion is **{dominant_emotion}**.",
        "",
        describe_theme_evidence(matches, "Best supporting moments"),
        "",
        "For a deeper answer, ask me to narrow it by section, compare it with another theme, or explain what changes before and after that moment.",
    ]
    return "\n".join(part for part in response if part)


def answer_analysis_question(question: str, context: dict) -> str:
    q = question.lower().strip()
    selected_section = find_section_for_question(question, context)
    default_section = selected_section or context.get("full") or (context.get("sections") or [None])[0]

    if not default_section:
        return "I do not have an analysis loaded yet. Run the analyzer first, then ask about a section, paragraph, peak, or emotion."

    if has_reader_recommendation_intent(question):
        return describe_reader_recommendation(question, context, selected_section, default_section)

    if any(phrase in q for phrase in ["most negative", "emotionally negative", "lowest", "darkest"]):
        section = strongest_peak_section(context, selected_section)
        candidates = [selected_section] if selected_section else all_chat_sections(context)
        rows = []
        for candidate in candidates:
            if not candidate:
                continue
            for row in candidate.get("paragraphs", []):
                try:
                    score = float(row.get("Score", 0.0))
                except Exception:
                    score = 0.0
                rows.append({"section": candidate, "row": row, "score": score})
        rows = sorted(rows, key=lambda item: item["score"])[:3]
        if rows:
            return describe_passages(rows, "Most emotionally negative parts")
        return describe_peak(section) if section else "I could not find a negative peak in the current analysis."

    if any(phrase in q for phrase in ["most positive", "emotionally positive", "highest positive", "brightest"]):
        candidates = [selected_section] if selected_section else all_chat_sections(context)
        rows = []
        for candidate in candidates:
            if not candidate:
                continue
            for row in candidate.get("paragraphs", []):
                try:
                    score = float(row.get("Score", 0.0))
                except Exception:
                    score = 0.0
                rows.append({"section": candidate, "row": row, "score": score})
        rows = sorted(rows, key=lambda item: item["score"], reverse=True)[:3]
        if rows:
            return describe_passages(rows, "Most emotionally positive parts")

    if any(word in q for word in ["peak", "highest", "strongest", "intense", "intensity"]):
        section = strongest_peak_section(context, selected_section)
        return describe_peak(section) if section else "I could not find a peak emotion in the current analysis."

    if any(word in q for word in ["shape", "arc", "rags", "tragedy", "cinderella", "icarus", "oedipus", "hole"]):
        return describe_story_shape(default_section)

    if any(phrase in q for phrase in ["main theme", "main themes", "what themes", "themes are", "concepts are", "ideas are"]):
        return describe_detected_themes(context, selected_section, default_section)

    emotion_words = set(EMOTION_LABELS + ["wonder", "mixed"])
    query_emotions = [emotion for emotion in emotion_words if re.search(rf"\b{re.escape(emotion)}\b", q)]
    if query_emotions:
        matches = find_relevant_passages(" ".join(query_emotions), context, selected_section, limit=4)
        if matches:
            return describe_passages(matches, f"Where **{query_emotions[0]}** appears")

    if is_generic_theme_question(question):
        return answer_theme_question(question, context, selected_section, default_section)

    if any(phrase in q for phrase in ["what happens", "plot", "summary", "story about", "part", "section", "chapter", "letter"]):
        matches = find_relevant_passages(question, context, selected_section, limit=2)
        if matches:
            return describe_passages(matches, "Relevant story moment")
        return describe_overview(default_section)

    matches = find_relevant_passages(question, context, selected_section, limit=3)
    if matches:
        return answer_open_question_from_matches(question, matches, default_section)

    return (
        f"I could not find a clear grounded answer for that exact wording, so here is the useful high-level read instead:\n\n"
        f"{describe_overview(default_section)}\n\n"
        "You can ask me in a more targeted way, like **Where does this theme appear?**, "
        "**What happens around paragraph 12?**, or **Compare love and fear in the story.**"
    )


def build_section_summary_rows_from_context(context: dict) -> pd.DataFrame:
    rows = []
    for section in context.get("sections", []):
        shape = section.get("story_shape") or {}
        rows.append({
            "Section": section.get("title", ""),
            "PeakScore": section.get("peak_score", ""),
            "PeakEmotion": section.get("peak_emotion", ""),
            "PeakEmoji": section.get("peak_emoji", ""),
            "PeakParagraphNumber": section.get("peak_paragraph_number", ""),
            "StoryShape": shape.get("name", ""),
            "StoryShapeSimilarity": shape.get("similarity", ""),
            "PeakText": section.get("peak_text", ""),
            "Summary": section.get("summary", ""),
        })
    return pd.DataFrame(rows)


def render_cached_analysis_results(context: dict) -> None:
    if not context:
        return

    input_type = context.get("input_type", "")
    character_count = int(context.get("character_count", 0) or 0)
    if input_type or character_count:
        render_input_info(input_type or "Previous analysis", character_count)

    sections = context.get("sections", [])
    mode = context.get("mode", "single")

    if mode == "single" and sections:
        section = sections[0]
        analysis = section.get("analysis")
        if analysis:
            st.markdown('<h3 class="chart-title">Interactive Emotion Flow</h3>', unsafe_allow_html=True)
            st.plotly_chart(
                plot_paragraph_graph(analysis["df"], "Paragraph Emotion Flow"),
                width="stretch",
            )
            show_analysis_block(analysis, "Final Paragraph Analysis")
        return

    if sections:
        render_section_heading("Section Results", "Section-wise Analysis")
        for section in sections:
            analysis = section.get("analysis")
            if not analysis:
                continue

            st.markdown(f"<h2>{html.escape(str(section.get('title', 'Section')))}</h2>", unsafe_allow_html=True)
            st.markdown('<h3 class="chart-title">Interactive Emotion Flow</h3>', unsafe_allow_html=True)
            st.plotly_chart(
                plot_paragraph_graph(analysis["df"], f"Emotion Flow for {section.get('title', 'Section')}"),
                width="stretch",
            )
            show_analysis_block(analysis, f"{section.get('title', 'Section')} Analysis")

    summary_df = build_section_summary_rows_from_context(context)
    full_analysis = (context.get("full") or {}).get("analysis")

    if not summary_df.empty and len(summary_df) > 1:
        render_section_heading("Full Text", "Peak Emotion Graph Across All Sections")
        st.plotly_chart(
            plot_peak_emotion_graph(summary_df),
            width="stretch",
        )

        highest_peak_idx = summary_df["PeakScore"].abs().idxmax()
        highest_peak_row = summary_df.loc[highest_peak_idx]
        st.markdown(f"""
<div class="peak-card">
    <h3>Highest Peak in the Entire Text</h3>
    <p><b>Section:</b> {html.escape(str(highest_peak_row['Section']))}</p>
    <p><b>Peak Emotion:</b> {html.escape(str(highest_peak_row['PeakEmotion']))} {html.escape(str(highest_peak_row['PeakEmoji']))}</p>
    <p><b>Peak Score:</b> {html.escape(str(highest_peak_row['PeakScore']))}</p>
    <p><b>Peak Paragraph Number:</b> {html.escape(str(highest_peak_row['PeakParagraphNumber']))}</p>
    <p>{html.escape(str(highest_peak_row['PeakText']))}</p>
</div>
""", unsafe_allow_html=True)

    if full_analysis is not None and mode != "single":
        show_analysis_block(full_analysis, "Entire Text Analysis")

    if not summary_df.empty and mode != "single":
        render_section_heading(
            "Readable Report",
            "Section Summary Table",
            "A clearer table view with wrapped text, section cards, and highlighted peak details.",
        )
        render_section_summary_report(summary_df)


PDF_PAGE_WIDTH = 595
PDF_PAGE_HEIGHT = 842
PDF_MARGIN = 48


def pdf_clean_text(value) -> str:
    if value is None:
        return ""

    text = html.unescape(str(value))
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)

    replacements = {
        chr(8216): "'",
        chr(8217): "'",
        chr(8220): '"',
        chr(8221): '"',
        chr(8211): "-",
        chr(8212): "-",
        chr(8230): "...",
        chr(8226): "-",
    }
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.encode("latin-1", "replace").decode("latin-1")


class PdfTextReport:
    def __init__(self, title: str):
        self.doc = fitz.open()
        self.page = None
        self.page_number = 0
        self.y = PDF_MARGIN
        self.add_page()
        self.write_heading(title, level=0)

    def add_page(self) -> None:
        self.page = self.doc.new_page(width=PDF_PAGE_WIDTH, height=PDF_PAGE_HEIGHT)
        self.page_number += 1
        self.y = PDF_MARGIN + 8
        self.page.insert_text(
            (PDF_MARGIN, 28),
            "Narrative Emotion Analyzer",
            fontsize=8,
            fontname="helv",
            color=(0.36, 0.43, 0.52),
        )
        self.page.insert_text(
            (PDF_PAGE_WIDTH - PDF_MARGIN - 46, PDF_PAGE_HEIGHT - 24),
            f"Page {self.page_number}",
            fontsize=8,
            fontname="helv",
            color=(0.36, 0.43, 0.52),
        )
        self.page.draw_line(
            (PDF_MARGIN, 36),
            (PDF_PAGE_WIDTH - PDF_MARGIN, 36),
            color=(0.84, 0.89, 0.91),
            width=0.6,
        )

    def ensure_space(self, height: float) -> None:
        if self.y + height > PDF_PAGE_HEIGHT - PDF_MARGIN:
            self.add_page()

    def write_heading(self, text: str, level: int = 1) -> None:
        sizes = {0: 20, 1: 15, 2: 12}
        colors = {0: (0.03, 0.08, 0.15), 1: (0.04, 0.23, 0.25), 2: (0.12, 0.18, 0.28)}
        size = sizes.get(level, 11)
        color = colors.get(level, (0.12, 0.18, 0.28))
        spacing = 18 if level <= 1 else 12

        if self.y > PDF_MARGIN + 18:
            self.y += 4
        self.ensure_space(size * 2.4)
        self.page.insert_text(
            (PDF_MARGIN, self.y),
            pdf_clean_text(text),
            fontsize=size,
            fontname="helv",
            color=color,
        )
        self.y += spacing
        if level <= 1:
            self.page.draw_line(
                (PDF_MARGIN, self.y),
                (PDF_PAGE_WIDTH - PDF_MARGIN, self.y),
                color=(0.87, 0.91, 0.92),
                width=0.5,
            )
            self.y += 12

    def write_text(
        self,
        text: str,
        font_size: float = 9.5,
        indent: float = 0,
        color: tuple[float, float, float] = (0.09, 0.13, 0.20),
        spacing_after: float = 7,
    ) -> None:
        text = pdf_clean_text(text)
        if not text.strip():
            self.y += spacing_after
            return

        line_height = font_size * 1.35
        max_width = PDF_PAGE_WIDTH - (PDF_MARGIN * 2) - indent
        wrap_width = max(24, int(max_width / max(4.2, font_size * 0.52)))

        for raw_line in text.split("\n"):
            if not raw_line.strip():
                self.ensure_space(line_height)
                self.y += line_height * 0.7
                continue

            wrapped_lines = textwrap.wrap(
                raw_line,
                width=wrap_width,
                break_long_words=True,
                break_on_hyphens=False,
            ) or [raw_line]

            for line in wrapped_lines:
                self.ensure_space(line_height)
                self.page.insert_text(
                    (PDF_MARGIN + indent, self.y),
                    line,
                    fontsize=font_size,
                    fontname="helv",
                    color=color,
                )
                self.y += line_height

        self.y += spacing_after

    def write_key_values(self, pairs: list[tuple[str, str]]) -> None:
        for label, value in pairs:
            if value is None or value == "":
                continue
            self.write_text(f"{label}: {value}", font_size=9.4, spacing_after=2)
        self.y += 6

    def write_bullets(self, items: list[str]) -> None:
        for item in items:
            self.write_text(f"- {item}", font_size=9.4, indent=10, spacing_after=2)
        self.y += 6

    def output(self) -> bytes:
        data = self.doc.tobytes(garbage=4, deflate=True)
        self.doc.close()
        return data


def pdf_analysis_sections(context: dict, include_full_first: bool = True) -> list[dict]:
    sections = []
    full_section = context.get("full")
    if include_full_first and full_section is not None:
        sections.append(full_section)
    sections.extend(context.get("sections", []))
    if not include_full_first and full_section is not None:
        sections.append(full_section)
    return sections


def pdf_section_rows(section: dict) -> list[dict]:
    analysis = section.get("analysis") or {}
    df = analysis.get("df")
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df.to_dict("records")
    return list(section.get("paragraphs", []))


def pdf_context_counts(context: dict) -> tuple[int, int]:
    sections = context.get("sections", [])
    paragraph_count = sum(len(section.get("paragraphs", [])) for section in sections)
    if not sections and context.get("full"):
        paragraph_count = len(context["full"].get("paragraphs", []))
    return len(sections), paragraph_count


def write_pdf_context_overview(writer: PdfTextReport, context: dict) -> None:
    section_count, paragraph_count = pdf_context_counts(context)
    writer.write_heading("Analysis Overview", level=1)
    writer.write_key_values([
        ("Input type", context.get("input_type", "Unknown")),
        ("Characters analyzed", str(context.get("character_count", 0) or 0)),
        ("Analysis mode", context.get("mode", "single")),
        ("Sections analyzed", str(section_count or 1)),
        ("Paragraphs scored", str(paragraph_count)),
    ])


def write_pdf_process_snapshot(writer: PdfTextReport, context: dict) -> None:
    engines = sorted({
        str((section.get("analysis") or {}).get("analysis_engine", "")).strip()
        for section in pdf_analysis_sections(context)
        if str((section.get("analysis") or {}).get("analysis_engine", "")).strip()
    })

    writer.write_heading("Processing Snapshot", level=1)
    writer.write_key_values([
        ("Analysis engine", ", ".join(engines) if engines else "Emotion model"),
        ("Source handling", "Uploaded or pasted text was extracted, cleaned, and normalized"),
        ("Unit of analysis", "Paragraphs"),
    ])
    writer.write_bullets([
        "Text was split into sections when section headings were detected.",
        "Each paragraph received a sentiment score, emotion label, confidence value, and emoji label.",
        "Scores were smoothed and compared with known narrative arc shapes.",
        "The chatbot answers were generated from the saved analysis context and current chat transcript.",
    ])


def write_pdf_analysis_summary(writer: PdfTextReport, section: dict, include_paragraphs: bool = False) -> None:
    title = str(section.get("title", "Analysis"))
    analysis = section.get("analysis") or {}
    shape = section.get("story_shape") or analysis.get("story_shape") or {}

    writer.write_heading(title, level=1)
    writer.write_key_values([
        ("Overall sentiment", section.get("overall_sentiment", "")),
        ("Overall valence score", format_score(section.get("overall_score", ""))),
        ("Dominant emotion", str(section.get("dominant_emotion", "")).strip()),
        ("Positive paragraphs", str(analysis.get("positive_count", ""))),
        ("Negative paragraphs", str(analysis.get("negative_count", ""))),
        ("Neutral paragraphs", str(analysis.get("neutral_count", ""))),
        ("Story shape", shape.get("name", "")),
        ("Story shape similarity", format_score(shape.get("similarity", "")) if shape else ""),
    ])

    if shape.get("description"):
        writer.write_text(f"Story shape meaning: {shape.get('description')}", font_size=9.4)

    writer.write_heading("Summary", level=2)
    writer.write_text(section.get("summary", ""), font_size=9.6)

    writer.write_heading("Peak Emotion Paragraph", level=2)
    writer.write_key_values([
        ("Peak paragraph", str(section.get("peak_paragraph_number", ""))),
        ("Peak emotion", str(section.get("peak_emotion", "")).strip()),
        ("Peak score", format_score(section.get("peak_score", ""))),
    ])
    writer.write_text(section.get("peak_text", ""), font_size=9.2, indent=12)

    if not include_paragraphs:
        return

    rows = pdf_section_rows(section)
    if not rows:
        return

    writer.write_heading("Paragraph Scoring Details", level=2)
    for row in rows:
        paragraph_label = (
            f"Paragraph {row.get('Paragraph', '')}: "
            f"{row.get('Emotion', '')} | score {format_score(row.get('Score', ''))} | "
            f"sentiment {row.get('Sentiment', '')} | confidence {format_score(row.get('EmotionConfidence', ''))}"
        )
        writer.write_text(paragraph_label, font_size=9.1, color=(0.04, 0.23, 0.25), spacing_after=2)
        writer.write_text(row.get("Text", ""), font_size=8.4, indent=14, spacing_after=6)


def write_pdf_chatbot_transcript(writer: PdfTextReport, messages: list[dict]) -> None:
    writer.write_heading("Chatbot Conversation", level=1)
    if not messages:
        writer.write_text("No chatbot conversation has been recorded yet.", font_size=9.6)
        return

    for message in messages:
        role = "You" if message.get("role") == "user" else "Story Guide"
        writer.write_text(f"{role}:", font_size=9.6, color=(0.04, 0.23, 0.25), spacing_after=2)
        writer.write_text(message.get("content", ""), font_size=9.1, indent=14, spacing_after=8)


def build_final_output_pdf(context: dict) -> bytes:
    writer = PdfTextReport("Final Emotion Analysis Output")
    write_pdf_context_overview(writer, context)
    for section in pdf_analysis_sections(context, include_full_first=True):
        write_pdf_analysis_summary(writer, section, include_paragraphs=False)
    return writer.output()


def build_complete_process_pdf(context: dict, chat_messages: list[dict]) -> bytes:
    writer = PdfTextReport("Complete Emotion Analysis Process")
    write_pdf_context_overview(writer, context)
    write_pdf_process_snapshot(writer, context)

    for section in pdf_analysis_sections(context, include_full_first=False):
        write_pdf_analysis_summary(writer, section, include_paragraphs=True)

    write_pdf_chatbot_transcript(writer, chat_messages)
    return writer.output()


def render_analysis_pdf_downloads(context: dict) -> None:
    if not context:
        return

    render_section_heading("Export", "PDF Downloads")

    if not FITZ_AVAILABLE:
        st.info("PDF export requires PyMuPDF, which is not available in this environment.")
        return

    chat_messages = list(st.session_state.get("analysis_chat_messages", []))
    suffix = str(context.get("id") or "analysis")[:10]
    final_pdf = build_final_output_pdf(context)
    process_pdf = build_complete_process_pdf(context, chat_messages)

    final_col, process_col = st.columns(2, gap="large")
    with final_col:
        st.download_button(
            label="Download Final Output PDF",
            data=final_pdf,
            file_name=f"emotion_analysis_final_output_{suffix}.pdf",
            mime="application/pdf",
            key=f"download_final_output_pdf_{suffix}",
        )

    with process_col:
        st.download_button(
            label="Download Entire Process PDF",
            data=process_pdf,
            file_name=f"emotion_analysis_entire_process_{suffix}.pdf",
            mime="application/pdf",
            key=f"download_entire_process_pdf_{suffix}",
        )


@st.fragment
def render_analysis_chatbot(context: dict) -> None:
    if not context:
        return

    if "analysis_chat_messages" not in st.session_state:
        st.session_state.analysis_chat_messages = []

    with st.container(key="story_bot_floating"):
        with st.popover("Chat", icon=":material/chat:"):
            with st.container(key="story_bot_panel"):
                with st.container(key="story_bot_book"):
                    left_page, right_page = st.columns([0.95, 1.12], gap="medium")

                    with left_page:
                        with st.container(key="story_bot_left_page"):
                            st.markdown(f"""
<div class="story-book-page-heading">
    <span>Left page</span>
    <h3>Story Guide</h3>
</div>
<div class="story-book-note">
Use this side like a reader's margin: choose a prepared question, then the answer appears on the right page.
</div>
""", unsafe_allow_html=True)

                            st.markdown('<div class="story-bot-quick-title">Quick questions</div>', unsafe_allow_html=True)
                            with st.container(key="story_bot_quick_questions"):
                                for row_start in range(0, len(QUICK_CHAT_QUESTIONS), 2):
                                    cols = st.columns(2, gap="small")
                                    for offset, col in enumerate(cols):
                                        index = row_start + offset
                                        if index >= len(QUICK_CHAT_QUESTIONS):
                                            continue
                                        label, question = QUICK_CHAT_QUESTIONS[index]
                                        with col:
                                            if st.button(label, key=f"story_bot_quick_{index}", width="stretch"):
                                                answer = answer_analysis_question(question, context)
                                                st.session_state.analysis_chat_messages.append({"role": "user", "content": question})
                                                st.session_state.analysis_chat_messages.append({"role": "assistant", "content": answer})

                    with right_page:
                        with st.container(key="story_bot_right_page"):
                            st.markdown("""
<div class="story-book-page-heading">
    <span>Right page</span>
    <h3>Conversation</h3>
</div>
""", unsafe_allow_html=True)

                            if st.session_state.analysis_chat_messages:
                                if st.button("Clear page", key="story_bot_clear", width="stretch"):
                                    st.session_state.analysis_chat_messages = []

                            messages_box = st.container(height=310, key="story_bot_messages")

                            with st.form("analysis_popup_chat_form", clear_on_submit=True):
                                prompt = st.text_input(
                                    "Ask a question",
                                    label_visibility="collapsed",
                                    placeholder="Ask about a theme, peak, plot moment, chapter, or emotion...",
                                )
                                submitted = st.form_submit_button("Ask Story Guide", width="stretch")

                            if submitted and prompt.strip():
                                answer = answer_analysis_question(prompt, context)
                                st.session_state.analysis_chat_messages.append({"role": "user", "content": prompt.strip()})
                                st.session_state.analysis_chat_messages.append({"role": "assistant", "content": answer})

                            messages = st.session_state.analysis_chat_messages
                            with messages_box:
                                if not messages:
                                    st.markdown(
                                        '<p class="story-bot-empty">This page will hold your questions and the guide’s answers.</p>',
                                        unsafe_allow_html=True,
                                    )
                                for message in messages:
                                    label = "You" if message["role"] == "user" else "Story Guide"
                                    label_class = "story-bot-user" if message["role"] == "user" else ""
                                    st.markdown(
                                        f'<div class="story-bot-speaker {label_class}">{html.escape(label)}</div>',
                                        unsafe_allow_html=True,
                                    )
                                    st.markdown(message["content"])

    render_analysis_pdf_downloads(context)
    inject_story_bot_drag_script()


def inject_story_bot_drag_script() -> None:
    components.html(
        """
<script>
(function () {
    const parentDoc = window.parent && window.parent.document ? window.parent.document : document;
    const parentWin = window.parent || window;
    const storageKey = "storyInsightBotPosition";

    function clamp(value, min, max) {
        return Math.min(Math.max(value, min), max);
    }

    function getSavedPosition() {
        try {
            return JSON.parse(parentWin.localStorage.getItem(storageKey) || "null");
        } catch (error) {
            return null;
        }
    }

    function savePosition(left, top) {
        try {
            parentWin.localStorage.setItem(storageKey, JSON.stringify({ left: left, top: top }));
        } catch (error) {}
    }

    function setup() {
        const bot = parentDoc.querySelector(".st-key-story_bot_floating");
        if (!bot || bot.dataset.dragReady === "true") {
            return;
        }

        bot.dataset.dragReady = "true";
        const saved = getSavedPosition();
        if (saved && Number.isFinite(saved.left) && Number.isFinite(saved.top)) {
            bot.style.left = clamp(saved.left, 8, parentWin.innerWidth - 90) + "px";
            bot.style.top = clamp(saved.top, 8, parentWin.innerHeight - 90) + "px";
            bot.style.right = "auto";
            bot.style.bottom = "auto";
        }

        const handle = bot.querySelector('[data-testid="stPopover"] button') || bot.querySelector("button");
        if (!handle) {
            return;
        }

        let lastExpanded = null;
        let animationTimer = null;

        function playBookAnimation(expanded) {
            if (lastExpanded === null) {
                lastExpanded = expanded;
                return;
            }

            if (expanded === lastExpanded) {
                return;
            }

            bot.classList.remove("story-bot-opening", "story-bot-closing");
            void bot.offsetWidth;
            bot.classList.add(expanded ? "story-bot-opening" : "story-bot-closing");
            clearTimeout(animationTimer);
            animationTimer = setTimeout(function () {
                bot.classList.remove("story-bot-opening", "story-bot-closing");
            }, expanded ? 620 : 560);
            lastExpanded = expanded;
        }

        function refreshBookState() {
            const panelOpen = Boolean(parentDoc.querySelector(".st-key-story_bot_panel"));
            const expanded = handle.getAttribute("aria-expanded") === "true" || panelOpen;
            playBookAnimation(expanded);
            bot.classList.toggle("story-bot-open", expanded);
            handle.removeAttribute("title");
            handle.setAttribute("aria-label", expanded ? "Close Story Guide" : "Open Story Guide");
        }

        refreshBookState();
        const observer = new MutationObserver(refreshBookState);
        observer.observe(handle, { attributes: true, attributeFilter: ["aria-expanded", "data-state", "class"] });
        parentDoc.addEventListener("click", function () {
            setTimeout(refreshBookState, 40);
        }, true);
        setInterval(refreshBookState, 500);

        let start = null;
        let suppressClick = false;

        handle.addEventListener("pointerdown", function (event) {
            if (event.button !== 0) {
                return;
            }
            const rect = bot.getBoundingClientRect();
            start = {
                pointerId: event.pointerId,
                x: event.clientX,
                y: event.clientY,
                left: rect.left,
                top: rect.top,
                moved: false
            };
            try {
                handle.setPointerCapture(event.pointerId);
            } catch (error) {}
        });

        handle.addEventListener("pointermove", function (event) {
            if (!start) {
                return;
            }
            const dx = event.clientX - start.x;
            const dy = event.clientY - start.y;
            if (Math.abs(dx) + Math.abs(dy) < 6 && !start.moved) {
                return;
            }

            start.moved = true;
            suppressClick = true;
            event.preventDefault();

            const nextLeft = clamp(start.left + dx, 8, parentWin.innerWidth - bot.offsetWidth - 8);
            const nextTop = clamp(start.top + dy, 8, parentWin.innerHeight - bot.offsetHeight - 8);
            bot.style.left = nextLeft + "px";
            bot.style.top = nextTop + "px";
            bot.style.right = "auto";
            bot.style.bottom = "auto";
        });

        handle.addEventListener("pointerup", function () {
            if (start && start.moved) {
                const rect = bot.getBoundingClientRect();
                savePosition(rect.left, rect.top);
            }
            start = null;
            setTimeout(function () {
                suppressClick = false;
            }, 80);
        });

        handle.addEventListener("click", function (event) {
            if (suppressClick) {
                event.preventDefault();
                event.stopImmediatePropagation();
            }
        }, true);
    }

    setup();
    setTimeout(setup, 300);
})();
</script>
        """,
        height=0,
    )


# ----------------------------
# Core text helpers
# ----------------------------
def classify_sentiment(score: float) -> str:
    if score >= 0.5:
        return "strongly positive"
    elif score > 0.05:
        return "positive"
    elif score <= -0.5:
        return "strongly negative"
    elif score < -0.05:
        return "negative"
    return "neutral"


def emotion_emoji(emotion: str) -> str:
    clean_mapping = {
        "joy": "\U0001f60a",
        "sadness": "\U0001f622",
        "fear": "\U0001f628",
        "anger": "\U0001f620",
        "disgust": "\U0001f922",
        "surprise": "\U0001f632",
        "wonder": "\u2728",
        "neutral": "\U0001f610",
        "mixed": "\U0001f3ad",
    }
    return clean_mapping.get(emotion, "\U0001f610")

    mapping = {
        "joy": "😊",
        "sadness": "😢",
        "fear": "😨",
        "anger": "😠",
        "disgust": "🤢",
        "surprise": "😲",
        "wonder": "✨",
        "neutral": "😐",
        "mixed": "🎭",
    }
    return mapping.get(emotion, "😐")


SECTION_NUMBER_WORD_PATTERN = (
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty(?:[-\s]+one|[-\s]+two|[-\s]+three|[-\s]+four|[-\s]+five|"
    r"[-\s]+six|[-\s]+seven|[-\s]+eight|[-\s]+nine)?|"
    r"thirty(?:[-\s]+one|[-\s]+two|[-\s]+three|[-\s]+four|[-\s]+five|"
    r"[-\s]+six|[-\s]+seven|[-\s]+eight|[-\s]+nine)?|"
    r"forty(?:[-\s]+one|[-\s]+two|[-\s]+three|[-\s]+four|[-\s]+five|"
    r"[-\s]+six|[-\s]+seven|[-\s]+eight|[-\s]+nine)?|"
    r"fifty(?:[-\s]+one|[-\s]+two|[-\s]+three|[-\s]+four|[-\s]+five|"
    r"[-\s]+six|[-\s]+seven|[-\s]+eight|[-\s]+nine)?"
)
SECTION_NUMBER_PATTERN = rf"(?:\d{{1,3}}|[ivxlcdm]+|{SECTION_NUMBER_WORD_PATTERN})"
NUMBERED_SECTION_LABEL_PATTERN = r"(?:letter|chapter|part|book|volume)"
FRONT_BACK_MATTER_PATTERN = (
    r"acknowledg(?:e)?ments?|conclusion|context|appendix(?:\s+[a-z0-9ivxlcdm]+)?|"
    r"prologue|epilogue|preface|introduction|afterword|contents|"
    r"table\s+of\s+contents|note\s+from\s+the\s+author|author'?s?\s+note"
)
SECTION_HEADING_PATTERN = (
    rf"(?:{NUMBERED_SECTION_LABEL_PATTERN})\s+{SECTION_NUMBER_PATTERN}"
    rf"(?:\s*[:.\-]\s*.+)?\.?"
    rf"|(?:{FRONT_BACK_MATTER_PATTERN})(?:\s*[:.\-]\s*.+)?\.?"
)
ANALYSIS_TARGET_WORDS = 100
MIN_ANALYSIS_WORDS = 12
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
ZIP_TEXT_EXTENSIONS = (".txt", ".pdf", ".docx")
ZIP_SUPPORTED_EXTENSIONS = ZIP_TEXT_EXTENSIONS + IMAGE_EXTENSIONS


def is_section_heading(line: str) -> bool:
    normalized = re.sub(r"\s+", " ", clean_extracted_text(line)).strip(" .")
    return bool(re.fullmatch(SECTION_HEADING_PATTERN, normalized, flags=re.IGNORECASE))


def normalize_section_title(line: str) -> str:
    normalized = re.sub(r"\s+", " ", clean_extracted_text(line)).strip(" .")
    return normalized.title()


def clean_extracted_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = text.replace("\x0c", "\n")
    text = text.replace("\x7f", "\u2022")

    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "â€œ": '"',
        "â€": '"',
        "â€™": "'",
        "â€˜": "'",
        "â€“": "-",
        "â€”": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_lines_for_paragraphs(text: str) -> str:
    text = clean_extracted_text(text)
    lines = text.split("\n")

    rebuilt_paragraphs = []
    current = []

    def flush_current():
        nonlocal current, rebuilt_paragraphs
        if current:
            paragraph = " ".join(current)
            paragraph = re.sub(r"\s+", " ", paragraph).strip()
            if paragraph:
                rebuilt_paragraphs.append(paragraph)
            current = []

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            flush_current()
            rebuilt_paragraphs.append("")
            continue

        if is_section_heading(line):
            flush_current()
            rebuilt_paragraphs.append(normalize_section_title(line))
            rebuilt_paragraphs.append("")
            continue

        if current:
            prev = current[-1]
            if not re.search(r"""[.!?:;"'\)\]]$""", prev):
                current.append(line)
            else:
                if re.match(r"""^[A-Z"']""", line):
                    flush_current()
                    current.append(line)
                else:
                    current.append(line)
        else:
            current.append(line)

    flush_current()

    text = "\n".join(rebuilt_paragraphs)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_sentences(text: str) -> list[str]:
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def normalize_text_for_analysis_units(text: str) -> str:
    lines = normalize_lines_for_paragraphs(text).splitlines()
    content_lines = []

    for line in lines:
        line = line.strip()
        if not line or is_section_heading(line):
            continue
        content_lines.append(line)

    return re.sub(r"\s+", " ", " ".join(content_lines)).strip()


def split_into_analysis_units(text: str) -> list[str]:
    text = normalize_text_for_analysis_units(text)
    if not text:
        return []

    sentences = split_into_sentences(text)
    if not sentences:
        return []

    units = []
    current = []
    current_words = 0

    def flush_current():
        nonlocal current, current_words
        if not current:
            return

        unit = re.sub(r"\s+", " ", " ".join(current)).strip()
        if len(unit.split()) >= MIN_ANALYSIS_WORDS:
            units.append(unit)
        elif units:
            units[-1] = f"{units[-1]} {unit}".strip()
        elif unit:
            units.append(unit)
        current = []
        current_words = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current and current_words + word_count > ANALYSIS_TARGET_WORDS:
            flush_current()

        current.append(sentence)
        current_words += word_count

    flush_current()
    return units


def split_into_paragraphs(text: str) -> list[str]:
    return split_into_analysis_units(text)


def detect_section_title_from_text(text: str) -> str:
    lines = [line.strip() for line in normalize_lines_for_paragraphs(text).splitlines() if line.strip()]
    for line in lines[:12]:
        if is_section_heading(line):
            return normalize_section_title(line)
    return "Pasted Section"


def section_has_body_text(section_text: str) -> bool:
    for line in section_text.splitlines():
        line = line.strip()
        if line and not is_section_heading(line):
            return True
    return False


def split_book_sections(text: str) -> list[tuple[str, str]]:
    text = normalize_lines_for_paragraphs(text)
    pattern = rf"(?im)^({SECTION_HEADING_PATTERN})\s*$"
    matches = list(re.finditer(pattern, text))

    if not matches:
        return [("Full Text", text)]

    sections = []
    for i, match in enumerate(matches):
        title = normalize_section_title(match.group(1))
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text and section_has_body_text(section_text):
            sections.append((title, section_text))

    return sections if sections else [("Full Text", text)]


def looks_like_section_text(text: str) -> bool:
    paragraphs = split_into_paragraphs(text)
    word_count = len(text.split())
    return len(paragraphs) >= 4 or word_count >= 800


# ----------------------------
# Emotion logic
# ----------------------------
def detect_emotion(score: float, text: str = "") -> str:
    t = text.lower()

    wonder_words = [
        "wonder", "wonders", "wonderful", "beauty", "beautiful", "splendour",
        "splendor", "glory", "magnificent", "radiant", "marvel", "marvellous",
        "marvelous", "awe", "astonishment", "eternal light", "surpassing",
        "sublime", "vision", "unknown", "discovery"
    ]
    joy_words = [
        "joy", "happy", "delight", "excited", "hope", "bright", "confidence",
        "pleasure", "cheerful", "success", "rejoice", "warmth", "enthusiasm",
        "gratitude", "blessings", "kindness", "love", "affection"
    ]
    sadness_words = [
        "sad", "sorrow", "grief", "lonely", "miserable", "weep", "despair",
        "melancholy", "pain", "suffering", "depressed", "farewell"
    ]
    fear_words = [
        "fear", "afraid", "terrified", "scared", "horror", "dread", "danger",
        "death", "desolation", "darkness", "threat", "fail", "never", "uncertain"
    ]
    anger_words = [
        "angry", "anger", "rage", "furious", "shouting", "yelling", "hatred",
        "fury", "revenge", "detestation", "contempt"
    ]
    disgust_words = [
        "disgust", "rotten", "filthy", "gross", "horrible smell", "nausea",
        "hideous", "loathing"
    ]
    surprise_words = [
        "surprised", "surprise", "shocked", "unexpected", "suddenly",
        "startled", "astonished"
    ]   

    def has_any(words: list[str]) -> bool:
        for word in words:
            if " " in word:
                if word in t:
                    return True
            elif re.search(rf"\b{re.escape(word)}\b", t):
                return True
        return False

    if has_any(disgust_words):
        return "disgust"
    if has_any(anger_words):
        return "anger"
    if has_any(fear_words):
        return "fear"
    if has_any(wonder_words):
        return "wonder"
    if has_any(surprise_words):
        return "surprise"
    if has_any(sadness_words):
        return "sadness"
    if has_any(joy_words):
        return "joy"

    if score >= 0.65:
        return "joy"
    if score > 0.05:
        return "neutral"
    if score <= -0.5:
        return "fear"
    if score < -0.05:
        return "sadness"
    return "neutral"


def normalize_emotion_scores(scores: dict) -> dict:
    cleaned = {str(label).lower().strip(): max(0.0, float(value)) for label, value in scores.items()}
    total = sum(cleaned.values())
    if total <= 0:
        return {"neutral": 1.0}
    return {label: value / total for label, value in cleaned.items()}


def fallback_emotion_scores(text: str) -> dict:
    score = sia.polarity_scores(text)["compound"]
    emotion = detect_emotion(score, text)
    scores = {label: 0.0 for label in EMOTION_LABELS}
    scores[emotion] = 0.74

    if score > 0.35:
        scores["joy"] += min(0.2, score * 0.2)
    elif score < -0.35:
        scores["sadness"] += min(0.14, abs(score) * 0.14)
        scores["fear"] += min(0.1, abs(score) * 0.1)
    else:
        scores["neutral"] += 0.18

    return normalize_emotion_scores(scores)


def classify_emotion_scores(text: str, max_words: int = 220) -> tuple[dict, str]:
    compact = re.sub(r"\s+", " ", clean_extracted_text(str(text))).strip()
    if not compact:
        return {"neutral": 1.0}, "empty"

    words = compact.split()
    clipped = " ".join(words[:max_words])
    classifier = load_emotion_classifier()

    if classifier is None:
        return fallback_emotion_scores(compact), "keyword/VADER fallback"

    try:
        results = classifier(clipped, truncation=True)
        if results and isinstance(results[0], list):
            results = results[0]

        scores = {}
        for item in results:
            label = str(item.get("label", "")).lower().strip()
            value = float(item.get("score", 0.0))
            if label:
                scores[label] = value

        if scores:
            return normalize_emotion_scores(scores), HF_EMOTION_MODEL
    except Exception:
        pass

    return fallback_emotion_scores(compact), "keyword/VADER fallback"


def emotion_from_scores(scores: dict) -> tuple[str, float]:
    if not scores:
        return "neutral", 0.0

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    emotion, confidence = ranked[0]
    if confidence < 0.28:
        return "neutral", float(confidence)
    return str(emotion), float(confidence)


def score_from_emotion_scores(scores: dict) -> float:
    normalized = normalize_emotion_scores(scores)
    score = sum(EMOTION_VALENCE.get(label, 0.0) * value for label, value in normalized.items())
    return round(max(-1.0, min(1.0, score)), 4)


def get_dominant_emotion(df: pd.DataFrame) -> str:
    if "EmotionConfidence" in df.columns:
        weights = df.groupby("Emotion")["EmotionConfidence"].sum().to_dict()
        emotion_counts = weights
        close_gap = 0.12
    else:
        emotion_counts = df["Emotion"].value_counts().to_dict()
        close_gap = 1.0

    non_neutral_counts = {k: v for k, v in emotion_counts.items() if k != "neutral"}

    if not non_neutral_counts:
        return "neutral"

    sorted_emotions = sorted(non_neutral_counts.items(), key=lambda x: x[1], reverse=True)

    if len(sorted_emotions) >= 2:
        top_emotion, top_count = sorted_emotions[0]
        second_emotion, second_count = sorted_emotions[1]
        threshold = close_gap * max(1.0, float(top_count))
        if top_count - second_count <= threshold and top_emotion != second_emotion:
            return "mixed"

    return sorted_emotions[0][0]


def resample_series(values: list[float] | np.ndarray, target_length: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if target_length <= 0:
        return np.asarray([], dtype=float)
    if len(values) == 0:
        return np.zeros(target_length, dtype=float)
    if len(values) == 1:
        return np.full(target_length, float(values[0]), dtype=float)

    source_x = np.linspace(0.0, 1.0, len(values))
    target_x = np.linspace(0.0, 1.0, target_length)
    return np.interp(target_x, source_x, values)


def smooth_scores(scores: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if len(values) < 3:
        return values

    window = min(5, len(values))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return values

    pad = window // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def centered_unit(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr - float(arr.mean())
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-9:
        return np.zeros_like(arr)
    return arr / norm


def scale_template_to_arc(template: list[float], smooth: np.ndarray, length: int) -> np.ndarray:
    shape = resample_series(template, length)
    if len(shape) == 0:
        return shape

    shape_std = float(shape.std())
    arc_std = float(smooth.std()) if len(smooth) else 0.0
    arc_mean = float(smooth.mean()) if len(smooth) else 0.0

    if shape_std <= 1e-9 or arc_std <= 1e-9:
        return np.full(length, arc_mean, dtype=float)

    scaled = ((shape - float(shape.mean())) / shape_std) * arc_std + arc_mean
    return np.clip(scaled, -1.0, 1.0)


def match_story_shape(scores: list[float]) -> dict:
    values = np.asarray(scores, dtype=float)
    smooth = smooth_scores(values)

    if len(values) < 3:
        return {
            "name": "Needs More Text",
            "similarity": 0.0,
            "description": "At least three analysis units are needed to compare a story shape.",
            "candidates": [],
            "smooth_values": smooth.tolist(),
            "shape_values": smooth.tolist(),
        }

    arc = centered_unit(resample_series(smooth, 72))
    candidates = []
    for name, shape in STORY_SHAPES.items():
        template = centered_unit(resample_series(shape["points"], 72))
        similarity = float(np.dot(arc, template)) if np.linalg.norm(arc) > 0 else 0.0
        candidates.append({
            "name": name,
            "similarity": round(similarity, 4),
            "description": shape["description"],
        })

    candidates = sorted(candidates, key=lambda item: item["similarity"], reverse=True)
    best = candidates[0]
    shape_values = scale_template_to_arc(STORY_SHAPES[best["name"]]["points"], smooth, len(values))

    return {
        "name": best["name"],
        "similarity": best["similarity"],
        "description": best["description"],
        "candidates": candidates,
        "smooth_values": [round(float(value), 4) for value in smooth],
        "shape_values": [round(float(value), 4) for value in shape_values],
    }


def get_keywords(text: str) -> dict:
    t = text.lower()
    return {
        "has_hope": any(word in t for word in ["hope", "promise", "confidence", "future", "success"]),
        "has_wonder": any(word in t for word in [
            "wonder", "beauty", "splendour", "splendor", "glory", "magnificent",
            "marvel", "radiant", "eternal light", "wonders", "sublime"
        ]),
        "has_ambition": any(word in t for word in [
            "discover", "voyage", "undertaking", "curiosity", "attain", "benefit",
            "passage", "generation", "magnet", "unknown", "unvisited", "enterprise"
        ]),
        "has_fear": any(word in t for word in [
            "fear", "danger", "desolation", "frost", "death", "fail", "never", "uncertain"
        ]),
        "has_affection": any(word in t for word in [
            "love", "kindness", "gratitude", "blessings", "dear", "sister", "affection"
        ]),
        "has_farewell": any(word in t for word in ["farewell", "again", "return", "meet", "never"]),
        "has_humanity": any(word in t for word in ["mankind", "humanity", "generation", "benefit"]),
    }


def build_writer_style_summary_from_paragraphs(paragraphs: list[str], df: pd.DataFrame, overall_sentiment: str) -> str:
    if not paragraphs:
        return ""

    full_text = " ".join(paragraphs)
    low_text = full_text.lower()
    dominant_emotion = get_dominant_emotion(df)

    start_text = paragraphs[0].lower()
    end_text = paragraphs[-1].lower()

    start_keys = get_keywords(start_text)
    end_keys = get_keywords(end_text)
    overall_keys = get_keywords(low_text)

    if overall_keys["has_affection"] and overall_keys["has_farewell"]:
        summary = (
            "This section moves from reflection into a more personal and emotional closing. "
            "The writing combines uncertainty and separation with affection and gratitude. "
            "By the end, the emotional tone becomes intimate, vulnerable, and heartfelt."
        )
    elif dominant_emotion in ["joy", "wonder"] and overall_keys["has_wonder"]:
        summary = (
            "This section is shaped by wonder, discovery, and imaginative energy. "
            "Its language presents the narrative as expansive and forward-looking. "
            "The emotional tone feels elevated, curious, and full of possibility."
        )
    elif dominant_emotion == "fear":
        summary = (
            "This section is driven by unease, danger, and emotional strain. "
            "Its language creates a tense atmosphere in which uncertainty and threat remain close at hand. "
            "The overall effect is dark, uneasy, and psychologically heavy."
        )
    elif dominant_emotion == "sadness":
        summary = (
            "This section carries a sorrowful and reflective emotional tone. "
            "Its language emphasizes loss, grief, and inward suffering. "
            "Rather than simple narration, it presents the emotional aftermath of painful events."
        )
    elif dominant_emotion == "anger":
        summary = (
            "This section is marked by emotional intensity, hostility, and force. "
            "The language suggests confrontation, resentment, or moral outrage. "
            "Its tone feels heated and emotionally charged."
        )
    elif dominant_emotion == "mixed":
        summary = (
            "This section blends more than one emotional current rather than settling into a single mood. "
            "Hope, fear, reflection, and feeling interact across the passage. "
            "That layered movement gives the writing a more human and literary texture."
        )
    else:
        summary = (
            f"This section is overall {overall_sentiment} and mainly shaped by {dominant_emotion}. "
            "Its emotional movement develops gradually through reflection and description. "
            "The result is a clear narrative mood with literary depth."
        )

    if overall_keys["has_humanity"] and "humanity" not in summary.lower():
        summary += " It also broadens personal feeling into a larger human meaning."

    if start_keys["has_wonder"] and end_keys["has_affection"] and "shifts" not in summary.lower():
        summary += " The emotional movement also shifts from outward observation to inward feeling."

    return summary


# ----------------------------
# Extraction helpers
# ----------------------------
def read_uploaded_bytes(uploaded_file) -> bytes:
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    data = uploaded_file.read()

    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    return data


def decode_text_bytes(file_bytes: bytes) -> str:
    if file_bytes.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in file_bytes[:200]:
        for encoding in ("utf-16", "utf-16-le", "utf-16-be"):
            try:
                return file_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue

    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue

    return file_bytes.decode("utf-8", errors="replace")


def normalize_artifact_line(line: str) -> str:
    return re.sub(r"\s+", " ", clean_extracted_text(line)).strip()


def is_page_number_line(line: str) -> bool:
    normalized = normalize_artifact_line(line)
    return bool(re.fullmatch(r"(?:page\s*)?\d{1,4}", normalized, flags=re.IGNORECASE))


def remove_repeated_page_artifacts(page_texts: list[str]) -> list[str]:
    if len(page_texts) < 2:
        return page_texts

    candidate_counts = Counter()
    page_line_lists = [page_text.splitlines() for page_text in page_texts]

    for lines in page_line_lists:
        non_empty_indices = [i for i, line in enumerate(lines) if normalize_artifact_line(line)]
        candidate_indices = set(non_empty_indices[:3] + non_empty_indices[-3:])

        for index in candidate_indices:
            normalized = normalize_artifact_line(lines[index])
            if normalized and (len(normalized) <= 80 or is_page_number_line(normalized)):
                candidate_counts[normalized.lower()] += 1

    threshold = max(2, math.ceil(len(page_texts) * 0.5))
    repeated_artifacts = {
        line for line, count in candidate_counts.items()
        if count >= threshold
    }

    cleaned_pages = []
    for lines in page_line_lists:
        non_empty_indices = [i for i, line in enumerate(lines) if normalize_artifact_line(line)]
        candidate_indices = set(non_empty_indices[:3] + non_empty_indices[-3:])
        kept_lines = []

        for index, line in enumerate(lines):
            normalized = normalize_artifact_line(line)
            should_drop = False

            if index in candidate_indices:
                should_drop = normalized.lower() in repeated_artifacts or is_page_number_line(normalized)

            if not should_drop:
                kept_lines.append(line)

        cleaned_pages.append("\n".join(kept_lines).strip())

    return cleaned_pages


def prepare_text_for_analysis(text: str) -> str:
    return normalize_lines_for_paragraphs(text)


def compact_for_comparison(text: str) -> str:
    text = prepare_text_for_analysis(text)
    return re.sub(r"\s+", " ", text).strip()


def first_difference_preview(left: str, right: str, radius: int = 90) -> str:
    limit = min(len(left), len(right))
    index = 0

    while index < limit and left[index] == right[index]:
        index += 1

    if index == limit and len(left) == len(right):
        return ""

    start = max(0, index - radius)
    end = index + radius
    left_preview = left[start:end]
    right_preview = right[start:end]
    return f"First difference near character {index}. File: {left_preview!r} | Pasted: {right_preview!r}"


def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    image = image.convert("L")
    if min(image.size) < 1600:
        scale = max(1, math.ceil(1600 / min(image.size)))
        image = image.resize((image.width * scale, image.height * scale))
    image = ImageOps.autocontrast(image)
    image = image.filter(ImageFilter.SHARPEN)
    return image


def extract_text_from_txt(uploaded_file) -> str:
    return decode_text_bytes(read_uploaded_bytes(uploaded_file))


def extract_text_from_docx(uploaded_file) -> str:
    if not DOCX_AVAILABLE:
        return ""
    doc = Document(io.BytesIO(read_uploaded_bytes(uploaded_file)))
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paras)


def extract_text_from_image(uploaded_file) -> str:
    if not OCR_AVAILABLE or not TESSERACT_AVAILABLE:
        return ""
    image = Image.open(io.BytesIO(read_uploaded_bytes(uploaded_file)))
    image = preprocess_image_for_ocr(image)
    return pytesseract.image_to_string(image, config="--oem 3 --psm 6")


def extract_text_from_pdf(uploaded_file) -> str:
    pdf_bytes = read_uploaded_bytes(uploaded_file)

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            page_text = clean_extracted_text(page_text)
            if page_text:
                pages_text.append(page_text)

        pages_text = remove_repeated_page_artifacts(pages_text)
        final_text = "\n\n".join(page for page in pages_text if page)
        if final_text.strip():
            return prepare_text_for_analysis(final_text)
    except Exception:
        pass

    if FITZ_AVAILABLE:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            all_pages = []

            for page in doc:
                blocks = page.get_text("blocks")
                blocks = sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))

                page_parts = []
                for block in blocks:
                    block_text = block[4].strip()
                    if not block_text:
                        continue

                    block_text = clean_extracted_text(block_text)
                    if block_text:
                        page_parts.append(block_text)

                page_text = "\n\n".join(page_parts).strip()

                if len(page_text) < 50 and OCR_AVAILABLE:
                    pix = page.get_pixmap(dpi=300)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    img = preprocess_image_for_ocr(img)
                    page_text = pytesseract.image_to_string(img, config="--oem 3 --psm 6")
                    page_text = clean_extracted_text(page_text)

                if page_text:
                    all_pages.append(page_text)

            all_pages = remove_repeated_page_artifacts(all_pages)
            final_text = "\n\n".join(page for page in all_pages if page)
            return prepare_text_for_analysis(final_text)

        except Exception:
            pass

    return ""


def is_image_file(uploaded_file) -> bool:
    return uploaded_file.name.lower().endswith(IMAGE_EXTENSIONS)


def natural_sort_key(value: str) -> list:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def make_named_bytes_file(name: str, data: bytes):
    file_obj = io.BytesIO(data)
    file_obj.name = name
    return file_obj


def image_ocr_unavailable_message() -> str:
    if not OCR_AVAILABLE:
        return "OCR is not available because pillow or pytesseract is not installed."
    if not TESSERACT_AVAILABLE:
        return (
            "OCR is not available because the Tesseract OCR engine was not found. "
            "Install Tesseract OCR or set TESSERACT_CMD to its tesseract.exe path, then restart the app."
        )
    return ""


def extract_text_from_image_files(uploaded_files: list) -> tuple[str, str]:
    unavailable_message = image_ocr_unavailable_message()
    if unavailable_message:
        return "", unavailable_message

    extracted_parts = []
    warning_parts = []

    for index, uploaded_file in enumerate(uploaded_files, start=1):
        file_label = uploaded_file.name or f"image {index}"
        try:
            text = prepare_text_for_analysis(extract_text_from_image(uploaded_file))
        except Exception as exc:
            warning_parts.append(f"Could not run OCR on {file_label}: {exc}")
            continue

        if text.strip():
            extracted_parts.append(text)
        else:
            warning_parts.append(
                f"No readable text was found in {file_label}. Try a clearer, higher-resolution image."
            )

    combined_text = prepare_text_for_analysis("\n\n".join(extracted_parts))
    return combined_text, " ".join(warning_parts)


def extract_text_from_zip(uploaded_file) -> tuple[str, int, str]:
    try:
        zip_bytes = read_uploaded_bytes(uploaded_file)
        archive = ZipFile(io.BytesIO(zip_bytes))
    except BadZipFile:
        return "", 0, "This ZIP file could not be opened."

    extracted_parts = []
    image_files = []
    warning_parts = []
    processed_count = 0

    with archive:
        members = [
            info for info in archive.infolist()
            if not info.is_dir()
            and not info.filename.startswith("__MACOSX/")
            and info.filename.lower().endswith(ZIP_SUPPORTED_EXTENSIONS)
        ]
        members = sorted(members, key=lambda info: natural_sort_key(info.filename))

        if not members:
            return "", 0, "No supported TXT, PDF, DOCX, JPG, PNG, or WebP files were found inside this ZIP."

        for info in members:
            file_name = os.path.basename(info.filename)
            file_name_lower = file_name.lower()
            member_file = make_named_bytes_file(file_name, archive.read(info))

            if file_name_lower.endswith(".txt"):
                extracted_parts.append(prepare_text_for_analysis(extract_text_from_txt(member_file)))
                processed_count += 1
            elif file_name_lower.endswith(".pdf"):
                text = prepare_text_for_analysis(extract_text_from_pdf(member_file))
                if text.strip():
                    extracted_parts.append(text)
                else:
                    warning_parts.append(f"Could not extract readable text from {file_name}.")
                processed_count += 1
            elif file_name_lower.endswith(".docx"):
                if DOCX_AVAILABLE:
                    extracted_parts.append(prepare_text_for_analysis(extract_text_from_docx(member_file)))
                else:
                    warning_parts.append(f"Could not read {file_name} because python-docx is not installed.")
                processed_count += 1
            elif file_name_lower.endswith(IMAGE_EXTENSIONS):
                image_files.append(member_file)

    if image_files:
        image_text, image_warning = extract_text_from_image_files(image_files)
        if image_text.strip():
            extracted_parts.append(image_text)
        if image_warning:
            warning_parts.append(image_warning)
        processed_count += len(image_files)

    combined_text = prepare_text_for_analysis("\n\n".join(part for part in extracted_parts if part.strip()))
    return combined_text, processed_count, " ".join(warning_parts)


def normalize_uploaded_files(uploaded_file) -> list:
    if uploaded_file is None:
        return []
    if isinstance(uploaded_file, list):
        return [file for file in uploaded_file if file is not None]
    return [uploaded_file]


def get_text_from_input(uploaded_file, pasted_text: str) -> tuple[str, str, str]:
    uploaded_files = normalize_uploaded_files(uploaded_file)

    if uploaded_files:
        if len(uploaded_files) > 1:
            if not all(is_image_file(file) for file in uploaded_files):
                return "", "Multiple files", "Multiple upload is supported for image files only."

            text, warning = extract_text_from_image_files(uploaded_files)
            return text, f"Image files ({len(uploaded_files)})", warning

        uploaded_file = uploaded_files[0]
        file_name = uploaded_file.name.lower()

        if file_name.endswith(".txt"):
            return prepare_text_for_analysis(extract_text_from_txt(uploaded_file)), "TXT file", ""

        if file_name.endswith(".pdf"):
            text = prepare_text_for_analysis(extract_text_from_pdf(uploaded_file))
            warning = ""
            if not text.strip():
                warning = "Could not extract readable text from this PDF. If it is a scanned PDF, OCR support may be missing or Tesseract may not be installed."
            return text, "PDF file", warning

        if file_name.endswith(".docx"):
            text = prepare_text_for_analysis(extract_text_from_docx(uploaded_file))
            warning = ""
            if not DOCX_AVAILABLE:
                warning = "python-docx is not installed, so .docx extraction is unavailable."
            return text, "DOCX file", warning

        if file_name.endswith(".doc"):
            return "", "DOC file", "Legacy .doc files are not reliably supported here. Please convert the file to .docx and upload again."

        if is_image_file(uploaded_file):
            text, warning = extract_text_from_image_files([uploaded_file])
            return text, "Image file", warning

        if file_name.endswith(".zip"):
            text, processed_count, warning = extract_text_from_zip(uploaded_file)
            if not text.strip() and not warning:
                warning = "No readable text could be extracted from the files inside this ZIP."
            return text, f"ZIP file ({processed_count} files)", warning

        return "", "Unsupported file", "Unsupported file format."

    if pasted_text.strip():
        return prepare_text_for_analysis(pasted_text.strip()), "Pasted text", ""

    return "", "No input", ""


# ----------------------------
# Analysis
# ----------------------------
def analyze_paragraphs(paragraphs: list[str]) -> pd.DataFrame:
    rows = []
    for i, paragraph in enumerate(paragraphs, start=1):
        emotion_scores, engine = classify_emotion_scores(paragraph)
        score = score_from_emotion_scores(emotion_scores)
        sentiment = classify_sentiment(score)
        emotion, emotion_confidence = emotion_from_scores(emotion_scores)
        emoji = emotion_emoji(emotion)
        sentence_count = len(split_into_sentences(paragraph))

        rows.append({
            "Paragraph": i,
            "SentenceCount": sentence_count,
            "Score": score,
            "Sentiment": sentiment,
            "Emotion": emotion,
            "EmotionConfidence": round(emotion_confidence, 4),
            "AnalysisEngine": engine,
            "Emoji": emoji,
            "Text": paragraph,
        })

    return pd.DataFrame(rows)


def build_analysis_from_section(text: str):
    paragraphs = split_into_paragraphs(text)
    if not paragraphs:
        return None

    df = analyze_paragraphs(paragraphs)

    story_shape = match_story_shape(df["Score"].tolist())
    df["SmoothScore"] = story_shape.get("smooth_values", df["Score"].tolist())
    df["StoryShapeScore"] = story_shape.get("shape_values", df["Score"].tolist())

    overall_score = round(float(df["Score"].mean()), 4)
    overall_sentiment = classify_sentiment(overall_score)

    positive_count = int((df["Score"] > 0.05).sum())
    negative_count = int((df["Score"] < -0.05).sum())
    neutral_count = int(((df["Score"] >= -0.05) & (df["Score"] <= 0.05)).sum())

    peak_idx = df["Score"].abs().idxmax()
    peak_row = df.loc[peak_idx]

    dominant_emotion = get_dominant_emotion(df)
    summary = build_writer_style_summary_from_paragraphs(paragraphs, df, overall_sentiment)
    analysis_engine = str(df["AnalysisEngine"].mode().iloc[0]) if "AnalysisEngine" in df else "Emotion model"

    return {
        "df": df,
        "paragraphs": paragraphs,
        "overall_score": overall_score,
        "overall_sentiment": overall_sentiment,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "peak_paragraph_number": int(peak_row["Paragraph"]),
        "peak_score": peak_row["Score"],
        "peak_emotion": peak_row["Emotion"],
        "peak_emoji": peak_row["Emoji"],
        "peak_text": peak_row["Text"],
        "dominant_emotion": dominant_emotion,
        "dominant_emoji": emotion_emoji(dominant_emotion),
        "story_shape": story_shape,
        "analysis_engine": analysis_engine,
        "summary": summary,
    }


def format_hover_text(text: str, width: int = 88) -> str:
    compact = re.sub(r"\s+", " ", clean_extracted_text(str(text))).strip()
    if not compact:
        return ""

    wrapped_lines = textwrap.wrap(
        compact,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return "<br>".join(html.escape(line) for line in wrapped_lines)


# ----------------------------
# Plotting
# ----------------------------
def plot_paragraph_graph(df: pd.DataFrame, title: str):
    plot_df = df.copy()
    plot_df["HoverText"] = plot_df["Text"].map(format_hover_text)
    if "EmotionConfidence" not in plot_df.columns:
        plot_df["EmotionConfidence"] = 0.0
    story_shape_name = ""
    if "StoryShapeScore" in plot_df.columns:
        matched = match_story_shape(plot_df["Score"].tolist())
        story_shape_name = matched.get("name", "")

    fig = px.line(plot_df, x="Paragraph", y="Score", markers=True, title=title)

    fig.update_traces(
        mode="lines+markers+text",
        text=plot_df["Emoji"],
        textposition="top center",
        customdata=plot_df[["Emotion", "Emoji", "EmotionConfidence", "SentenceCount", "HoverText"]].values,
        hovertemplate=(
            "<b>Paragraph %{x}</b><br>"
            "Emotion score: %{y}<br>"
            "Emotion: %{customdata[0]} %{customdata[1]}<br>"
            "Confidence: %{customdata[2]:.2f}<br>"
            "Sentence Count: %{customdata[3]}<br>"
            "Text: %{customdata[4]}<extra></extra>"
        ),
        name="Emotion score",
        line=dict(color="#14b8a6", width=4),
        marker=dict(size=10, color="#ffffff", line=dict(color="#14b8a6", width=3))
    )

    if "SmoothScore" in plot_df.columns:
        fig.add_scatter(
            x=plot_df["Paragraph"],
            y=plot_df["SmoothScore"],
            mode="lines",
            name="Smoothed arc",
            line=dict(color="#0f766e", width=3, shape="spline"),
            hovertemplate="<b>Smoothed arc</b><br>Paragraph %{x}<br>Score: %{y:.3f}<extra></extra>",
        )

    if "StoryShapeScore" in plot_df.columns and story_shape_name:
        fig.add_scatter(
            x=plot_df["Paragraph"],
            y=plot_df["StoryShapeScore"],
            mode="lines",
            name=f"{story_shape_name} shape",
            line=dict(color="#f9735b", width=3, dash="dot", shape="spline"),
            hovertemplate=f"<b>{html.escape(story_shape_name)} shape</b><br>Paragraph %{{x}}<br>Score: %{{y:.3f}}<extra></extra>",
        )

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="rgba(255,255,255,0)",
        paper_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#152033", family="Inter, Segoe UI, Arial, sans-serif", size=15),
        title=dict(font=dict(size=23, color="#152033"), x=0.02, xanchor="left"),
        hoverlabel=dict(
            align="left",
            bgcolor="#ffffff",
            bordercolor="#14b8a6",
            font=dict(color="#152033", family="Inter, Segoe UI, Arial, sans-serif", size=13),
        ),
        hovermode="closest",
        xaxis_title="Paragraph Number",
        yaxis_title="Emotion Valence Score",
        height=470,
        margin=dict(l=60, r=90, t=80, b=70),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12, color="#334155"),
        ),
        xaxis=dict(
            title_font=dict(size=16, color="#152033"),
            tickfont=dict(size=13, color="#637082"),
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title_font=dict(size=16, color="#152033"),
            tickfont=dict(size=13, color="#637082"),
            gridcolor="rgba(21,32,51,0.1)",
            zeroline=False
        )
    )

    fig.add_hline(y=0, line_dash="dash", line_color="#f9735b", line_width=2)
    return fig


def plot_peak_emotion_graph(peak_df: pd.DataFrame):
    plot_df = peak_df.copy()
    plot_df["HoverPeakText"] = plot_df["PeakText"].map(format_hover_text)

    fig = px.line(plot_df, x="Section", y="PeakScore", markers=True, title="Peak Emotion Flow Across All Sections")

    fig.update_traces(
        mode="lines+markers+text",
        text=plot_df["PeakEmoji"],
        textposition="top center",
        customdata=plot_df[["PeakEmotion", "PeakEmoji", "HoverPeakText"]].values,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Peak emotion score: %{y}<br>"
            "Peak Emotion: %{customdata[0]} %{customdata[1]}<br>"
            "Peak Text: %{customdata[2]}<extra></extra>"
        ),
        line=dict(color="#f9735b", width=4),
        marker=dict(size=11, color="#ffffff", line=dict(color="#f9735b", width=3))
    )

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="rgba(255,255,255,0)",
        paper_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#152033", family="Inter, Segoe UI, Arial, sans-serif", size=15),
        title=dict(font=dict(size=23, color="#152033"), x=0.02, xanchor="left"),
        hoverlabel=dict(
            align="left",
            bgcolor="#ffffff",
            bordercolor="#f9735b",
            font=dict(color="#152033", family="Inter, Segoe UI, Arial, sans-serif", size=13),
        ),
        hovermode="closest",
        xaxis_title="Section",
        yaxis_title="Peak Emotion Valence Score",
        height=520,
        margin=dict(l=60, r=90, t=80, b=90),
        xaxis=dict(
            title_font=dict(size=16, color="#152033"),
            tickfont=dict(size=13, color="#637082"),
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title_font=dict(size=16, color="#152033"),
            tickfont=dict(size=13, color="#637082"),
            gridcolor="rgba(21,32,51,0.1)",
            zeroline=False
        )
    )

    fig.add_hline(y=0, line_dash="dash", line_color="#14b8a6", line_width=2)
    return fig


# ----------------------------
# Display helpers
# ----------------------------
def show_analysis_block(analysis: dict, heading: str):
    st.markdown(f"""
<div class="result-kicker">Results</div>
<h2>{html.escape(heading)}</h2>
""", unsafe_allow_html=True)
    render_metric_grid(analysis)
    render_distribution_grid(analysis)
    render_story_shape_card(analysis)

    st.markdown(f"""
<div class="peak-card">
    <h3>Peak Emotion Paragraph</h3>
    <p><b>Paragraph:</b> {analysis['peak_paragraph_number']}</p>
    <p><b>Emotion:</b> {html.escape(str(analysis['peak_emotion']))} {html.escape(str(analysis['peak_emoji']))}</p>
    <p><b>Score:</b> {html.escape(str(analysis['peak_score']))}</p>
    <p>{html.escape(str(analysis['peak_text']))}</p>
</div>
<div class="summary-card">
    <h3>Summary</h3>
    <p>{html.escape(str(analysis["summary"]))}</p>
</div>
""", unsafe_allow_html=True)


def trigger_slow_auto_scroll() -> None:
    components.html(
        """
<script>
(function () {
    const parentWindow = window.parent || window;
    let cancelled = false;
    let timer = null;

    function getScrollState() {
        try {
            const doc = parentWindow.document;
            const body = doc.body;
            const root = doc.scrollingElement || doc.documentElement || body;
            const top = parentWindow.scrollY || root.scrollTop || body.scrollTop || 0;
            const max = Math.max(
                body ? body.scrollHeight : 0,
                root ? root.scrollHeight : 0,
                doc.documentElement ? doc.documentElement.scrollHeight : 0
            ) - parentWindow.innerHeight;
            return { root: root, top: top, max: Math.max(0, max) };
        } catch (error) {
            const root = document.scrollingElement || document.documentElement || document.body;
            return {
                root: root,
                top: root.scrollTop || 0,
                max: Math.max(0, root.scrollHeight - window.innerHeight)
            };
        }
    }

    function removeListeners() {
        try {
            parentWindow.removeEventListener("wheel", stop, { passive: true });
            parentWindow.removeEventListener("touchstart", stop, { passive: true });
            parentWindow.removeEventListener("mousedown", stop, false);
            parentWindow.removeEventListener("keydown", stop, false);
        } catch (error) {}
    }

    function stop() {
        cancelled = true;
        if (timer) {
            clearInterval(timer);
        }
        removeListeners();
    }

    try {
        parentWindow.addEventListener("wheel", stop, { passive: true });
        parentWindow.addEventListener("touchstart", stop, { passive: true });
        parentWindow.addEventListener("mousedown", stop, false);
        parentWindow.addEventListener("keydown", stop, false);
    } catch (error) {}

    setTimeout(function () {
        timer = setInterval(function () {
            if (cancelled) {
                stop();
                return;
            }

            const state = getScrollState();
            if (state.max - state.top <= 8) {
                stop();
                return;
            }

            try {
                parentWindow.scrollBy({ top: 2, left: 0, behavior: "auto" });
            } catch (error) {
                state.root.scrollTop = state.top + 2;
            }
        }, 20);
    }, 700);
})();
</script>
        """,
        height=0,
    )


# ----------------------------
# UI
# ----------------------------
render_hero()
render_section_heading(
    "Analysis Workspace",
    "Bring in a story.",
    "Upload a source file or paste text directly, then run the analyzer to reveal the emotional movement below.",
)

upload_col, text_col = st.columns([0.9, 1.1], gap="large")

with upload_col:
    upload_mode = st.radio(
        "Source type",
        ["Single file", "Multiple image files"],
        horizontal=True
    )

    if upload_mode == "Multiple image files":
        uploaded_file = st.file_uploader(
            "Upload image files",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a file",
            type=["txt", "docx", "doc", "pdf", "jpg", "jpeg", "png", "webp", "zip"]
        )

with text_col:
    pasted_text = st.text_area(
        "Paste text",
        height=260,
        placeholder="Paste a chapter, letter, passage, or full story excerpt here."
    )

option_col, table_col, action_col = st.columns([1, 1, 1.35], gap="large")
with option_col:
    show_extracted_text = st.checkbox("Show extracted text", value=False)
with table_col:
    show_paragraph_table = st.checkbox("Show paragraph-wise table", value=False)
with action_col:
    analysis_requested = st.button("Analyze Emotion Flow", width="stretch")

render_visual_tiles()

# ----------------------------
# Analyze
# ----------------------------
if analysis_requested:
    with st.spinner("Analyzing text... please wait"):
        text_data, input_type, warning_message = get_text_from_input(uploaded_file, pasted_text)

        if warning_message:
            st.warning(warning_message)

        if not text_data.strip():
            st.warning("Please upload a TXT, DOCX, PDF, JPG, PNG, ZIP file, multiple image files, or paste some text.")
        else:
            text_data = prepare_text_for_analysis(text_data)

            render_input_info(input_type, len(text_data))

            if uploaded_file is not None and pasted_text.strip():
                file_compact = compact_for_comparison(text_data)
                pasted_compact = compact_for_comparison(pasted_text)

                if file_compact == pasted_compact:
                    st.success("Uploaded file text matches the pasted text after analysis normalization.")
                else:
                    st.warning(
                        "Uploaded file text does not exactly match the pasted text after normalization. "
                        "The analysis will use the uploaded file because a file was provided."
                    )
                    st.caption(first_difference_preview(file_compact, pasted_compact))

            if show_extracted_text:
                with st.expander("View Extracted Text"):
                    st.text_area("Extracted text used for analysis", text_data, height=360, disabled=True)
                    st.download_button(
                        label="Download Extracted Text",
                        data=text_data.encode("utf-8"),
                        file_name="extracted_text_used_for_analysis.txt",
                        mime="text/plain",
                        key="download_extracted_text_used_for_analysis",
                    )

            sections = split_book_sections(text_data)

            if len(sections) == 1 and sections[0][0] == "Full Text" and looks_like_section_text(text_data):
                detected_title = detect_section_title_from_text(text_data)
                sections = [(detected_title, text_data)]

            if len(sections) == 1 and sections[0][0] == "Full Text":
                analysis = build_analysis_from_section(text_data)

                if analysis is None:
                    st.error("Could not analyze the text properly.")
                else:
                    if show_paragraph_table:
                        with st.expander("View Paragraph-wise Emotion Results"):
                            paragraph_df = analysis["df"][["Paragraph", "SentenceCount", "Score", "Sentiment", "Emotion", "EmotionConfidence", "Emoji", "Text"]]
                            render_paragraph_detail_report(paragraph_df, "Full Text Paragraph Details")
                            st.download_button(
                                label="Download Paragraph Details CSV",
                                data=paragraph_df.to_csv(index=False).encode("utf-8"),
                                file_name="paragraph_emotion_details.csv",
                                mime="text/csv",
                                key="download_full_text_paragraph_details_csv",
                            )

                    st.markdown('<h3 class="chart-title">Interactive Emotion Flow</h3>', unsafe_allow_html=True)
                    st.plotly_chart(
                        plot_paragraph_graph(analysis["df"], "Paragraph Emotion Flow"),
                        width="stretch"
                    )

                    show_analysis_block(analysis, "Final Paragraph Analysis")
                    chat_context = build_analysis_chat_context(
                        [("Full Text", analysis)],
                        analysis,
                        mode="single",
                        input_type=input_type,
                        character_count=len(text_data),
                    )
                    set_analysis_chat_context(chat_context)
                    trigger_slow_auto_scroll()

            else:
                render_section_heading("Section Results", "Section-wise Analysis")

                peak_rows = []
                all_text_parts = []
                section_analyses = []

                for section_index, (section_title, section_text) in enumerate(sections, start=1):
                    analysis = build_analysis_from_section(section_text)
                    if analysis is None:
                        continue

                    all_text_parts.append(section_text)
                    section_analyses.append((section_title, analysis))

                    st.markdown(f"<h2>{html.escape(section_title)}</h2>", unsafe_allow_html=True)

                    if show_paragraph_table:
                        with st.expander(f"View Paragraph-wise Emotion Results for {section_title}"):
                            paragraph_df = analysis["df"][["Paragraph", "SentenceCount", "Score", "Sentiment", "Emotion", "EmotionConfidence", "Emoji", "Text"]]
                            render_paragraph_detail_report(paragraph_df, f"{section_title} Paragraph Details")
                            safe_section_name = re.sub(r"[^A-Za-z0-9_-]+", "_", section_title).strip("_") or "section"
                            st.download_button(
                                label=f"Download {section_title} Paragraph Details CSV",
                                data=paragraph_df.to_csv(index=False).encode("utf-8"),
                                file_name=f"{safe_section_name.lower()}_paragraph_emotion_details.csv",
                                mime="text/csv",
                                key=f"download_section_{section_index}_{safe_section_name.lower()}_paragraph_details_csv",
                            )

                    st.markdown('<h3 class="chart-title">Interactive Emotion Flow</h3>', unsafe_allow_html=True)
                    st.plotly_chart(
                        plot_paragraph_graph(analysis["df"], f"Emotion Flow for {section_title}"),
                        width="stretch"
                    )

                    show_analysis_block(analysis, f"{section_title} Analysis")

                    peak_rows.append({
                        "Section": section_title,
                        "PeakScore": analysis["peak_score"],
                        "PeakEmotion": analysis["peak_emotion"],
                        "PeakEmoji": analysis["peak_emoji"],
                        "PeakParagraphNumber": analysis["peak_paragraph_number"],
                        "StoryShape": analysis["story_shape"].get("name", ""),
                        "StoryShapeSimilarity": analysis["story_shape"].get("similarity", ""),
                        "PeakText": analysis["peak_text"],
                        "Summary": analysis["summary"],
                    })

                full_book_text = "\n\n".join(all_text_parts)
                full_book_analysis = build_analysis_from_section(full_book_text)

                if peak_rows and len(peak_rows) > 1:
                    peak_df = pd.DataFrame(peak_rows)

                    render_section_heading("Full Text", "Peak Emotion Graph Across All Sections")
                    st.plotly_chart(
                        plot_peak_emotion_graph(peak_df),
                        width="stretch"
                    )

                    highest_peak_idx = peak_df["PeakScore"].abs().idxmax()
                    highest_peak_row = peak_df.loc[highest_peak_idx]

                    st.markdown(f"""
<div class="peak-card">
    <h3>Highest Peak in the Entire Text</h3>
    <p><b>Section:</b> {html.escape(str(highest_peak_row['Section']))}</p>
    <p><b>Peak Emotion:</b> {html.escape(str(highest_peak_row['PeakEmotion']))} {html.escape(str(highest_peak_row['PeakEmoji']))}</p>
    <p><b>Peak Score:</b> {html.escape(str(highest_peak_row['PeakScore']))}</p>
    <p><b>Peak Paragraph Number:</b> {html.escape(str(highest_peak_row['PeakParagraphNumber']))}</p>
    <p>{html.escape(str(highest_peak_row['PeakText']))}</p>
</div>
""", unsafe_allow_html=True)

                if full_book_analysis is not None:
                    show_analysis_block(full_book_analysis, "Entire Text Analysis")

                if peak_rows:
                    summary_df = pd.DataFrame(peak_rows)
                    render_section_heading(
                        "Readable Report",
                        "Section Summary Table",
                        "A clearer table view with wrapped text, section cards, and highlighted peak details.",
                    )
                    render_section_summary_report(summary_df)

                    st.download_button(
                        label="Download Section Summary CSV",
                        data=summary_df.to_csv(index=False).encode("utf-8"),
                        file_name="section_summary_results.csv",
                        mime="text/csv",
                        key="download_section_summary_results_csv",
                    )

                if all_text_parts:
                    chat_context = build_analysis_chat_context(
                        section_analyses,
                        full_book_analysis,
                        mode="sections",
                        input_type=input_type,
                        character_count=len(text_data),
                    )
                    set_analysis_chat_context(chat_context)
                    trigger_slow_auto_scroll()
elif st.session_state.get("analysis_chat_context"):
    render_cached_analysis_results(st.session_state.analysis_chat_context)

if st.session_state.get("analysis_chat_context"):
    render_analysis_chatbot(st.session_state.analysis_chat_context)
