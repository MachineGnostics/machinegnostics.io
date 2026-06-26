from html import escape

import streamlit as st


MG_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

.stApp {
    background:
        radial-gradient(circle at top left, rgba(14,165,164,.16), transparent 28%),
        radial-gradient(circle at top right, rgba(56,189,248,.10), transparent 24%),
        linear-gradient(180deg, #07111f 0%, #0b1726 45%, #0f172a 100%);
    color: #e2e8f0;
    font-family: 'IBM Plex Sans', 'Helvetica Neue', sans-serif;
}

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6,
.stApp p,
.stApp li,
.stApp span,
.stApp label {
    color: #e2e8f0;
}

.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stCaptionContainer"] p {
    color: #cbd5e1;
}

.stApp hr {
    border-color: rgba(148,163,184,.18) !important;
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07111f 0%, #0b1726 100%);
}

div[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

div[data-testid="stSidebar"] input,
div[data-testid="stSidebar"] textarea {
    color: #0f172a !important;
}

.mg-hero {
    background: linear-gradient(135deg, rgba(15,23,42,.96), rgba(8,15,30,.96));
    border: 1px solid rgba(124,231,223,.14);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 18px 44px rgba(0,0,0,.22);
    margin-bottom: 1rem;
}

.mg-hero-eyebrow {
    color: #7ce7df;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.mg-hero h1 {
    margin: 0 0 8px 0;
    color: #f8fafc;
    font-size: 2rem;
}

.mg-hero p {
    color: #cbd5e1;
    margin: 0;
    line-height: 1.65;
}

.mg-section-title {
    margin: 0 0 12px 0;
    font-size: 1.1rem;
    letter-spacing: 0.02em;
    color: #f8fafc;
}

.mg-callout {
    background: linear-gradient(180deg, #f4fbfa 0%, #eefaf8 100%);
    border: 1px solid rgba(124,231,223,.16);
    border-left: 4px solid #0ea5a4;
    border-radius: 14px;
    padding: 14px 16px;
    color: #334155;
    line-height: 1.6;
}

.stButton > button {
    background: #0f766e;
    color: #fff;
    border: 1px solid rgba(124,231,223,.22);
    border-radius: 16px;
    padding: 0.9rem 1.1rem;
    min-height: 3.1rem;
    font-size: 0.96rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    box-shadow: 0 18px 32px rgba(8,15,30,.32);
    transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease, border-color 160ms ease;
}

.stButton > button:hover {
    border: 1px solid rgba(144,255,245,.42);
    color: #fff;
    transform: translateY(-2px);
    box-shadow: 0 22px 40px rgba(8,15,30,.42);
    filter: brightness(1.06);
}

.stButton > button:focus-visible {
    outline: none;
    box-shadow: 0 0 0 3px rgba(20,184,166,.28), 0 22px 40px rgba(8,15,30,.42);
}

div[data-testid="stForm"] {
    border: none !important;
    padding: 0 !important;
}
</style>
"""


def apply_mg_theme() -> None:
    st.markdown(MG_THEME_CSS, unsafe_allow_html=True)


def render_mg_hero(title: str, subtitle: str, eyebrow: str = "Machine Gnostics") -> None:
    st.markdown(
        f"""
        <div class="mg-hero">
            <div class="mg-hero-eyebrow">{escape(eyebrow)}</div>
            <h1>{escape(title)}</h1>
            <p>{escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
