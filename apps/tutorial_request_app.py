import os
import smtplib
import ssl
from urllib.parse import quote
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape

import streamlit as st


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Machine Gnostics Learning Hub",
    page_icon="MG",
    layout="centered",
)


# ── Tutorial data ────────────────────────────────────────────────────────────
TUTORIALS = {
    "Installation": {
        "badge": "INST",
        "description": "Get Machine Gnostics installed and ready across local, notebook, and cloud workflows.",
        "resources": [
            ("Installation Guide", "https://docs.machinegnostics.com/latest/installation/", "Docs"),
            ("Tutorial Overview", "https://docs.machinegnostics.com/latest/tutorials/overview/", "Docs"),
            ("Tutorials Library", "https://docs.machinegnostics.com/latest/tutorials/tutorials/", "Library"),
            ("Machine Gnostics Home", "https://machinegnostics.com/", "Website"),
            ("GitHub Repository", "https://github.com/MachineGnostics/machinegnostics", "GitHub"),
        ],
    },
    "Gnostic Distribution Functions": {
        "badge": "GDF",
        "description": "Study how GDFs expose structure with a hands-on notebook and an interactive Streamlit demo.",
        "resources": [
            ("Overview Tutorial", "https://docs.machinegnostics.com/latest/tutorials/overview/", "Docs"),
            ("GDF Tutorial Section", "https://docs.machinegnostics.com/latest/tutorials/tutorials/#advanced-analysis", "Docs"),
            ("Google Colab Notebook", "https://colab.research.google.com/github/MachineGnostics/machinegnostics/blob/dev-002/tutorials/tutorial_magcal_02_gnostic_distribution_functions.ipynb", "Colab"),
            ("Streamlit App", "https://machinegnosticsio-gdf.streamlit.app/", "Play"),
            ("GitHub Notebook", "https://github.com/MachineGnostics/machinegnostics/blob/dev-002/tutorials/tutorial_magcal_02_gnostic_distribution_functions.ipynb", "GitHub"),
        ],
    },
    "Data Analysis Models": {
        "badge": "DA",
        "description": "Explore the broader data analysis stack: models, tests, intervals, and homogeneity checks.",
        "resources": [
            ("Data Analysis Models", "https://docs.machinegnostics.com/latest/da/da_models/", "Docs"),
            ("EGDF", "https://docs.machinegnostics.com/latest/da/egdf/", "Docs"),
            ("ELDF", "https://docs.machinegnostics.com/latest/da/eldf/", "Docs"),
            ("QGDF", "https://docs.machinegnostics.com/latest/da/qgdf/", "Docs"),
            ("QLDF", "https://docs.machinegnostics.com/latest/da/qldf/", "Docs"),
            ("Cluster Analysis", "https://docs.machinegnostics.com/latest/da/cluster_analysis/", "Docs"),
            ("Interval Analysis", "https://docs.machinegnostics.com/latest/da/interval_analysis/", "Docs"),
            ("Data Homogeneity", "https://docs.machinegnostics.com/latest/da/homogeneity/", "Docs"),
            ("Data Scedasticity", "https://docs.machinegnostics.com/latest/da/scedasticity/", "Docs"),
            ("Data Membership", "https://docs.machinegnostics.com/latest/da/membership/", "Docs"),
        ],
    },
    "ML regression": {
        "badge": "REG",
        "description": "Work through regression-oriented Machine Gnostics models, datasets, and evaluation guidance.",
        "resources": [
            ("Regression Models", "https://docs.machinegnostics.com/latest/models/reg/lin_reg/", "Docs"),
            ("Polynomial Regressor", "https://docs.machinegnostics.com/latest/models/reg/poly_reg/", "Docs"),
            ("Decision Tree Regressor", "https://docs.machinegnostics.com/latest/models/cart/dt_reg/", "Docs"),
            ("Random Forest Regressor", "https://docs.machinegnostics.com/latest/models/cart/rf_reg/", "Docs"),
            ("Gnostic Boosting Regressor", "https://docs.machinegnostics.com/latest/models/cart/gb_reg/", "Docs"),
            ("Regression Data", "https://docs.machinegnostics.com/latest/datasets/reg_data/", "Docs"),
            ("R2 Score", "https://docs.machinegnostics.com/latest/metrics/r2/", "Docs"),
            ("Model Gallery", "https://docs.machinegnostics.com/latest/tutorials/tutorials/#regression", "Library"),
        ],
    },
    "ML classification": {
        "badge": "CLS",
        "description": "Use Machine Gnostics classification models, metrics, and datasets for decision-ready workflows.",
        "resources": [
            ("Classification Models", "https://docs.machinegnostics.com/latest/models/cls/log_reg/", "Docs"),
            ("Multi Class Classifier", "https://docs.machinegnostics.com/latest/models/cls/multi_class/", "Docs"),
            ("Decision Tree Classifier", "https://docs.machinegnostics.com/latest/models/cart/dt_cls/", "Docs"),
            ("Random Forest Classifier", "https://docs.machinegnostics.com/latest/models/cart/rf_cls/", "Docs"),
            ("Gnostic Boosting Classifier", "https://docs.machinegnostics.com/latest/models/cart/gb_cls/", "Docs"),
            ("Classification Data", "https://docs.machinegnostics.com/latest/datasets/cls_data/", "Docs"),
            ("Classification Report", "https://docs.machinegnostics.com/latest/metrics/classification_report/", "Docs"),
            ("Model Gallery", "https://docs.machinegnostics.com/latest/tutorials/tutorials/#classification", "Library"),
        ],
    },
    "ML Clustering": {
        "badge": "CLU",
        "description": "Explore clustering, local structure, and cluster quality measures in the MG framework.",
        "resources": [
            ("KMeans", "https://docs.machinegnostics.com/latest/models/cluster/kmeans/", "Docs"),
            ("Estimating Local Clustering", "https://docs.machinegnostics.com/latest/models/cluster/glc/", "Docs"),
            ("Moon Data", "https://docs.machinegnostics.com/latest/datasets/moon_data/", "Docs"),
            ("Silhouette Score", "https://docs.machinegnostics.com/latest/metrics/silhouette_score/", "Docs"),
            ("Cluster Gallery", "https://docs.machinegnostics.com/latest/tutorials/tutorials/#clustering", "Library"),
        ],
    },
    "MlFlow Integration": {
        "badge": "MLF",
        "description": "Track runs, compare experiments, and connect Machine Gnostics workflows with Mlflow.",
        "resources": [
            ("MLflow Integration", "https://docs.machinegnostics.com/latest/tutorials/tutorials/#mlflow", "Library"),
            ("Mlflow Tracking", "https://docs.machinegnostics.com/latest/models/sup/mlflow/", "Docs"),
            ("Cross Validation", "https://docs.machinegnostics.com/latest/models/sup/cross_val/", "Docs"),
            ("Train Test Split", "https://docs.machinegnostics.com/latest/models/sup/train_test_split/", "Docs"),
            ("Tutorials Library", "https://docs.machinegnostics.com/latest/tutorials/tutorials/", "Library"),
        ],
    },
    "Web Apps": {
        "badge": "WEB",
        "description": "Open ready-to-run Streamlit demos and web experiences from the Machine Gnostics ecosystem.",
        "resources": [
            ("Ideal Gnostic Cycle App", "https://machinegnosticsio-igc.streamlit.app/", "Play"),
            ("GDF Streamlit App", "https://machinegnosticsio-gdf.streamlit.app/", "Play"),
            ("Interval Analysis App", "https://machinegnosticsio-intv.streamlit.app/", "Play"),
            ("Machine Gnostics Benchmark", "https://machinegnostics.com/benchmark/", "Website"),
            ("Developer Entry Point", "https://machinegnostics.com/developers/", "Website"),
        ],
    },
    "Mathematical Gnostics Books": {
        "badge": "BOOK",
        "description": "Read the foundational theory behind Machine Gnostics and the mathematical ideas it is built on.",
        "resources": [
            ("Math Gnostics Books", "https://www.math-gnostics.eu/books/", "Books"),
            ("Concepts", "https://docs.machinegnostics.com/latest/mg/concepts/", "Docs"),
            ("Principles", "https://docs.machinegnostics.com/latest/mg/principles/", "Docs"),
            ("Architecture", "https://docs.machinegnostics.com/latest/mg/architecture/", "Docs"),
            ("Distributions Functions", "https://docs.machinegnostics.com/latest/mg/gdf/", "Docs"),
            ("References", "https://docs.machinegnostics.com/latest/ref/references/", "Docs"),
            ("History", "https://docs.machinegnostics.com/latest/stories/history/", "Docs"),
        ],
    },
    "Anscombe Case Study": {
        "badge": "ANS",
        "description": "Follow a complete Anscombe Quartet case study that compares classical statistics and Machine Gnostics workflows.",
        "resources": [
            ("GitHub Repo", "https://github.com/nirmalparmarphd/machinegnostics-anscombe-data", "GitHub"),
            ("Part 0 - Setup and Orientation", "https://colab.research.google.com/github/nirmalparmarphd/machinegnostics-anscombe-data/blob/main/part_0_setup.ipynb", "Colab"),
            ("Part 1 - Gnostic Metrics", "https://colab.research.google.com/github/nirmalparmarphd/machinegnostics-anscombe-data/blob/main/part_1_gnostic_metrics.ipynb", "Colab"),
            ("Part 2 - Distribution Functions", "https://colab.research.google.com/github/nirmalparmarphd/machinegnostics-anscombe-data/blob/main/part_2_gnostic_distribution_functions.ipynb", "Colab"),
            ("Part 3 - Interval Analysis", "https://colab.research.google.com/github/nirmalparmarphd/machinegnostics-anscombe-data/blob/main/part_3_gnostic_marginal_interval_analysis.ipynb", "Colab"),
            ("Part 4 - Linear Regression", "https://colab.research.google.com/github/nirmalparmarphd/machinegnostics-anscombe-data/blob/main/part_4_gnostic_linear_regression.ipynb", "Colab"),
            ("Quick Study Notebook", "https://colab.research.google.com/github/nirmalparmarphd/machinegnostics-anscombe-data/blob/main/anscombe_data.ipynb", "Colab"),
        ],
    },
}


def is_valid_email(value: str) -> bool:
    return "@" in value and "." in value.split("@")[-1]


def get_secret_value(key: str) -> str:
    try:
        return st.secrets.get(key, "")
    except FileNotFoundError:
        return os.environ.get(key, "")
    except KeyError:
        return os.environ.get(key, "")


def build_track_card(track_name: str) -> str:
    info = TUTORIALS[track_name]
    resources_html = "".join(
                f'<li style="margin:6px 0;"><span style="display:inline-block;min-width:72px;padding:2px 8px;margin-right:8px;border-radius:999px;background:rgba(14,165,164,.16);color:#7ce7df;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;">{escape(kind)}</span><a href="{url}" style="color:#d8fffb;text-decoration:none;font-weight:600;">{escape(name)}</a></li>'
        for name, url, kind in info["resources"]
    )
    return f"""
        <div style="background:linear-gradient(180deg, rgba(15,23,42,.92) 0%, rgba(15,23,42,.84) 100%);border:1px solid rgba(124,231,223,.14);border-left:5px solid #0EA5A4;border-radius:18px;padding:20px 22px;margin-bottom:16px;box-shadow:0 18px 36px rgba(0,0,0,.22);">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                <div style="width:38px;height:38px;border-radius:12px;background:linear-gradient(135deg,#0F172A,#0EA5A4);color:#fff;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;letter-spacing:.05em;">{info['badge']}</div>
        <div>
                    <div style="font-size:18px;font-weight:700;color:#f8fafc;line-height:1.2;">{escape(track_name)}</div>
                    <div style="font-size:13px;color:#94a3b8;">{escape(info['description'])}</div>
        </div>
      </div>
      <ul style="padding-left:0;margin:14px 0 0;list-style:none;">{resources_html}</ul>
    </div>
    """


def build_email_html(first_name: str, last_name: str, selected: list[str], sender_email: str) -> str:
    safe_name = escape(f"{first_name} {last_name}".strip())
    unsubscribe_subject = quote("Unsubscribe from Machine Gnostics emails")
    unsubscribe_link = f"mailto:{escape(sender_email)}?subject={unsubscribe_subject}"
    track_blocks = "".join(
        f"""
        <div style="background:#f4fbfa;border:1px solid rgba(14,165,164,.18);border-left:4px solid #0EA5A4;border-radius:12px;padding:20px 22px;margin:0 0 18px;">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                        <div style="width:34px;height:34px;border-radius:10px;background:#0F172A;color:#fff;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;letter-spacing:.05em;">{TUTORIALS[topic]['badge']}</div>
            <h2 style="margin:0;color:#0F172A;font-size:20px;">{escape(topic)}</h2>
          </div>
          <p style="color:#334155;line-height:1.6;margin:0 0 14px;">{escape(TUTORIALS[topic]['description'])}</p>
          <div style="margin:0 0 8px;font-size:12px;text-transform:uppercase;letter-spacing:.08em;font-weight:700;color:#0F766E;">Learning Resources</div>
          <ul style="padding-left:18px;margin:0;">
            {''.join(f'<li style="margin:6px 0;"><span style="font-size:11px;font-weight:700;color:#0f766e;text-transform:uppercase;letter-spacing:.06em;">{escape(kind)}</span> · <a href="{url}" style="color:#0EA5A4;text-decoration:none;">{escape(name)}</a></li>' for name, url, kind in TUTORIALS[topic]['resources'])}
          </ul>
        </div>
        """
        for topic in selected
    )

    selected_summary = ", ".join(selected)

    return f"""
    <!DOCTYPE html>
    <html>
        <head>
            <meta name="color-scheme" content="light only">
            <meta name="supported-color-schemes" content="light only">
        </head>
    <body style="font-family:'IBM Plex Sans',Arial,sans-serif;background:linear-gradient(180deg,#f8fbfc 0%,#edf6f5 100%);margin:0;padding:28px;">
      <div style="max-width:680px;margin:auto;background:#fff;border-radius:20px;overflow:hidden;box-shadow:0 22px 50px rgba(15,23,42,.12);border:1px solid rgba(15,23,42,.08);">
                <div style="background:linear-gradient(180deg,#f8fbfc 0%,#edf6f5 100%);padding:38px 34px;border-bottom:1px solid rgba(15,23,42,.08);">
                    <div style="font-size:13px;letter-spacing:.1em;text-transform:uppercase;color:#0f766e;font-weight:700;">Machine Gnostics</div>
                    <h1 style="color:#0F172A;margin:10px 0 6px;font-size:30px;line-height:1.1;">Your Learning Pack</h1>
                    <p style="color:#334155;margin:0;line-height:1.6;">Curated for the tracks you selected.</p>
        </div>
        <div style="padding:32px;">
          <p style="font-size:16px;color:#0F172A;margin-top:0;">Hi <strong>{safe_name}</strong>,</p>
                    <p style="color:#334155;line-height:1.7;">Thanks for choosing Machine Gnostics learning tracks. Below is the exact set of resources that matches your selection.</p>
                    <div style="background:#f8fafc;border:1px solid rgba(15,23,42,.08);border-radius:14px;padding:14px 16px;margin:18px 0 22px;">
            <div style="font-size:12px;color:#64748b;text-transform:uppercase;letter-spacing:.08em;font-weight:700;margin-bottom:8px;">Selected Tracks</div>
            <div style="font-size:14px;color:#0F172A;line-height:1.7;">{escape(selected_summary)}</div>
          </div>
          {track_blocks}
                    <div style="margin-top:16px;padding:16px 18px;border-top:1px solid rgba(15,23,42,.08);text-align:center;">
                        <div style="font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#0f766e;margin-bottom:8px;">Follow Machine Gnostics</div>
                        <div style="font-size:14px;line-height:1.9;color:#334155;margin:0 auto 12px;max-width:520px;">
                            <a href="https://github.com/MachineGnostics/machinegnostics" style="color:#0EA5A4;text-decoration:none;margin:0 10px;">GitHub</a>
                            <a href="https://discord.gg/WMMUaeJe2X" style="color:#0EA5A4;text-decoration:none;margin:0 10px;">Discord</a>
                            <a href="https://www.linkedin.com/company/109036022/" style="color:#0EA5A4;text-decoration:none;margin:0 10px;">LinkedIn</a>
                            <a href="https://pypi.org/project/machinegnostics/" style="color:#0EA5A4;text-decoration:none;margin:0 10px;">PyPI</a>
                            <a href="https://www.instagram.com/machinegnostics/" style="color:#0EA5A4;text-decoration:none;margin:0 10px;">Instagram</a>
                            <a href="https://www.youtube.com/@MachineGnostics" style="color:#0EA5A4;text-decoration:none;margin:0 10px;">YouTube</a>
                        </div>
                        <div style="margin-top:10px;font-size:12px;line-height:1.6;color:#64748b;">
                            If you prefer not to receive future emails, you can <a href="{unsubscribe_link}" style="color:#0F172A;text-decoration:underline;font-weight:600;">unsubscribe here</a> or reply to this message.
                        </div>
                        <div style="font-weight:700;color:#0F172A;">Machine Gnostics · <a href="https://machinegnostics.com/" style="color:#0EA5A4;text-decoration:none;">machinegnostics.com</a></div>
                        <p style="margin:10px 0 0;color:#64748b;font-size:13px;">Small Data, Big Impact</p>
                    </div>
        </div>
      </div>
    </body>
    </html>
    """


def send_email(
    sender_email: str,
    app_password: str,
    recipient_email: str,
    first_name: str,
    last_name: str,
    selected: list[str],
) -> None:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Machine Gnostics Learning Pack - {', '.join(selected)}"
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["List-Unsubscribe"] = f"<mailto:{sender_email}?subject={quote('Unsubscribe from Machine Gnostics emails')}>"

    html = build_email_html(first_name=first_name, last_name=last_name, selected=selected, sender_email=sender_email)
    msg.attach(MIMEText(html, "html"))

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, app_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())


# ── UI ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
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

.stMultiSelect [data-baseweb="tag"] {
    background: #0EA5A4 !important;
}

.stButton > button {
    background: #0f766e;
    color: #fff;
    border: 1px solid rgba(124,231,223,.22);
    border-radius: 16px;
    padding: 0.95rem 1.25rem;
    min-height: 3.2rem;
    font-size: 0.98rem;
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
""",
    unsafe_allow_html=True,
)

st.title("Machine Gnostics Learning Concierge")
st.markdown(
    "Choose the MG learning tracks you want, preview the exact resources before you send, and receive the pack by email."
)

st.divider()

sender_email = get_secret_value("GMAIL_SENDER_EMAIL")
app_password = get_secret_value("GMAIL_APP_PASSWORD")

with st.sidebar:
    st.header("Delivery Settings")
    st.markdown(
        "This app sends mail from the Machine Gnostics account configured in Streamlit secrets. "
        "Set `GMAIL_SENDER_EMAIL` and `GMAIL_APP_PASSWORD` in your deployment secrets."
    )
    if sender_email and app_password:
        st.success("Secrets loaded successfully.")
    else:
        st.warning(
            "No secrets were found. Add them in `.streamlit/secrets.toml`, Streamlit Cloud secrets, or environment variables."
        )

st.subheader("Choose Your MG Learning Tracks")
selected_tutorials = st.multiselect(
    "Select one or more tracks:",
    options=list(TUTORIALS.keys()),
    format_func=lambda x: f"{TUTORIALS[x]['badge']}  {x}",
    placeholder="Click to pick tracks...",
)

st.subheader("Preview before sending")
if selected_tutorials:
    st.caption("This is the resource bundle that will be included in your email.")
    for track_name in selected_tutorials:
        st.markdown(build_track_card(track_name), unsafe_allow_html=True)
else:
    st.info("Pick one or more tracks to preview the email content.")

st.subheader("Your Details")
col1, col2 = st.columns(2)
with col1:
    first_name = st.text_input("First Name", placeholder="Jane")
with col2:
    last_name = st.text_input("Last Name", placeholder="Doe")

recipient_email = st.text_input("Your Email Address", placeholder="jane.doe@example.com")

# st.markdown(
#     """
#     <div style="margin-top:8px;padding:14px 16px;border:1px solid rgba(124,231,223,.14);border-radius:14px;background:rgba(15,23,42,.55);color:#cbd5e1;line-height:1.6;">
#       <strong style="color:#f8fafc;">Consent note</strong><br>
#       To receive this email and other Machine Gnostics communication, I agree to share my name and email address.
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

required_consent = st.checkbox(
    "I agree to share my name and email address to receive Machine Gnostics tutorials and communication.",
)
receive_updates = st.checkbox(
    "Optional: I want to receive updates and future communications from Machine Gnostics.",
)

st.divider()

if st.button("Send Learning Pack", type="primary", use_container_width=True):
    errors = []
    if not selected_tutorials:
        errors.append("Please select at least one learning track.")
    if not first_name.strip():
        errors.append("First name is required.")
    if not last_name.strip():
        errors.append("Last name is required.")
    if not recipient_email.strip() or not is_valid_email(recipient_email.strip()):
        errors.append("Please enter a valid recipient email.")
    if not required_consent:
        errors.append("Please tick the consent checkbox to continue.")
    if not sender_email.strip() or not is_valid_email(sender_email.strip()):
        errors.append("Please enter your Gmail address in the sidebar.")
    if not app_password.strip():
        errors.append("Please enter your Gmail App Password in the sidebar.")

    if errors:
        for error in errors:
            st.error(error)
    else:
        with st.spinner("Sending your Machine Gnostics learning pack..."):
            try:
                send_email(
                    sender_email=sender_email.strip(),
                    app_password=app_password.strip(),
                    recipient_email=recipient_email.strip(),
                    first_name=first_name.strip(),
                    last_name=last_name.strip(),
                    selected=selected_tutorials,
                )
                st.success(
                    f"Email sent to **{recipient_email}**! "
                    f"Check your inbox for: {', '.join(selected_tutorials)}."
                )
                if receive_updates:
                    st.info("You’ll also receive updates and future communications from Machine Gnostics.")
                st.balloons()
            except smtplib.SMTPAuthenticationError:
                st.error(
                    "Authentication failed. Make sure you're using a Gmail **App Password**, "
                    "not your regular password. [Create one here](https://myaccount.google.com/apppasswords)."
                )
            except Exception as ex:
                st.error(f"Failed to send email: {ex}")

st.markdown(
    """
<div style="margin-top:28px;padding:18px 0 8px;border-top:1px solid rgba(15,23,42,.10);text-align:center;color:#475569;">
  <div style="font-weight:700;color:#0F172A;">Machine Gnostics</div>
  <a href="https://machinegnostics.com/" target="_blank" style="color:#0EA5A4;text-decoration:none;">machinegnostics.com</a>
        <div style="margin-top:14px;padding:14px 16px;border:1px solid rgba(124,231,223,.14);border-radius:14px;background:rgba(15,23,42,.55);text-align:center;color:#cbd5e1;">
        <div style="font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#7ce7df;margin-bottom:10px;">Follow Machine Gnostics</div>
        <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:10px 14px;line-height:1.6;">
            <a href="https://github.com/MachineGnostics/machinegnostics" target="_blank" style="color:#d8fffb;text-decoration:none;">GitHub</a>
            <a href="https://discord.gg/WMMUaeJe2X" target="_blank" style="color:#d8fffb;text-decoration:none;">Discord</a>
            <a href="https://www.linkedin.com/company/109036022/" target="_blank" style="color:#d8fffb;text-decoration:none;">LinkedIn</a>
            <a href="https://pypi.org/project/machinegnostics/" target="_blank" style="color:#d8fffb;text-decoration:none;">PyPI</a>
            <a href="https://www.instagram.com/machinegnostics/" target="_blank" style="color:#d8fffb;text-decoration:none;">Instagram</a>
            <a href="https://www.youtube.com/@MachineGnostics" target="_blank" style="color:#d8fffb;text-decoration:none;">YouTube</a>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)