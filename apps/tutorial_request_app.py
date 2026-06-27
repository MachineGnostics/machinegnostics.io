import csv
import json
import os
import re
import smtplib
import ssl
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape
from pathlib import Path
from urllib.parse import quote

import requests
import streamlit as st
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials


st.set_page_config(
    page_title="Machine Gnostics Learning Hub",
    page_icon="MG",
    layout="centered",
    initial_sidebar_state="collapsed",
)


FREE_TUTORIAL_URL = "https://machinegnostics-learning-pack.streamlit.app/"
DEFAULT_NEWSLETTER_STORAGE = Path(__file__).resolve().parent.parent / "data" / "newsletter_subscribers.csv"
GOOGLE_SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
GOOGLE_SHEETS_HEADERS = [
    "created_at_utc",
    "first_name",
    "last_name",
    "email",
    "user_profile",
    "selected_tracks",
    "newsletter_opt_in",
]

USER_PROFILE_OPTIONS = [
    "Developer",
    "Researcher",
    "Business",
    "Enthusiast",
    "Student",
    "Collaborator",
]

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
        "description": "Study how GDFs expose structure with a hands-on notebook and an interactive demo.",
        "resources": [
            ("Overview Tutorial", "https://docs.machinegnostics.com/latest/tutorials/overview/", "Docs"),
            ("GDF Tutorial Section", "https://docs.machinegnostics.com/latest/tutorials/tutorials/#advanced-analysis", "Docs"),
            ("EGDF", "https://docs.machinegnostics.com/latest/da/egdf/", "Docs"),
            ("ELDF", "https://docs.machinegnostics.com/latest/da/eldf/", "Docs"),
            ("QGDF", "https://docs.machinegnostics.com/latest/da/qgdf/", "Docs"),
            ("QLDF", "https://docs.machinegnostics.com/latest/da/qldf/", "Docs"),
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
    "Non-coders Web Apps": {
        "badge": "NO-CODE",
        "description": "Explore ready-to-use web apps designed for non-coders who want to learn by clicking, comparing, and observing.",
        "resources": [
            ("Anscombe Exploration", "https://machinegnostics-anscombe-banchmark.streamlit.app/", "Play"),
            ("Mean & Data Spread", "https://machinegnosticsio-data-spread-example.streamlit.app/", "Play"),
            ("GDFs", "https://machinegnosticsio-gdf.streamlit.app/", "Play"),
            ("IGC - Ideal Gnostic Cycle", "https://machinegnosticsio-igc.streamlit.app/", "Play"),
            ("Marginal Interval Analysis", "https://machinegnosticsio-intv.streamlit.app/", "Play"),
            ("Linear Regression", "https://machinegnosticsio-lin-reg.streamlit.app/", "Play"),
            ("Polynomial Regression", "https://machinegnosticsio-poly-reg.streamlit.app/", "Play"),
            ("Logistic Regression", "https://machinegnosticsio-logi-reg.streamlit.app/", "Play"),
        ],
    },
    "MG Concept Videos": {
        "badge": "VID",
        "description": "Watch AI-generated, high-level conceptual videos that explain Machine Gnostics ideas in a visual format.",
        "resources": [
            ("Machine Gnostics - A step towards non-statistical AI!", "https://youtu.be/YRYL0Yz-1tw?si=GYufKEJojnkMZg-e", "Video"),
            ("What is Machine Gnostics?", "https://youtu.be/CnXFweFXtYw?si=KfafAvHE4wgOvjYy", "Video"),
            ("The End of Randomness", "https://youtu.be/ZweSmDETOKk?si=dwgBxPqANnAqozRa", "Video"),
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
            ("Anscombe Exploration", "https://machinegnostics-anscombe-banchmark.streamlit.app/", "Play"),
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


def normalize_google_sheet_id(value: str) -> str:
    trimmed_value = value.strip()
    if "/spreadsheets/d/" in trimmed_value:
        match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", trimmed_value)
        if match:
            return match.group(1)
    return trimmed_value


def get_google_sheets_settings() -> dict[str, str] | None:
    spreadsheet_id = normalize_google_sheet_id(get_secret_value("GOOGLE_SHEET_ID"))
    worksheet_name = get_secret_value("GOOGLE_WORKSHEET_NAME").strip() or "Tutorial_Requests"
    service_account_table = st.secrets.get("GOOGLE_SERVICE_ACCOUNT", None)
    service_account_json = get_secret_value("GOOGLE_SERVICE_ACCOUNT_JSON").strip()

    if not spreadsheet_id or (not service_account_table and not service_account_json):
        return None

    return {
        "spreadsheet_id": spreadsheet_id,
        "worksheet_name": worksheet_name,
        "service_account_json": service_account_json,
        "service_account_table": service_account_table,
    }


def get_service_account_info(settings: dict[str, str]) -> dict[str, str]:
    if settings.get("service_account_table"):
        return dict(settings["service_account_table"])
    return json.loads(settings["service_account_json"])


def make_credentials(settings: dict[str, str]) -> Credentials:
    service_account_info = get_service_account_info(settings)
    return Credentials.from_service_account_info(service_account_info, scopes=GOOGLE_SHEETS_SCOPES)


def build_auth_headers(credentials: Credentials) -> dict[str, str]:
    credentials.refresh(Request())
    return {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json",
    }


def ensure_worksheet_exists(spreadsheet_id: str, worksheet_name: str, headers: dict[str, str]) -> None:
    metadata_url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}?fields=sheets.properties.title"
    response = requests.get(metadata_url, headers=headers, timeout=30)
    response.raise_for_status()
    sheet_titles = [sheet["properties"]["title"] for sheet in response.json().get("sheets", [])]
    if worksheet_name in sheet_titles:
        return

    create_url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}:batchUpdate"
    payload = {
        "requests": [
            {
                "addSheet": {
                    "properties": {
                        "title": worksheet_name,
                        "gridProperties": {"rowCount": 1000, "columnCount": 10},
                    }
                }
            }
        ]
    }
    response = requests.post(create_url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()


def get_sheet_rows(spreadsheet_id: str, worksheet_name: str, headers: dict[str, str]) -> list[list[str]]:
    range_name = requests.utils.quote(worksheet_name, safe="")
    values_url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{range_name}!A:G"
    response = requests.get(values_url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json().get("values", [])


def append_sheet_row(
    spreadsheet_id: str,
    worksheet_name: str,
    values: list[str],
    headers: dict[str, str],
) -> None:
    range_name = requests.utils.quote(worksheet_name, safe="")
    append_url = (
        f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{range_name}!A:G:append"
        "?valueInputOption=RAW&insertDataOption=INSERT_ROWS"
    )
    response = requests.post(append_url, headers=headers, json={"values": [values]}, timeout=30)
    response.raise_for_status()


def update_sheet_header_row(spreadsheet_id: str, worksheet_name: str, headers: list[str], auth_headers: dict[str, str]) -> None:
    range_name = requests.utils.quote(worksheet_name, safe="")
    update_url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{range_name}!A1:Z1?valueInputOption=RAW"
    response = requests.put(update_url, headers=auth_headers, json={"values": [headers]}, timeout=30)
    response.raise_for_status()


def append_newsletter_to_google_sheets(
    first_name: str,
    last_name: str,
    email: str,
    user_profile: str,
    selected: list[str],
) -> tuple[bool, str]:
    settings = get_google_sheets_settings()
    if not settings:
        return False, "google_settings_missing"

    try:
        credentials = make_credentials(settings)
        headers = build_auth_headers(credentials)
        ensure_worksheet_exists(settings["spreadsheet_id"], settings["worksheet_name"], headers)
        rows = get_sheet_rows(settings["spreadsheet_id"], settings["worksheet_name"], headers)
    except Exception as ex:
        status_code = getattr(getattr(ex, "response", None), "status_code", None)
        if status_code in {403, 404}:
            return False, "google_permission_denied"
        return False, f"google_write_failed:{type(ex).__name__}"

    if not rows:
        append_sheet_row(settings["spreadsheet_id"], settings["worksheet_name"], GOOGLE_SHEETS_HEADERS, headers)
        rows = [GOOGLE_SHEETS_HEADERS]

    header_row = rows[0]
    if "user_profile" not in header_row:
        update_sheet_header_row(settings["spreadsheet_id"], settings["worksheet_name"], GOOGLE_SHEETS_HEADERS, headers)
        header_row = GOOGLE_SHEETS_HEADERS

    normalized_email = email.strip().lower()
    if "email" in header_row:
        email_index = header_row.index("email")
        for row in rows[1:]:
            if len(row) > email_index and row[email_index].strip().lower() == normalized_email:
                return False, "duplicate"

    append_sheet_row(
        settings["spreadsheet_id"],
        settings["worksheet_name"],
        [
            datetime.now(timezone.utc).isoformat(),
            first_name.strip(),
            last_name.strip(),
            email.strip(),
            user_profile.strip(),
            " | ".join(selected),
            "yes",
        ],
        headers,
    )
    return True, "saved"


def save_newsletter_subscriber(
    first_name: str,
    last_name: str,
    email: str,
    user_profile: str,
    selected: list[str],
) -> tuple[bool, str]:
    google_saved, google_status = append_newsletter_to_google_sheets(first_name, last_name, email, user_profile, selected)
    if google_status in {"saved", "duplicate"}:
        return google_saved, google_status

    if get_google_sheets_settings() is not None:
        return False, google_status

    storage_path = DEFAULT_NEWSLETTER_STORAGE
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_email = email.strip().lower()
    if storage_path.exists():
        with storage_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("email", "").strip().lower() == normalized_email:
                    return False, "duplicate"

    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "first_name": first_name.strip(),
        "last_name": last_name.strip(),
        "email": email.strip(),
        "user_profile": user_profile.strip(),
        "selected_tracks": " | ".join(selected),
        "newsletter_opt_in": "yes",
    }
    write_header = not storage_path.exists() or storage_path.stat().st_size == 0
    with storage_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(record)
    return True, "csv_saved"


def get_resource_label(url: str, kind: str) -> str:
    lowered_url = url.lower()
    if "youtu.be" in lowered_url or "youtube.com" in lowered_url:
        return "VIDEO"
    if "colab.research.google.com" in lowered_url:
        return "CODE"
    return kind.upper()


def build_track_card(track_name: str) -> str:
    info = TUTORIALS[track_name]
    resources_html = "".join(
        f'<li style="margin:6px 0;"><span style="display:inline-block;min-width:72px;padding:2px 8px;margin-right:8px;border-radius:999px;background:rgba(14,165,164,.16);color:#7ce7df;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;">{escape(get_resource_label(url, kind))}</span><a href="{url}" style="color:#d8fffb;text-decoration:none;font-weight:600;">{escape(name)}</a></li>'
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
            {''.join(f'<li style="margin:6px 0;"><span style="font-size:11px;font-weight:700;color:#0f766e;text-transform:uppercase;letter-spacing:.06em;">{escape(get_resource_label(url, kind))}</span> · <a href="{url}" style="color:#0EA5A4;text-decoration:none;">{escape(name)}</a></li>' for name, url, kind in TUTORIALS[topic]['resources'])}
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
          <p style="color:#334155;margin:0;line-height:1.6;">Curated for the tracks you selected. Free resources, OSS software, and GPLv3-licensed learning tools.</p>
        </div>
        <div style="padding:32px;">
          <p style="font-size:16px;color:#0F172A;margin-top:0;">Hi <strong>{safe_name}</strong>,</p>
          <p style="color:#334155;line-height:1.7;">Thanks for choosing Machine Gnostics learning tracks. Below is the exact set of resources that matches your selection.</p>
          <div style="background:#f8fafc;border:1px solid rgba(15,23,42,.08);border-radius:14px;padding:14px 16px;margin:18px 0 22px;">
            <div style="font-size:12px;color:#64748b;text-transform:uppercase;letter-spacing:.08em;font-weight:700;margin-bottom:8px;">Selected Tracks</div>
            <div style="font-size:14px;color:#0F172A;line-height:1.7;">{escape(selected_summary)}</div>
          </div>
          <div style="background:linear-gradient(180deg,#ecfdf5 0%,#f0fdfa 100%);border:1px solid rgba(14,165,164,.18);border-radius:16px;padding:18px 20px;margin:0 0 22px;">
            <div style="font-size:12px;color:#0f766e;text-transform:uppercase;letter-spacing:.08em;font-weight:700;margin-bottom:8px;">Need an extra free tutorial?</div>
            <p style="margin:0 0 14px;color:#334155;line-height:1.7;">If you want additional free tutorials, use the button below to open the Machine Gnostics Learning Pack request app.</p>
            <a href="{FREE_TUTORIAL_URL}" style="display:inline-block;background:#0f766e;color:#fff;text-decoration:none;font-weight:700;letter-spacing:.04em;text-transform:uppercase;padding:12px 18px;border-radius:12px;">Order Free Tutorial</a>
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
    msg.attach(MIMEText(build_email_html(first_name, last_name, selected, sender_email), "html"))

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, app_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
.stApp { background: linear-gradient(180deg, #07111f 0%, #0b1726 45%, #0f172a 100%); color: #e2e8f0; font-family: 'IBM Plex Sans', 'Helvetica Neue', sans-serif; }
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5,.stApp h6,.stApp p,.stApp li,.stApp span,.stApp label { color: #e2e8f0; }
.stApp [data-testid="stMarkdownContainer"] p,.stApp [data-testid="stCaptionContainer"] p { color: #cbd5e1; }
.stButton > button { background: #0f766e; color: #fff; border: 1px solid rgba(124,231,223,.22); border-radius: 16px; padding: 0.95rem 1.25rem; min-height: 3.2rem; font-size: 0.98rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }
div[data-testid="stForm"] { border: none !important; padding: 0 !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Machine Gnostics Learning Hub - Free Resources")
st.markdown("Choose the MG learning tracks you want, preview the exact resources before you send, and receive the pack by email.")

sender_email = get_secret_value("GMAIL_SENDER_EMAIL")
app_password = get_secret_value("GMAIL_APP_PASSWORD")

with st.sidebar:
    with st.expander("Delivery Settings", expanded=False):
        if sender_email and app_password:
            st.success("Secrets loaded successfully.")
        else:
            st.warning("Set GMAIL_SENDER_EMAIL and GMAIL_APP_PASSWORD in secrets.")
        st.caption("To save subscribers privately in Google Sheets, add GOOGLE_SHEET_ID and GOOGLE_SERVICE_ACCOUNT.")
        if st.button("Test Google Sheets connection", use_container_width=True):
            settings = get_google_sheets_settings()
            if not settings:
                st.error("Google Sheets secrets are missing.")
            else:
                try:
                    credentials = make_credentials(settings)
                    headers = build_auth_headers(credentials)
                    ensure_worksheet_exists(settings["spreadsheet_id"], settings["worksheet_name"], headers)
                    st.success("Google Sheets connection is working.")
                except Exception as ex:
                    status_code = getattr(getattr(ex, "response", None), "status_code", None)
                    if status_code in {403, 404}:
                        st.error("Google Sheets is not shared with the service account email.")
                    else:
                        st.error(f"Google Sheets connection failed: {type(ex).__name__}")

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

user_profile = st.selectbox(
    "User Profile",
    options=USER_PROFILE_OPTIONS,
    index=None,
    placeholder="Choose the profile that best describes you...",
)

recipient_email = st.text_input("Your Email Address", placeholder="jane.doe@example.com")
required_consent = st.checkbox("I agree to share my name and email address to receive Machine Gnostics tutorials and communication.")
receive_updates = st.checkbox("Optional: I want to receive updates and future communications from Machine Gnostics.")

st.divider()

if st.button("Send Learning Pack", type="primary", use_container_width=True):
    errors = []
    if not selected_tutorials:
        errors.append("Please select at least one learning track.")
    if not first_name.strip():
        errors.append("First name is required.")
    if not last_name.strip():
        errors.append("Last name is required.")
    if not user_profile:
        errors.append("Please select a user profile.")
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
        newsletter_saved, newsletter_status = save_newsletter_subscriber(
            first_name=first_name,
            last_name=last_name,
            email=recipient_email,
            user_profile=user_profile,
            selected=selected_tutorials,
        )
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
                st.success(f"Email sent to **{recipient_email}**! Check your inbox for: {', '.join(selected_tutorials)}.")
                if newsletter_status in {"saved", "csv_saved"}:
                    st.info("Your details were saved privately for future updates.")
                elif newsletter_status == "duplicate":
                    st.info("Your email was already on the newsletter list, so it was not added twice.")
                elif newsletter_status == "google_permission_denied":
                    st.error("The spreadsheet is not shared with the service account email.")
                elif newsletter_status.startswith("google_write_failed"):
                    st.error(f"The app could not save the subscriber row to Google Sheets ({newsletter_status}).")
                elif newsletter_status in {"google_libraries_missing", "google_settings_missing"}:
                    st.error("The app could not save the subscriber row to Google Sheets. Check the secrets and logs.")
                if receive_updates:
                    st.info("You’ll also receive updates and future communications from Machine Gnostics.")
                st.link_button("Need another free tutorial? Open the request app", FREE_TUTORIAL_URL, use_container_width=True)
                st.balloons()
            except smtplib.SMTPAuthenticationError:
                st.error("Authentication failed. Make sure you're using a Gmail App Password, not your regular password.")
            except Exception as ex:
                st.error(f"Failed to send email: {ex}")
