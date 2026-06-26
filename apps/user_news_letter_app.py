import json
import base64
import os
import re
import smtplib
import ssl
import time
from pathlib import Path
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape

import requests
import streamlit as st
import streamlit.components.v1 as components
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials

GMAIL_SENDER_EMAIL_KEY = "GMAIL_SENDER_EMAIL"
GMAIL_APP_PASSWORD_KEY = "GMAIL_APP_PASSWORD"
GOOGLE_SHEET_ID_KEY = "GOOGLE_SHEET_ID"
GOOGLE_WORKSHEET_NAME_KEY = "GOOGLE_WORKSHEET_NAME"
GOOGLE_SERVICE_ACCOUNT_TABLE_KEY = "GOOGLE_SERVICE_ACCOUNT"
GOOGLE_SERVICE_ACCOUNT_JSON_KEY = "GOOGLE_SERVICE_ACCOUNT_JSON"
LOGIN_USERNAME_KEY = "NEWSLETTER_APP_USERNAME"
LOGIN_PASSWORD_KEY = "NEWSLETTER_APP_PASSWORD"

DEFAULT_WORKSHEET_NAME = "Tutorial_Requests"
DEFAULT_NEWSLETTER_SUBJECT = "Machine Gnostics Newsletter"
GOOGLE_SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SEND_DELAY_SECONDS = 2
SEND_PAUSE_EVERY = 10
SEND_PAUSE_SECONDS = 20


def load_logo_image_src() -> str | None:
    logo_path = Path(__file__).resolve().parents[1] / "docs" / "images" / "logo.png"
    if not logo_path.exists():
        return None
    logo_bytes = logo_path.read_bytes()
    return f"data:image/png;base64,{base64.b64encode(logo_bytes).decode('utf-8')}"


def load_logo_image_bytes() -> bytes | None:
    logo_path = Path(__file__).resolve().parents[1] / "docs" / "images" / "logo.png"
    if not logo_path.exists():
        return None
    return logo_path.read_bytes()

st.set_page_config(page_title="Machine Gnostics Newsletter Studio", page_icon="MG", layout="wide")

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
}

.stButton > button:hover {
    border: 1px solid rgba(144,255,245,.42);
    color: #fff;
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07111f 0%, #0b1726 100%);
}

div[data-testid="stForm"] {
    border: none !important;
    padding: 0 !important;
}
</style>
""",
    unsafe_allow_html=True,
)


def get_secret_value(key: str) -> str:
    try:
        return str(st.secrets.get(key, "")).strip()
    except Exception:
        return os.environ.get(key, "").strip()


def is_valid_email(value: str) -> bool:
    return "@" in value and "." in value.split("@")[-1]


def normalize_google_sheet_id(value: str) -> str:
    trimmed_value = value.strip()
    if "/spreadsheets/d/" in trimmed_value:
        match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", trimmed_value)
        if match:
            return match.group(1)
    return trimmed_value


def get_google_settings() -> dict[str, object] | None:
    spreadsheet_id = normalize_google_sheet_id(get_secret_value(GOOGLE_SHEET_ID_KEY))
    worksheet_name = get_secret_value(GOOGLE_WORKSHEET_NAME_KEY).strip() or DEFAULT_WORKSHEET_NAME
    service_account_table = None
    if GOOGLE_SERVICE_ACCOUNT_TABLE_KEY in st.secrets:
        service_account_table = dict(st.secrets[GOOGLE_SERVICE_ACCOUNT_TABLE_KEY])
    service_account_json = get_secret_value(GOOGLE_SERVICE_ACCOUNT_JSON_KEY).strip()

    if not spreadsheet_id:
        return None
    if not service_account_table and not service_account_json:
        return None

    return {
        "spreadsheet_id": spreadsheet_id,
        "worksheet_name": worksheet_name,
        "service_account_table": service_account_table,
        "service_account_json": service_account_json,
    }


def get_service_account_info(settings: dict[str, object]) -> dict[str, str]:
    table = settings.get("service_account_table")
    if isinstance(table, dict) and table:
        return dict(table)
    service_account_json = str(settings.get("service_account_json", ""))
    if not service_account_json.strip():
        raise ValueError(
            "Google service account secret is missing. Add a [GOOGLE_SERVICE_ACCOUNT] table or GOOGLE_SERVICE_ACCOUNT_JSON."
        )
    return json.loads(service_account_json)


def build_google_headers(settings: dict[str, object]) -> dict[str, str]:
    credentials = Credentials.from_service_account_info(
        get_service_account_info(settings),
        scopes=GOOGLE_SHEETS_SCOPES,
    )
    credentials.refresh(Request())
    return {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json",
    }


def get_sheet_rows(spreadsheet_id: str, worksheet_name: str, headers: dict[str, str]) -> list[list[str]]:
    range_name = requests.utils.quote(worksheet_name, safe="")
    values_url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{range_name}!A:Z"
    response = requests.get(values_url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json().get("values", [])


def load_recipients() -> list[dict[str, str]]:
    settings = get_google_settings()
    if not settings:
        return []

    headers = build_google_headers(settings)
    rows = get_sheet_rows(str(settings["spreadsheet_id"]), str(settings["worksheet_name"]), headers)
    if not rows:
        return []

    header_row = [cell.strip().lower() for cell in rows[0]]
    email_index = header_row.index("email") if "email" in header_row else 3
    first_name_index = header_row.index("first_name") if "first_name" in header_row else 1
    last_name_index = header_row.index("last_name") if "last_name" in header_row else 2
    opt_in_index = header_row.index("newsletter_opt_in") if "newsletter_opt_in" in header_row else None

    recipients: list[dict[str, str]] = []
    seen_emails: set[str] = set()

    for row in rows[1:]:
        if len(row) <= email_index:
            continue

        email = row[email_index].strip()
        if not is_valid_email(email):
            continue

        if opt_in_index is not None and len(row) > opt_in_index:
            opt_in_value = row[opt_in_index].strip().lower()
            if opt_in_value not in {"yes", "true", "1"}:
                continue

        normalized_email = email.lower()
        if normalized_email in seen_emails:
            continue

        seen_emails.add(normalized_email)
        recipients.append(
            {
                "email": email,
                "first_name": row[first_name_index].strip() if len(row) > first_name_index else "",
                "last_name": row[last_name_index].strip() if len(row) > last_name_index else "",
            }
        )

    return recipients


def build_email_html(
    subject: str,
    opening_statement: str,
    body_text: str,
    author_name: str,
    author_email: str,
    logo_src: str | None,
    logo_cid: str | None,
    image_src: str | None,
    image_cid: str | None,
) -> str:
    safe_subject = escape(subject)
    safe_opening_statement = escape(opening_statement)
    safe_author_name = escape(author_name)
    safe_author_email = escape(author_email)
    body_html = escape(body_text).replace("\n", "<br>")

    logo_html = ""
    logo_image_src = logo_src
    if logo_cid:
        logo_image_src = f"cid:{logo_cid}"
    if logo_image_src:
        logo_html = f"""
                        <div style="flex:0 0 auto;margin-left:auto;display:flex;align-items:center;justify-content:flex-end;">
                            <img src="{logo_image_src}" alt="Machine Gnostics logo" style="width:52px;height:52px;object-fit:contain;display:block;">
                        </div>
            """

    image_html = ""
    if image_src:
        image_html = f"""
                <div style="margin:24px 0 0;">
                    <img src="{image_src}" alt="Newsletter image" style="width:100%;max-width:560px;border-radius:18px;display:block;border:1px solid rgba(15,23,42,.08);box-shadow:0 18px 36px rgba(0,0,0,.12);">
                </div>
                """
    elif image_cid:
        image_html = f"""
                <div style="margin:24px 0 0;">
                    <img src="cid:{image_cid}" alt="Newsletter image" style="width:100%;max-width:560px;border-radius:18px;display:block;border:1px solid rgba(15,23,42,.08);box-shadow:0 18px 36px rgba(0,0,0,.12);">
                </div>
                """

    return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta name="color-scheme" content="light only">
            <meta name="supported-color-schemes" content="light only">
        </head>
        <body style="font-family:'IBM Plex Sans',Arial,sans-serif;background:linear-gradient(180deg,#f8fbfc 0%,#edf6f5 100%);margin:0;padding:28px;">
            <div style="max-width:700px;margin:auto;background:#fff;border-radius:20px;overflow:hidden;box-shadow:0 22px 50px rgba(15,23,42,.12);border:1px solid rgba(15,23,42,.08);">
                <div style="background:linear-gradient(180deg,#f8fbfc 0%,#edf6f5 100%);padding:38px 34px;border-bottom:1px solid rgba(15,23,42,.08);">
                    <div style="display:flex;align-items:center;justify-content:space-between;gap:18px;">
                        <div style="min-width:0;flex:1 1 auto;">
                            <div style="font-size:13px;letter-spacing:.1em;text-transform:uppercase;color:#0f766e;font-weight:700;">Machine Gnostics Newsletter</div>
                            <h1 style="color:#0F172A;margin:10px 0 6px;font-size:30px;line-height:1.1;">{safe_subject}</h1>
                            <p style="color:#334155;margin:0;line-height:1.6;">{safe_opening_statement}</p>
                        </div>
                        {logo_html}
                    </div>
                </div>
                <div style="padding:32px;">
                    {f'''<div style="background:#f8fafc;border:1px solid rgba(15,23,42,.08);border-radius:16px;padding:18px 20px;margin:0 0 20px;">
                        <div style="font-size:12px;text-transform:uppercase;letter-spacing:.08em;font-weight:700;color:#0f766e;margin-bottom:8px;">Author Details</div>
                        <div style="font-size:16px;font-weight:700;color:#0F172A;">{safe_author_name}</div>
                        <div style="font-size:14px;color:#334155;line-height:1.6;">{safe_author_email}</div>
                    </div>''' if safe_author_name.strip() else ''}
                    <div style="font-size:16px;line-height:1.8;color:#0F172A;white-space:normal;">{body_html}</div>
                    {image_html}
                    <div style="margin-top:28px;padding:16px 18px;border-top:1px solid rgba(15,23,42,.08);text-align:center;">
                        <div style="font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#0f766e;margin-bottom:8px;">Stay Connected</div>
                        <div style="font-size:14px;line-height:1.9;color:#334155;margin:0 auto 12px;max-width:520px;">
                            <a href="https://github.com/MachineGnostics/machinegnostics" style="color:#0EA5A4;text-decoration:none;margin:0 10px;">GitHub</a>
                            <a href="https://discord.gg/WMMUaeJe2X" style="color:#0EA5A4;text-decoration:none;margin:0 10px;">Discord</a>
                            <a href="https://www.linkedin.com/company/109036022/" style="color:#0EA5A4;text-decoration:none;margin:0 10px;">LinkedIn</a>
                            <a href="https://pypi.org/project/machinegnostics/" style="color:#0EA5A4;text-decoration:none;margin:0 10px;">PyPI</a>
                            <a href="https://www.instagram.com/machinegnostics/" style="color:#0EA5A4;text-decoration:none;margin:0 10px;">Instagram</a>
                            <a href="https://www.youtube.com/@MachineGnostics" style="color:#0EA5A4;text-decoration:none;margin:0 10px;">YouTube</a>
                        </div>
                        <div style="font-size:13px;line-height:1.7;color:#475569;max-width:560px;margin:0 auto;">
                            You are receiving this newsletter because you subscribed, opted in, or otherwise agreed to receive updates from Machine Gnostics.
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """


def build_plain_text(
    subject: str,
    opening_statement: str,
    body_text: str,
    author_name: str,
) -> str:
    author_line = f"Author: {author_name}\n\n" if author_name.strip() else ""
    return (
        f"Subject: {subject}\n\n"
        f"{opening_statement}\n\n"
        f"{author_line}"
        f"{body_text}\n\n"
        "You are receiving this newsletter because you subscribed, opted in, or otherwise agreed to receive updates from Machine Gnostics.\n"
    )


def send_newsletter(
    sender_email: str,
    app_password: str,
    recipients: list[dict[str, str]],
    subject: str,
    opening_statement: str,
    body_text: str,
    author_name: str,
    author_email: str,
    image_name: str | None,
    image_bytes: bytes | None,
    image_mime: str | None,
) -> tuple[int, list[str]]:
    delivered = 0
    failed: list[str] = []

    for recipient in recipients:
        full_name = " ".join(
            part for part in [recipient.get("first_name", ""), recipient.get("last_name", "")] if part
        ).strip()
        personal_salutation = f"Hi {full_name}," if full_name else "Hello,"

        message = MIMEMultipart("related")
        message["Subject"] = subject
        message["From"] = sender_email
        message["To"] = recipient["email"]
        message["Importance"] = "high"
        message["X-Priority"] = "1"
        message["Precedence"] = "personal"

        personalized_body = f"{personal_salutation}\n\n{body_text}"
        plain_text = build_plain_text(
            subject,
            opening_statement,
            personalized_body,
            author_name,
        )

        email_image_cid = None
        if image_bytes:
            email_image_cid = "newsletter-image"

        logo_image_cid = "newsletter-logo"

        html_body = build_email_html(
            subject=subject,
            opening_statement=opening_statement,
            body_text=personalized_body,
            author_name=author_name,
            author_email=author_email,
            logo_src=None,
            logo_cid=logo_image_cid,
            image_src=None,
            image_cid=email_image_cid,
        )

        alternative = MIMEMultipart("alternative")
        alternative.attach(MIMEText(plain_text, "plain"))
        alternative.attach(MIMEText(html_body, "html"))
        message.attach(alternative)

        if image_bytes and image_name:
            mime_root = (image_mime or "image/jpeg").split("/")[0]
            if mime_root == "image":
                subtype = (image_mime or "image/jpeg").split("/")[-1]
                image_part = MIMEImage(image_bytes, _subtype=subtype)
            else:
                image_part = MIMEApplication(image_bytes)
            image_part.add_header("Content-Disposition", "inline")
            image_part.add_header("Content-ID", "<newsletter-image>")
            message.attach(image_part)

        logo_bytes = load_logo_image_bytes()
        if logo_bytes:
            logo_part = MIMEImage(logo_bytes, _subtype="png")
            logo_part.add_header("Content-Disposition", "inline")
            logo_part.add_header("Content-ID", "<newsletter-logo>")
            message.attach(logo_part)

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as server:
                server.login(sender_email, app_password)
                server.sendmail(sender_email, recipient["email"], message.as_string())
            delivered += 1
            time.sleep(SEND_DELAY_SECONDS)
            if delivered % SEND_PAUSE_EVERY == 0:
                time.sleep(SEND_PAUSE_SECONDS)
        except Exception:
            failed.append(recipient["email"])

    return delivered, failed


def authenticate_user() -> bool:
    if st.session_state.get("newsletter_authenticated"):
        return True

    st.sidebar.subheader("Team Login")
    username = st.sidebar.text_input("Login name", key="newsletter_login_name").strip()
    password = st.sidebar.text_input("Password", type="password", key="newsletter_login_password").strip()
    expected_username = get_secret_value(LOGIN_USERNAME_KEY)
    expected_password = get_secret_value(LOGIN_PASSWORD_KEY)

    if expected_username and expected_password:
        st.sidebar.caption("Login secrets loaded.")
    else:
        st.sidebar.warning("Login secrets are missing or empty.")

    if st.sidebar.button("Login", use_container_width=True):
        if username == expected_username and password == expected_password:
            st.session_state["newsletter_authenticated"] = True
            st.sidebar.success("Login successful.")
        else:
            st.sidebar.error("Invalid login.")

    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state["newsletter_authenticated"] = False
        st.sidebar.info("Logged out.")

    return st.session_state.get("newsletter_authenticated", False)


def main() -> None:
    st.title("Machine Gnostics Newsletter Studio")
    st.markdown(
        "Compose a newsletter, choose the recipients from the Google Sheet used by the tutorial app, and send it from your Gmail account."
    )

    with st.sidebar:
        st.warning(
            "Broadcasting tool: double-check the subject, audience, author details, and message before sending."
        )

    countdown_placeholder = st.sidebar.empty()

    if not authenticate_user():
        st.info("Log in with the app-specific username and password to access the newsletter composer.")
        return

    sender_email = get_secret_value(GMAIL_SENDER_EMAIL_KEY).strip()
    app_password = get_secret_value(GMAIL_APP_PASSWORD_KEY).strip()

    if not sender_email or not app_password:
        st.error("Missing Gmail sending secrets.")
        return

    recipients = load_recipients()
    st.subheader("Recipient Source")
    st.write(f"Loaded {len(recipients)} unique recipient(s) from the subscriber sheet.")
    if recipients:
        st.dataframe(recipients, use_container_width=True, hide_index=True, height=390)

    st.subheader("Newsletter Composer")
    subject = st.text_input("Subject", value=DEFAULT_NEWSLETTER_SUBJECT)
    opening_statement = st.text_input("Opening statement", value="Small Data, Big Impact")
    author_name = st.text_input("Author name", placeholder="Machine Gnostics Team")
    author_email = st.text_input("Author email", value=sender_email)
    body_text = st.text_area(
        "Body message",
        height=260,
        placeholder="Write the newsletter body here...",
    )

    uploaded_image = st.file_uploader(
        "Attach one image",
        type=["png", "jpg", "jpeg", "webp", "gif"],
        accept_multiple_files=False,
    )
    image_bytes = uploaded_image.getvalue() if uploaded_image else None
    image_name = uploaded_image.name if uploaded_image else None
    image_mime = uploaded_image.type if uploaded_image else None

    preview_image_src = None
    logo_image_src = load_logo_image_src()
    if image_bytes:
        mime_type = image_mime or "image/png"
        preview_image_src = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"
    st.subheader("Preview")
    preview_html = build_email_html(
        subject=subject or DEFAULT_NEWSLETTER_SUBJECT,
        opening_statement=opening_statement or "Small Data, Big Impact",
        body_text=body_text or "",
        author_name=author_name or "",
        author_email=author_email,
        logo_src=logo_image_src,
        logo_cid=None,
        image_cid="newsletter-image" if image_bytes else None,
        image_src=preview_image_src,
    )
    components.html(preview_html, height=900, scrolling=True)

    copyable_html = preview_html
    components.html(
        f"""
            <div style="display:flex;justify-content:flex-end;margin:8px 0 0;">
                <button id="copy-body-btn" style="background:#0f766e;color:#fff;border:none;border-radius:12px;padding:0.7rem 1rem;font-weight:700;cursor:pointer;">
                    Copy HTML
                </button>
            </div>
            <script>
                const copyButton = document.getElementById('copy-body-btn');
                copyButton.addEventListener('click', async () => {{
                    try {{
                        const htmlContent = {json.dumps(copyable_html)};
                        if (navigator.clipboard && window.ClipboardItem) {{
                            await navigator.clipboard.write([
                                new ClipboardItem({{
                                    'text/html': new Blob([htmlContent], {{ type: 'text/html' }}),
                                    'text/plain': new Blob([htmlContent], {{ type: 'text/plain' }})
                                }})
                            ]);
                        }} else {{
                            await navigator.clipboard.writeText(htmlContent);
                        }}
                        copyButton.textContent = 'Copied HTML';
                        setTimeout(() => copyButton.textContent = 'Copy HTML', 1500);
                    }} catch (error) {{
                        copyButton.textContent = 'Copy failed';
                        setTimeout(() => copyButton.textContent = 'Copy HTML', 1500);
                    }}
                }});
            </script>
            """,
            height=70,
    )

    st.caption(
        "Send note: emails are sent one at a time with a short pause between messages to reduce the chance of Gmail throttling or account blocking."
    )

    if st.button("Send Newsletter", type="primary", use_container_width=True):
        errors: list[str] = []
        if not subject.strip():
            errors.append("Subject is required.")
        if not body_text.strip():
            errors.append("Body message is required.")
        if not author_email.strip() or not is_valid_email(author_email.strip()):
            errors.append("Author email must be valid.")
        if not recipients:
            errors.append("No recipients were loaded from the Google Sheet.")

        if errors:
            for error in errors:
                st.error(error)
            return

        stop_requested = False
        cancel_button = st.sidebar.button("Stop broadcast", use_container_width=True)
        countdown_box = st.sidebar.empty()

        for remaining_seconds in range(10, 0, -1):
            countdown_box.error(f"Sending in {remaining_seconds} second(s). Click Stop broadcast now to cancel.")
            if cancel_button:
                stop_requested = True
                break
            time.sleep(1)

        if stop_requested:
            countdown_box.error("Broadcast canceled.")
            return

        countdown_box.error("Sending now...")

        with st.spinner("Sending newsletter..."):
            delivered, failed = send_newsletter(
                sender_email=sender_email,
                app_password=app_password,
                recipients=recipients,
                subject=subject.strip(),
                opening_statement=opening_statement.strip() or "Small Data, Big Impact",
                body_text=body_text.strip(),
                author_name=author_name.strip(),
                author_email=author_email.strip(),
                image_name=image_name,
                image_bytes=image_bytes,
                image_mime=image_mime,
            )

        if delivered:
            st.success(f"Newsletter sent to {delivered} recipient(s).")
        if failed:
            st.warning(f"Failed to send to {len(failed)} recipient(s).")
            st.write(failed)


if __name__ == "__main__":
    main()
