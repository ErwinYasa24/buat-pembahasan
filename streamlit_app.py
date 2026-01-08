"""Streamlit app to generate explanations directly on Google Sheets."""

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import gspread
from google.oauth2.service_account import Credentials

from app.ai_utils import generate_ai_explanations
from app.config import DEFAULT_MODEL

# --- Constants ---
DEFAULT_ENV_PATH = ".env"
GOOGLE_CREDS_ENV = "GOOGLE_SERVICE_ACCOUNT_FILE"
SERVICE_ACCOUNT_SECRET_KEY = "gcp_service_account"
EXPLANATION_COLUMN = "explanation_ai"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
]
AUTO_SAVE_BATCH_SIZE = 50
MAX_BATCH_RETRIES = 2
RETRY_BACKOFF_SECONDS = [60, 120]
DEFAULT_SESSION_STATE = {
    "stage": "init",
    "sheet_input": "",
    "client_creds": None,
    "sheet_names": [],
    "selected_sheets": [],
    "dataframes": {},
    "worksheet_objs": {},
    "explanation_cols": {},
    "last_update_summary": [],
    "pending_save": False,
    "_creds_tmpdir": None,
}


def centered_button(label: str, key: str, type: str = "primary", disabled: bool = False):
    col_left, col_center, col_right = st.columns([3, 2, 3])
    with col_center:
        return st.button(
            label,
            key=key,
            type=type,
            disabled=disabled,
            use_container_width=True,
        )


def load_environment(path: str = DEFAULT_ENV_PATH) -> None:
    """Load environment variables from .env if present."""
    if os.path.exists(path):
        load_dotenv(path)


def extract_spreadsheet_id(user_input: str) -> Optional[str]:
    """Extract spreadsheet ID from full URL or return stripped input."""
    if not user_input:
        return None
    url_pattern = r"/spreadsheets/d/([a-zA-Z0-9-_]+)"
    match = re.search(url_pattern, user_input)
    if match:
        return match.group(1)
    return user_input.strip()


@st.cache_resource(show_spinner=False)
def get_gspread_client(credentials_path: str) -> gspread.Client:
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(
            f"File kredensial tidak ditemukan: {credentials_path}. "
            "Unggah file service account JSON atau set variabel lingkungan."
        )
    credentials = Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
    return gspread.authorize(credentials)


def ensure_explanation_column(worksheet: gspread.Worksheet, headers: List[str]) -> int:
    """Ensure explanation column exists, return its 1-based index."""
    if EXPLANATION_COLUMN in headers:
        return headers.index(EXPLANATION_COLUMN) + 1

    next_col_index = len(headers) + 1
    a1 = gspread.utils.rowcol_to_a1(1, next_col_index)
    worksheet.update(a1, [[EXPLANATION_COLUMN]])
    return next_col_index


def fetch_dataframe(worksheet: gspread.Worksheet) -> pd.DataFrame:
    records = worksheet.get_all_records()
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df.columns = df.columns.str.strip()
    if EXPLANATION_COLUMN not in df.columns:
        df[EXPLANATION_COLUMN] = ""
    return df


def update_explanation_column(
    worksheet: gspread.Worksheet,
    column_index: int,
    values: List[str],
) -> None:
    if not values:
        return
    start_cell = gspread.utils.rowcol_to_a1(2, column_index)
    end_cell = gspread.utils.rowcol_to_a1(len(values) + 1, column_index)
    worksheet.update(f"{start_cell}:{end_cell}", [[v] for v in values])


def init_page() -> None:
    st.set_page_config(
        page_title="Buat Pembahasan",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        div[data-testid="stSidebar"] {display: none;}
        .main {max-width: 900px; margin: 0 auto; padding: 1rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h1 style='text-align:center;'>Buat Pembahasan</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;'>Tempel link Google Sheet, pilih worksheet, lalu isi kolom <code>explanation_ai</code>.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)


def set_stage(stage: str, *, pending_save: bool = False) -> None:
    st.session_state["stage"] = stage
    st.session_state["pending_save"] = pending_save
    st.rerun()


def ensure_session_defaults() -> None:
    for key, value in DEFAULT_SESSION_STATE.items():
        st.session_state.setdefault(key, value)


def reset_session() -> None:
    for key, value in DEFAULT_SESSION_STATE.items():
        if isinstance(value, dict):
            st.session_state[key] = {}
        elif isinstance(value, list):
            st.session_state[key] = []
        else:
            st.session_state[key] = value
    set_stage("init")


def _materialize_secret_credentials() -> Optional[str]:
    try:
        secret_payload = st.secrets[SERVICE_ACCOUNT_SECRET_KEY]
    except Exception:
        secret_payload = None

    if not secret_payload:
        return None

    if isinstance(secret_payload, str):
        content = secret_payload.strip()
        if not content:
            return None
    else:
        content = json.dumps(secret_payload)

    tmp_dir = st.session_state.get("_creds_tmpdir")
    if not tmp_dir:
        tmp_dir = tempfile.mkdtemp(prefix="gcp-creds-")
        st.session_state["_creds_tmpdir"] = tmp_dir

    path = Path(tmp_dir) / "service_account.json"
    path.write_text(content)
    return str(path)


def resolve_default_credentials() -> Optional[str]:
    existing = st.session_state.get("client_creds")
    if existing and os.path.exists(existing):
        return existing

    secret_path = _materialize_secret_credentials()
    if secret_path and os.path.exists(secret_path):
        st.session_state["client_creds"] = secret_path
        return secret_path

    env_path = os.getenv(GOOGLE_CREDS_ENV)
    if env_path and os.path.exists(env_path):
        st.session_state["client_creds"] = env_path
        return env_path

    return None


def stage_init(default_creds: Optional[str]) -> None:
    col_left, col_mid, col_right = st.columns([1, 2, 1])
    with col_mid:
        sheet_input = st.text_input(
            "Link atau ID Google Sheet",
            key="sheet_input",
            help="Tempel link Google Sheet atau langsung ID-nya.",
            placeholder="https://docs.google.com/spreadsheets/d/…",
        )

        if default_creds and os.path.exists(default_creds):
            st.session_state["client_creds"] = default_creds
        else:
            uploaded_file = st.file_uploader(
                "Unggah file service account JSON",
                type=["json"],
                help="File JSON service account dengan akses edit ke spreadsheet.",
            )
            if uploaded_file is not None:
                tmp_dir = st.session_state.get("_creds_tmpdir")
                if not tmp_dir:
                    tmp_dir = tempfile.mkdtemp(prefix="gcp-creds-")
                    st.session_state["_creds_tmpdir"] = tmp_dir
                temp_path = Path(tmp_dir) / uploaded_file.name
                temp_path.write_bytes(uploaded_file.getvalue())
                st.session_state["client_creds"] = str(temp_path)
            else:
                st.warning(
                    "Unggah file JSON service account, set variabel lingkungan "
                    f"`{GOOGLE_CREDS_ENV}`, atau simpan kredensial di `st.secrets['{SERVICE_ACCOUNT_SECRET_KEY}']`."
                )

    creds_final = st.session_state.get("client_creds")
    if sheet_input.strip() and creds_final:
        # Tombol untuk memuat spreadsheet pertama kali
        clicked = centered_button(
            "Muat Spreadsheet",
            key="load_sheet",
        )
        if clicked:
            sheet_id = extract_spreadsheet_id(sheet_input)
            if not sheet_id:
                with col_mid:
                    st.error("Masukkan link atau ID Google Sheet yang valid.")
            else:
                try:
                    client = get_gspread_client(creds_final)
                    spreadsheet = client.open_by_key(sheet_id)
                    sheet_names = [ws.title for ws in spreadsheet.worksheets()]
                    st.session_state["spreadsheet_id"] = sheet_id
                    st.session_state["spreadsheet_name"] = spreadsheet.title
                    st.session_state["sheet_names"] = sheet_names
                    st.session_state["worksheet_title"] = sheet_names[0] if sheet_names else None
                    st.session_state["dataframes"] = {}
                    st.session_state["worksheet_objs"] = {}
                    st.session_state["explanation_cols"] = {}
                    st.session_state["last_update_summary"] = []
                    with col_mid:
                        st.success(f"Berhasil memuat spreadsheet: {spreadsheet.title}")
                    set_stage("loaded")
                except Exception as exc:  # pragma: no cover
                    with col_mid:
                        st.error(f"Gagal memuat spreadsheet: {exc}")
    else:
        with col_mid:
            st.info("Masukkan link dan kredensial untuk menampilkan tombol muat.")


def stage_loaded() -> None:
    names = st.session_state.get("sheet_names", [])
    if not names:
        st.warning("Spreadsheet belum dimuat.")
        set_stage("init")
        return

    default_sheet = st.session_state.get("worksheet_title")
    default_selection = [default_sheet] if default_sheet in names else names[:1]
    selected = st.multiselect(
        "Pilih worksheet",
        names,
        default=default_selection,
        key="worksheet_select",
    )
    st.session_state["selected_sheets"] = selected

    disabled = len(selected) == 0
    # Tombol untuk menarik data dari worksheet terpilih
    fetch_clicked = centered_button(
        "Ambil Data Worksheet",
        disabled=disabled,
        key="fetch_data",
    )
    if fetch_clicked:
        creds_final = st.session_state.get("client_creds")
        sheet_id = st.session_state.get("spreadsheet_id")
        if not creds_final or not sheet_id:
            st.error("Spreadsheet atau kredensial belum siap.")
            return
        try:
            client = get_gspread_client(creds_final)
            spreadsheet = client.open_by_key(sheet_id)
            new_df_map: Dict[str, pd.DataFrame] = {}
            new_worksheet_objs: Dict[str, gspread.Worksheet] = {}
            new_explanation_cols: Dict[str, int] = {}

            for sheet in selected:
                worksheet = spreadsheet.worksheet(sheet)
                headers = [h.strip() for h in worksheet.row_values(1)]
                explanation_col_index = ensure_explanation_column(worksheet, headers)
                sheet_df = fetch_dataframe(worksheet)
                new_df_map[sheet] = sheet_df
                new_worksheet_objs[sheet] = worksheet
                new_explanation_cols[sheet] = explanation_col_index

            st.session_state["dataframes"] = new_df_map
            st.session_state["worksheet_objs"] = new_worksheet_objs
            st.session_state["explanation_cols"] = new_explanation_cols
            st.session_state["last_update_summary"] = []
            st.success("Berhasil mengambil data dari worksheet terpilih.")
            set_stage("fetched")
        except Exception as exc:  # pragma: no cover
            st.error(f"Gagal mengambil data worksheet: {exc}")


def display_tabs(df_map: Dict[str, pd.DataFrame]) -> None:
    tabs = st.tabs(list(df_map.keys()))
    for tab, sheet_name in zip(tabs, df_map.keys()):
        with tab:
            st.subheader(f"Worksheet: {sheet_name}")
            st.dataframe(df_map[sheet_name].head(), use_container_width=True)
            st.write(f"Total soal: {len(df_map[sheet_name])}")


def stage_fetched(default_mode: int = 0) -> None:
    df_map = st.session_state.get("dataframes", {})
    if not df_map:
        st.warning("Belum ada data worksheet. Muat ulang spreadsheet.")
        set_stage("loaded")
        return

    display_tabs(df_map)
    show_summary()

    mode = st.selectbox(
        "Mode generate",
        ["Hanya baris kosong", "Semua baris"],
        index=default_mode,
        key="mode_select",
    )

    # Tombol untuk menjalankan generasi pembahasan AI
    generate_clicked = centered_button(
        "Generate Pembahasan AI",
        key="generate_ai",
    )
    if generate_clicked:
        creds_final = st.session_state.get("client_creds")
        sheet_id = st.session_state.get("spreadsheet_id")
        if not creds_final or not sheet_id:
            st.error("Kredensial atau spreadsheet belum siap.")
            return

        client = get_gspread_client(creds_final)
        spreadsheet = client.open_by_key(sheet_id)

        worksheet_objs = st.session_state.get("worksheet_objs", {})
        explanation_cols = st.session_state.get("explanation_cols", {})

        updated_any = False
        sheet_summaries: List[str] = []

        for sheet_name, sheet_df in df_map.items():
            worksheet = worksheet_objs.get(sheet_name)
            if worksheet is None:
                worksheet = spreadsheet.worksheet(sheet_name)
                headers = [h.strip() for h in worksheet.row_values(1)]
                explanation_cols[sheet_name] = ensure_explanation_column(
                    worksheet, headers
                )
                worksheet_objs[sheet_name] = worksheet

            if mode == "Hanya baris kosong":
                mask = (
                    sheet_df[EXPLANATION_COLUMN].astype(str).str.strip().eq("")
                    | sheet_df[EXPLANATION_COLUMN].isna()
                )
                target_indices = sheet_df[mask].index.tolist()
            else:
                target_indices = sheet_df.index.tolist()

            if not target_indices:
                sheet_summaries.append(
                    f"{sheet_name}: tidak ada baris yang perlu diproses."
                )
                continue

            updated_rows: List[int] = []
            total_batches = max(1, (len(target_indices) + AUTO_SAVE_BATCH_SIZE - 1) // AUTO_SAVE_BATCH_SIZE)
            for batch_index in range(total_batches):
                start = batch_index * AUTO_SAVE_BATCH_SIZE
                batch = target_indices[start : start + AUTO_SAVE_BATCH_SIZE]
                batch_updates: List[int] = []
                for attempt in range(MAX_BATCH_RETRIES + 1):
                    try:
                        with st.spinner(
                            f"Memproses {sheet_name} (batch {batch_index + 1}/{total_batches})..."
                        ):
                            batch_updates = generate_ai_explanations(
                                sheet_df,
                                batch,
                                model_name=DEFAULT_MODEL,
                            )
                    except Exception as exc:  # pragma: no cover
                        st.warning(
                            f"{sheet_name}: batch {batch_index + 1} gagal diproses ({exc})."
                        )
                        batch_updates = []
                    if batch_updates:
                        break
                    if attempt < MAX_BATCH_RETRIES:
                        wait_seconds = RETRY_BACKOFF_SECONDS[
                            min(attempt, len(RETRY_BACKOFF_SECONDS) - 1)
                        ]
                        st.info(
                            f"{sheet_name}: batch {batch_index + 1} kosong. "
                            f"Mencoba ulang dalam {wait_seconds} detik..."
                        )
                        time.sleep(wait_seconds)
                if not batch_updates:
                    st.warning(
                        f"{sheet_name}: batch {batch_index + 1} gagal setelah retry."
                    )
                if batch_updates:
                    updated_rows.extend(batch_updates)
                    values = sheet_df[EXPLANATION_COLUMN].fillna("").astype(str).tolist()
                    update_explanation_column(
                        worksheet,
                        explanation_cols[sheet_name],
                        values,
                    )
                    st.info(
                        f"{sheet_name}: batch {batch_index + 1} tersimpan ke Google Sheets."
                    )

            if updated_rows:
                df_map[sheet_name] = sheet_df
                updated_any = True
                sheet_summaries.append(
                    f"{sheet_name}: {len(updated_rows)} baris diperbarui dan tersimpan."
                )
            else:
                sheet_summaries.append(
                    f"{sheet_name}: tidak ada baris yang diperbarui."
                )

        st.session_state["dataframes"] = df_map
        st.session_state["worksheet_objs"] = worksheet_objs
        st.session_state["explanation_cols"] = explanation_cols
        st.session_state["last_update_summary"] = sheet_summaries

        if updated_any:
            st.success("Pembahasan berhasil diperbarui dan langsung tersimpan ke Google Sheets.")
            set_stage("saved", pending_save=False)
        else:
            st.warning("Tidak ada baris yang berhasil diperbarui.")
            st.session_state["stage"] = "fetched"
            st.session_state["pending_save"] = False


def show_summary() -> None:
    summary = st.session_state.get("last_update_summary", [])
    if summary:
        st.markdown("**Ringkasan Pemrosesan:**")
        for message in summary:
            st.write("- " + message)


def stage_generated() -> None:
    show_summary()
    # Tombol untuk menyimpan hasil pembahasan kembali ke Google Sheets
    save_clicked = centered_button(
        "Simpan ke Google Sheets",
        key="save_sheet",
    )
    if save_clicked:
        df_map = st.session_state.get("dataframes", {})
        worksheet_objs = st.session_state.get("worksheet_objs", {})
        explanation_cols = st.session_state.get("explanation_cols", {})

        save_errors = []
        for sheet_name, sheet_df in df_map.items():
            worksheet = worksheet_objs.get(sheet_name)
            col_index = explanation_cols.get(sheet_name)
            if worksheet is None or col_index is None:
                save_errors.append(sheet_name)
                continue
            try:
                values = sheet_df[EXPLANATION_COLUMN].fillna("").astype(str).tolist()
                update_explanation_column(worksheet, col_index, values)
            except Exception as exc:  # pragma: no cover
                save_errors.append(f"{sheet_name} ({exc})")

        if save_errors:
            st.error(
                "Gagal menyimpan untuk sheet: " + ", ".join(save_errors)
            )
        else:
            st.success("Berhasil menyimpan kolom explanation_ai ke Google Sheets.")
            set_stage("saved")


def stage_saved() -> None:
    show_summary()
    st.success("Proses selesai.")
    restart_clicked = centered_button(
        "Mulai Sesi Baru",
        key="restart_session",
        type="secondary",
    )
    if restart_clicked:
        reset_session()


def main() -> None:
    load_environment()
    init_page()
    ensure_session_defaults()

    default_creds = resolve_default_credentials()
    stage = st.session_state.get("stage", "init")

    if stage == "init":
        stage_init(default_creds)
    elif stage == "loaded":
        stage_loaded()
    else:
        df_map = st.session_state.get("dataframes", {})
        if not df_map:
            st.warning("Belum ada worksheet yang dimuat.")
            set_stage("loaded")
            return

        if stage == "fetched":
            stage_fetched()
        elif stage == "generated":
            stage_generated()
        elif stage == "saved":
            stage_saved()

if __name__ == "__main__":
    main()
