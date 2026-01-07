"""Fungsi terkait pemuatan data dan utilitas dataframe."""

import io
import os
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .config import REQUIRED_COLUMNS


def list_excel_sheets(file_bytes: bytes) -> List[str]:
    """Mengembalikan nama sheet yang tersedia pada berkas Excel."""
    with pd.ExcelFile(io.BytesIO(file_bytes)) as xls:
        return list(xls.sheet_names)


def load_sheet_frames(
    file_bytes: bytes,
    filename: str,
    encoding: str,
    sheets: Optional[Iterable[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Membaca berkas menjadi kumpulan DataFrame per sheet."""

    suffix = os.path.splitext(filename)[1].lower()
    buffer = io.BytesIO(file_bytes)

    if suffix == ".csv":
        # Padukan CSV sebagai single sheet agar alur tetap konsisten.
        df = pd.read_csv(buffer, encoding=encoding)
        sheet_name = os.path.splitext(os.path.basename(filename))[0] or "Sheet1"
        return {sheet_name: df}

    with pd.ExcelFile(buffer) as xls:
        available_sheets = list(xls.sheet_names)
        if sheets is None:
            target_sheets = available_sheets
        else:
            target_sheets = [sheet for sheet in sheets if sheet in available_sheets]

        workbook: Dict[str, pd.DataFrame] = {}
        for sheet_name in target_sheets:
            workbook[sheet_name] = xls.parse(sheet_name)
        return workbook


def validate_columns(df: pd.DataFrame) -> List[str]:
    """Mengembalikan daftar kolom yang belum tersedia pada dataframe."""
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]
