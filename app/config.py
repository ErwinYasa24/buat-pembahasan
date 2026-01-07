"""Konfigurasi dasar untuk aplikasi pembahasan."""

import os
from typing import Dict, List, Tuple

DEFAULT_MODEL = "gemini-2.0-flash"
# Ganti nilai berikut dengan API key Gemini Anda bila tidak memakai environment variable.
# Contoh: GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "AIz..."
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "ISI_API_KEY"

REQUIRED_COLUMNS: List[str] = [
    "no",
    "id",
    "category",
    "question_categories_id",
    "sub_category",
    "sub_category_id",
    "program",
    "parent_id",
    "question",
    "answer_header_true",
    "answer_header_false",
    "option_a_text",
    "option_a_score",
    "option_b_text",
    "option_b_score",
    "option_c_text",
    "option_c_score",
    "option_d_text",
    "option_d_score",
    "option_e_text",
    "option_e_score",
    "option_number",
    "option_number_score",
    "explanation",
    "explanation_media",
    "tags",
    "answer_type",
    "question_keyword",
    "question_type",
    "type",
    "bloom_level",
    "in_modules",
    "explanation_ai",
]

OPTION_COLUMNS: Dict[str, Tuple[str, str]] = {
    "A": ("option_a_text", "option_a_score"),
    "B": ("option_b_text", "option_b_score"),
    "C": ("option_c_text", "option_c_score"),
    "D": ("option_d_text", "option_d_score"),
    "E": ("option_e_text", "option_e_score"),
    "Pilihan": ("option_number", "option_number_score"),
}
