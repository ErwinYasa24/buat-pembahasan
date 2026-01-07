"""Utilitas untuk membangun prompt dan memanggil model AI."""

import html
import json
import re
from typing import Dict, List, Optional, Sequence

import pandas as pd
import streamlit as st

from .config import GEMINI_API_KEY, OPTION_COLUMNS

OPTION_TEXT_LABELS = [label for label in OPTION_COLUMNS.keys() if label != "Pilihan"]
MIN_NUMERIC_PARAGRAPHS = 4
TABLE_STYLE = "border-collapse:collapse; table-layout:auto; width:50%"
TABLE_CELL_STYLE = "text-align:center; white-space:nowrap"


def _sanitize_text(value: str) -> str:
    """Remove HTML tags, entities, and collapse whitespace without altering wording."""

    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_label(value: object) -> str:
    """Normalize category/subcategory strings for comparisons."""

    if value is None:
        return ""
    text = str(value)
    return re.sub(r"\s+", " ", text).strip().lower()


def _should_skip_row(category_value: object, sub_category_value: object) -> Optional[str]:
    """Return reason if a row must be skipped based on TIU subcategories."""

    category_norm = _normalize_label(category_value)
    if category_norm != "tiu":
        return None

    sub_norm = _normalize_label(sub_category_value)
    if sub_norm.startswith("figural"):
        return "subkategori TIU figural di-skip."
    if sub_norm.startswith("numerik deret"):
        return "subkategori TIU numerik deret di-skip."
    return None


def _is_tiu_numerik(category_value: object, sub_category_value: object) -> bool:
    """Return True if category is TIU and subcategory is Numerik."""

    category_norm = _normalize_label(category_value)
    if category_norm != "tiu":
        return False
    sub_norm = _normalize_label(sub_category_value)
    return sub_norm.startswith("numerik") and not sub_norm.startswith("numerik deret")


def _is_verbal_silogisme(category_value: object, sub_category_value: object) -> bool:
    category_norm = _normalize_label(category_value)
    sub_norm = _normalize_label(sub_category_value)
    return category_norm == "tiu" and sub_norm == "verbal silogisme"


def _is_verbal_analogi(category_value: object, sub_category_value: object) -> bool:
    category_norm = _normalize_label(category_value)
    sub_norm = _normalize_label(sub_category_value)
    return category_norm == "tiu" and sub_norm == "verbal analogi"


def _is_verbal_analitis(category_value: object, sub_category_value: object) -> bool:
    category_norm = _normalize_label(category_value)
    sub_norm = _normalize_label(sub_category_value)
    return category_norm == "tiu" and sub_norm == "verbal analitis"


def _extract_option_scores(row: pd.Series) -> List[Optional[float]]:
    """Return numeric scores for options A-E in order."""

    scores: List[Optional[float]] = []
    for label in OPTION_TEXT_LABELS:
        _, score_col = OPTION_COLUMNS[label]
        score = row.get(score_col)
        if pd.isna(score):
            scores.append(None)
            continue
        try:
            numeric = float(score)
        except (TypeError, ValueError):
            scores.append(None)
            continue
        scores.append(numeric)
    return scores


def _format_score(score: Optional[float]) -> Optional[str]:
    """Format numeric score without trailing zeros."""

    if score is None:
        return None
    text = f"{score}".rstrip("0").rstrip(".")
    return text or "0"


def _score_sort_key(idx: int, scores: List[Optional[float]]) -> tuple[float, int]:
    score = scores[idx] if idx < len(scores) else None
    score_value = score if score is not None else float("-inf")
    return (-score_value, idx)


def _order_indices(
    indices: Sequence[int],
    scores: List[Optional[float]],
    option_map: List[str],
) -> List[int]:
    """Order option indices from highest to lowest score, stable by option order."""

    seen: set[int] = set()
    filtered: List[int] = []
    for idx in indices:
        if idx in seen or idx >= len(option_map):
            continue
        if not option_map[idx]:
            continue
        seen.add(idx)
        filtered.append(idx)
    return sorted(filtered, key=lambda idx: _score_sort_key(idx, scores))


def _label_with_score(option_text: str, score: Optional[float], include_score: bool) -> str:
    label = option_text.strip()
    if include_score:
        score_text = _format_score(score)
        if score_text:
            label = f"{label} ({score_text})"
    return label


def _ensure_trailing_period(text: str) -> str:
    if not text:
        return text
    if text.endswith((".", "!", "?")):
        return text
    return f"{text}."


def _wrap_math_tex(text: str) -> str:
    """Wrap inline TeX in a math-tex span for TIU numerik output."""

    if not text:
        return text
    wrapped = re.sub(
        r"(\\\(.+?\\\))",
        r'<span class="math-tex">\1</span>',
        text,
    )
    wrapped = re.sub(
        r"(\\\[.+?\\\])",
        r'<span class="math-tex">\1</span>',
        wrapped,
    )
    return wrapped


def _split_numeric_paragraphs(text: str) -> List[str]:
    cleaned = text.strip()
    cleaned = re.sub(r"^\s*<p[^>]*>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</p>\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?span[^>]*>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<br\s*/?>", "\n", cleaned, flags=re.IGNORECASE)
    parts = [part.strip() for part in cleaned.splitlines()]
    return [part for part in parts if part]


def _truncate_text(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... (dipotong)"


def _format_paragraph(text: str, styled: bool) -> str:
    if styled:
        return f"<p style=\"text-align:justify\">{text}</p>"
    return f"<p>{text}</p>"


def _is_escaped(text: str, idx: int) -> bool:
    backslashes = 0
    cursor = idx - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def _repair_json_text(raw_text: str) -> str:
    """Escape invalid JSON backslashes and newlines within strings."""

    result: List[str] = []
    in_string = False
    idx = 0
    length = len(raw_text)
    while idx < length:
        char = raw_text[idx]
        if char == "\"" and not _is_escaped(raw_text, idx):
            in_string = not in_string
            result.append(char)
            idx += 1
            continue
        if in_string:
            if char == "\n":
                result.append("\\n")
                idx += 1
                continue
            if char == "\r":
                result.append("\\r")
                idx += 1
                continue
            if char == "\\":
                if idx + 1 < length:
                    nxt = raw_text[idx + 1]
                    if nxt in "\"\\/bfnrtu":
                        result.append(char)
                        result.append(nxt)
                        idx += 2
                        continue
                result.append("\\\\")
                idx += 1
                continue
        result.append(char)
        idx += 1
    return "".join(result)


def _dedupe_paragraphs(paragraphs: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for paragraph in paragraphs:
        text = str(paragraph).strip()
        if not text:
            continue
        normalized = re.sub(r"\s+", " ", text).strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(text)
    return result


def _normalize_detail_paragraphs(value: object) -> List[str]:
    if isinstance(value, str):
        paragraphs = [value]
    elif value is None:
        paragraphs = []
    else:
        paragraphs = list(value)
    return [str(item).strip() for item in paragraphs if str(item).strip()]


def _parse_response(raw_text: str) -> Optional[Dict[str, object]]:
    parsed: Optional[Dict[str, object]] = None
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        repaired = _repair_json_text(raw_text)
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = raw_text[start : end + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    repaired_candidate = _repair_json_text(candidate)
                    try:
                        parsed = json.loads(repaired_candidate)
                    except json.JSONDecodeError:
                        parsed = None

    return parsed


_REASON_PREFIXES = [
    re.compile(r"^\s*jawaban\s+yang\s+kurang\s+tepat\s*[:\-]*\s*", re.IGNORECASE),
    re.compile(r"^\s*jawaban\s+kurang\s+tepat\s*[:\-]*\s*", re.IGNORECASE),
    re.compile(r"^\s*jawaban\s+salah\s*[:\-]*\s*", re.IGNORECASE),
    re.compile(r"^\s*opsi\s+salah\s+[A-E0-9]+\s*[:\-]*\s*", re.IGNORECASE),
    re.compile(r"^\s*-\s*(opsi|pilihan)\s+[A-E0-9]+\s*[:\-]*\s*", re.IGNORECASE),
    re.compile(r"^\s*(opsi|pilihan)\s+[A-E0-9]+\s*[:\-]*\s*", re.IGNORECASE),
]


def _strip_reason_prefix(reason: str) -> str:
    """Remove repeated labels like 'Jawaban yang kurang tepat' or '- Opsi 1:' from reasons."""

    cleaned = reason.strip()
    while True:
        updated = cleaned
        for pattern in _REASON_PREFIXES:
            match = pattern.match(updated)
            if match:
                updated = updated[match.end() :].strip()
        updated = updated.lstrip("-: ").strip()
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned


def _strip_option_echo(reason: str, option_text: str) -> str:
    """Remove duplicated option text at start of reason."""

    cleaned_reason = reason.strip()
    option_clean = option_text.strip()
    if not option_clean:
        return cleaned_reason

    lowered_option = option_clean.lower()

    while True:
        lowered_reason = cleaned_reason.lower()
        if not lowered_reason.startswith(lowered_option):
            break
        trimmed = cleaned_reason[len(option_clean) :].lstrip(" ,:.-").strip()
        if not trimmed:
            cleaned_reason = ""
            break
        cleaned_reason = trimmed

    return cleaned_reason


def _capitalize_sentence(text: str) -> str:
    """Capitalize the first alphabetic character unless preceded by a colon."""

    cleaned = text.strip()
    if not cleaned:
        return cleaned

    prefix, sep, suffix = cleaned.partition(":")
    if sep and suffix.strip():
        return cleaned

    for idx, char in enumerate(cleaned):
        if char.isalpha():
            return cleaned[:idx] + char.upper() + cleaned[idx + 1 :]
        if char.isdigit():
            return cleaned

    return cleaned


def _extract_proper_tokens(*texts: str) -> set[str]:
    """Collect capitalized tokens to preserve capitalization (names, etc.)."""

    tokens: set[str] = set()
    for text in texts:
        if not text:
            continue
        for token in re.findall(r"[A-Za-zÀ-ÿ']+", text):
            if token and token[0].isupper():
                tokens.add(token)
    return tokens


def _normalize_reason_capital(text: str, preserve_tokens: set[str]) -> str:
    """Ensure sentences after colon start lowercase unless preserved."""

    original = text
    if not text:
        return original

    leading_spaces = len(text) - len(text.lstrip())
    working = text[leading_spaces:]

    prefix, sep, suffix = working.partition(":")
    if sep and suffix:
        rest = suffix.lstrip()
        if rest.lower().startswith("adalah "):
            rest = rest[len("adalah ") :]
        if rest:
            match = re.match(r"[A-Za-zÀ-ÿ']+", rest)
            if match:
                word = match.group(0)
                if word not in preserve_tokens and word.lower() != "anda":
                    rest = word.lower() + rest[len(word) :]
        space_after = suffix[: len(suffix) - len(suffix.lstrip())] or " "
        if not rest:
            return original
        new_working = prefix + sep + space_after + rest
        return text[:leading_spaces] + new_working

    cleaned = working
    idx = 0
    while idx < len(cleaned) and not cleaned[idx].isalpha():
        idx += 1
    if idx >= len(cleaned):
        return original

    match = re.match(r"[A-Za-zÀ-ÿ']+", cleaned[idx:])
    if not match:
        return original

    word = match.group(0)
    if word in preserve_tokens or word.lower() == "anda":
        return original

    lowered = word[0].lower() + word[1:]
    return text[: leading_spaces + idx] + lowered + cleaned[idx + len(word) :]


def _enrich_reason(
    option_text: str,
    reason: str,
    correct_text: str,
    question_summary: str,
) -> str:
    """Ensure alasan opsi salah cukup detail (minimal dua kalimat)."""

    cleaned_reason = reason.strip()
    if cleaned_reason and not cleaned_reason.endswith(('.', '!', '?')):
        cleaned_reason += '.'

    sentence_count = cleaned_reason.count('.') + cleaned_reason.count('!') + cleaned_reason.count('?')
    word_count = len(cleaned_reason.split())

    supplements: List[str] = []
    if sentence_count < 2 or word_count < 25:
        if option_text and correct_text:
            supplements.append(
                f"penjelasan ini masih menyoroti {option_text} tanpa mengaitkannya dengan inti soal mengenai {correct_text}."
            )
        elif correct_text:
            supplements.append(
                f"penjelasan ini belum menunjukkan keterkaitan dengan tuntutan soal tentang {correct_text}."
            )
        else:
            supplements.append(
                "penjelasan perlu menyebutkan secara spesifik mengapa pilihan ini tidak memenuhi kebutuhan soal."
            )

    enriched = cleaned_reason
    if supplements:
        enriched = enriched if enriched else ""
        if enriched and not enriched.endswith(' '):
            enriched += ' '
        enriched += ' '.join(supplements)

    return enriched.strip()


def _infer_correct_indices(row: pd.Series, option_texts: List[str]) -> List[int]:
    """Determine correct option indices, prioritizing explicit scores."""

    indices: List[int] = []
    mapping = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "PILIHAN A": 0,
        "PILIHAN B": 1,
        "PILIHAN C": 2,
        "PILIHAN D": 3,
        "PILIHAN E": 4,
    }

    scored_options: List[tuple[int, float]] = []
    option_keys = list(OPTION_COLUMNS.keys())
    for idx, (_, (_, score_col)) in enumerate(OPTION_COLUMNS.items()):
        score = row.get(score_col)
        if pd.isna(score):
            continue
        try:
            numeric = float(score)
        except (TypeError, ValueError):
            continue
        scored_options.append((idx, numeric))

    if scored_options:
        max_score = max(value for _, value in scored_options)
        for idx, value in scored_options:
            if value != max_score:
                continue
            label = option_keys[idx]
            if label == "Pilihan":
                option_letter = str(row.get("option_number", "")).strip().upper()
                mapped = mapping.get(option_letter)
                if mapped is not None:
                    indices.append(mapped)
            else:
                indices.append(idx)

    if indices:
        return list(dict.fromkeys(indices))

    option_number = str(row.get("option_number", "")).strip().upper()
    if option_number in mapping and mapping[option_number] < len(option_texts):
        return [mapping[option_number]]

    answer_true = row.get("answer_header_true")
    if pd.notna(answer_true):
        cleaned_answer = _sanitize_text(answer_true).lower()
        for idx, text in enumerate(option_texts):
            candidate = text.lower().strip()
            if not candidate:
                continue
            if (
                cleaned_answer == candidate
                or cleaned_answer.endswith(candidate)
                or candidate in cleaned_answer
            ):
                indices.append(idx)
        if indices:
            return indices

    explanation_text = row.get("explanation") or row.get("explanation_ai")
    if pd.notna(explanation_text):
        cleaned_explanation = _sanitize_text(explanation_text).lower()
        match = re.search(
            r"jawaban yang (?:benar|tepat)\s*[:\-]\s*(.+)",
            cleaned_explanation,
        )
        if match:
            answer_segment = match.group(1)
            for idx, text in enumerate(option_texts):
                candidate = text.lower().strip()
                if not candidate:
                    continue
                if answer_segment.startswith(candidate) or candidate in answer_segment:
                    indices.append(idx)
                    break
        if indices:
            return indices

    return []


def build_prompt(row: pd.Series) -> Dict[str, object]:
    question_text = str(row.get("question", "")).strip()
    question_summary = _sanitize_text(question_text)
    option_map: List[str] = []
    option_instruction_rows: List[str] = []
    for idx, label in enumerate(OPTION_TEXT_LABELS):
        text_col, _ = OPTION_COLUMNS[label]
        option_text = row.get(text_col)
        cleaned = ""
        if pd.notna(option_text) and str(option_text).strip():
            cleaned = _sanitize_text(option_text)
            option_instruction_rows.append(f"Opsi {idx + 1}: {cleaned}")
        option_map.append(cleaned)

    tags = row.get("tags")
    metadata_parts = [
        f"Kategori: {row.get('category')}" if pd.notna(row.get("category")) else None,
        f"Subkategori: {row.get('sub_category')}" if pd.notna(row.get("sub_category")) else None,
        f"Program: {row.get('program')}" if pd.notna(row.get("program")) else None,
        f"Tag: {tags}" if pd.notna(tags) else None,
        f"Keyword: {row.get('question_keyword')}" if pd.notna(row.get("question_keyword")) else None,
    ]
    metadata = "\n".join(part for part in metadata_parts if part)

    omit_incorrect = _is_tiu_numerik(row.get("category"), row.get("sub_category"))
    is_verbal_silogisme = _is_verbal_silogisme(row.get("category"), row.get("sub_category"))
    is_verbal_analogi = _is_verbal_analogi(row.get("category"), row.get("sub_category"))
    is_verbal_analitis = _is_verbal_analitis(row.get("category"), row.get("sub_category"))

    if omit_incorrect:
        format_template = (
            "<p style=\"text-align:justify\"><strong>Jawaban yang tepat: ...</strong></p>\n"
            "<p style=\"text-align:justify\">Penjelasan naratif dengan line break...</p>"
        )
        intro_text = (
            "Gunakan gaya bahasa formal, edukatif, dan jelas dengan langkah penyelesaian yang runtut."
        )
    elif is_verbal_silogisme:
        format_template = (
            "<p style=\"text-align:justify\"><strong>Jawaban yang tepat: ...</strong></p>\n"
            "<p style=\"text-align:justify\">Premis:</p>\n"
            "<p style=\"text-align:justify\">Simbol logika:</p>\n"
            "<p style=\"text-align:justify\">Proses penarikan kesimpulan:</p>\n"
            "<p style=\"text-align:justify\">Kesimpulan akhir:</p>\n"
            "<p style=\"text-align:justify\"><strong>Jawaban yang kurang tepat:</strong></p>\n"
            "<p style=\"text-align:justify\">- <strong>Opsi salah 1:</strong> alasan singkat...</p>"
        )
        intro_text = (
            "Pembahasan harus jelas, terstruktur, dan mudah dimengerti dengan penjelasan singkat "
            "menggunakan konsep silogisme."
        )
    elif is_verbal_analogi:
        format_template = (
            "<p style=\"text-align:justify\"><strong>Jawaban yang tepat: ...</strong></p>\n"
            "<p style=\"text-align:justify\"><strong>Alasan:</strong></p>\n"
            "<p style=\"text-align:justify\"><strong>1. Poin alasan pertama:</strong> penjelasan...</p>\n"
            "<p style=\"text-align:justify\"><strong>2. Poin alasan kedua:</strong> penjelasan...</p>\n"
            "<p style=\"text-align:justify\"><strong>3. Poin alasan ketiga:</strong> penjelasan...</p>\n"
            "<p style=\"text-align:justify\"><strong>Jawaban yang kurang tepat:</strong></p>\n"
            "<p style=\"text-align:justify\">- <strong>Opsi salah 1:</strong> alasan singkat...</p>"
        )
        intro_text = (
            "Pembahasan harus jelas dan mudah dimengerti. Berikan alasan jawaban benar dalam "
            "2-3 poin terstruktur."
        )
    elif is_verbal_analitis:
        format_template = (
            "<p style=\"text-align:justify\"><strong>Jawaban yang tepat: ...</strong></p>\n"
            "<p style=\"text-align:justify\">Paragraf penjelasan runtut...</p>\n"
            "<p style=\"text-align:justify\">Paragraf penjelasan lanjutan...</p>\n"
            "<table border=\"1\" cellpadding=\"0\" cellspacing=\"0\" style=\""
            + TABLE_STYLE
            + "\">...</table>\n"
            "<p style=\"text-align:justify\">Kalimat ringkasan setelah tabel...</p>\n"
            "<p style=\"text-align:justify\"><strong>Jawaban yang kurang tepat:</strong></p>\n"
            "<p style=\"text-align:justify\">- <strong>Opsi salah 1:</strong> alasan singkat...</p>"
        )
        intro_text = (
            "Pembahasan harus jelas, runtut, dan mudah dimengerti dengan penjelasan naratif "
            "tanpa bullet atau numbering."
        )
    else:
        format_template = (
            "<p style=\"text-align:justify\"><strong>Jawaban yang tepat: ...</strong></p>\n"
            "<p style=\"text-align:justify\">Paragraf penjelasan lanjutan...</p>\n"
            "<p style=\"text-align:justify\"><strong>Jawaban yang kurang tepat:</strong></p>\n"
            "<p style=\"text-align:justify\">- <strong>Opsi salah 1:</strong> alasan singkat...</p>\n"
            "<p style=\"text-align:justify\">- <strong>Opsi salah 2:</strong> alasan singkat...</p>"
        )
        intro_text = (
            "Pembahasan harus jelas dan mudah dimengerti. Jelaskan alasan jawaban benar secara menyeluruh "
            "(minimal 3 kalimat) dan uraikan mengapa tiap opsi salah tidak memenuhi kriteria."
        )

    instructions = (
        "Tulis pembahasan dalam bahasa Indonesia menggunakan HTML tanpa menambahkan <!DOCTYPE>, <html>, <head>, atau <body>. "
        f"{intro_text} "
        "Output harus berupa rangkaian tag <p> seperti contoh berikut dan tidak boleh memiliki teks di luar tag tersebut.\n\n"
        f"Format wajib diikuti:\n{format_template}\n\n"
        "Aturan tambahan:\n"
        "- Paragraf pertama harus menyatakan jawaban benar dengan awalan 'Jawaban yang tepat:' diikuti penjelasan singkat. Sertakan teks opsi benar secara utuh sebelum penjelasan.\n"
        "- Paragraf kedua (dan tambahan bila perlu) menjelaskan alasan jawaban benar secara detail (minimal 2 kalimat).\n"
        "- Jangan menuliskan label huruf seperti A/B/C di dalam isi jawaban. Fokus pada isi opsi saja.\n"
        "- Jangan menambahkan bobot atau skor ke dalam teks opsi maupun alasan.\n"
        "- Semua nilai string dalam JSON harus berupa teks polos tanpa tag HTML, kecuali `table_html` jika diminta.\n"
        "- Khusus TIU Numerik, boleh menggunakan tag `<strong>` dan `<em>` di dalam `detail_paragraphs`.\n"
    )

    if omit_incorrect:
        instructions += (
            "\n- Khusus TIU subkategori Numerik, fokus pada jawaban yang tepat saja. "
            "Isi `incorrect_reasons` dengan objek kosong {} dan jangan memberikan alasan opsi salah.\n"
            "- Jelaskan langkah demi langkah secara naratif dengan gaya formal dan edukatif.\n"
            "- Setiap elemen `detail_paragraphs` adalah satu paragraf terpisah.\n"
            "- Pisahkan kalimat penjelasan dan rumus ke paragraf terpisah (rumus berdiri sendiri di paragraf sendiri).\n"
            "- Jangan gunakan `<br />`, `<ol>`, `<ul>`, atau tag `<p>` di dalam `detail_paragraphs`.\n"
            "- Boleh menggunakan tag `<strong>` dan `<em>` untuk penekanan sederhana di dalam paragraf.\n"
            "- Hindari label seperti 'Langkah 1/2/3' dan jangan gunakan <ol>/<ul>.\n"
            "- Isi `correct_summary` hanya dengan jawaban yang tepat tanpa penjelasan tambahan.\n"
            r"- Gunakan MathTeX inline dengan pembungkus `\( ... \)` untuk rumus di dalam kalimat.\n"
            "- Jangan menuliskan tag `<span class=\"math-tex\">` karena sistem akan membungkus otomatis.\n"
            "- Jika ada rumus umum, tulis kalimat '... dihitung dengan rumus' lalu tampilkan rumus, "
            "lanjutkan dengan paragraf 'Substitusikan nilainya:', 'Hitung selisihnya:', "
            "'Sederhanakan pecahan:', dan tutup dengan paragraf kesimpulan.\n"
            "- Pastikan JSON valid dengan meng-escape backslash sebagai `\\\\` di dalam string JSON."
        )
    elif is_verbal_silogisme:
        instructions += (
            "\n- Jelaskan jawaban secara singkat dan terstruktur menggunakan konsep silogisme.\n"
            "- Identifikasi seluruh premis (premis mayor, minor, tambahan jika ada) dan jelaskan "
            "masing-masing dalam kalimat sehari-hari.\n"
            "- Representasikan setiap premis ke dalam simbol logika (P, Q, R, dst) sebagai rumus bantu "
            "dan jelaskan makna setiap simbol.\n"
            "- Jelaskan proses penarikan kesimpulan secara bertahap dan naratif, tekankan hubungan logis "
            "antar premis serta batasan kesimpulan.\n"
            "- Akhiri dengan kesimpulan akhir yang eksplisit dan mudah dipahami.\n"
            "- Gunakan bahasa sederhana, logis, edukatif; hindari simbol logika formal tanpa penjelasan konsep.\n"
            "- Bagi `detail_paragraphs` menjadi 4 paragraf: premis, simbol, proses, kesimpulan."
        )
    elif is_verbal_analogi:
        instructions += (
            "\n- Buat paragraf kedua berisi '<strong>Alasan:</strong>'.\n"
            "- Sampaikan 2-3 poin alasan sebagai paragraf terpisah dengan format "
            "`<strong>1. ...:</strong> penjelasan...`, lanjut ke poin 2 dan 3.\n"
            "- Setiap poin alasan harus menjelaskan hubungan inti analogi secara ringkas dan jelas."
        )
    elif is_verbal_analitis:
        instructions += (
            "\n- Gunakan paragraf naratif tanpa bullet/numbering; jangan gunakan <ol>, <ul>, "
            "atau awalan 1/2/3.\n"
            "- Jelaskan penempatan jadwal atau urutan secara bertahap dalam 3-5 paragraf.\n"
            "- Jika memungkinkan, sertakan tabel jadwal dalam `table_html` (boleh string kosong jika tidak ada).\n"
            f"- Gunakan tabel HTML dengan style table ` {TABLE_STYLE} ` dan setiap <td> memakai "
            f"style `{TABLE_CELL_STYLE}` serta `border=\"1\" cellpadding=\"0\" cellspacing=\"0\"`.\n"
            "- Isi `detail_paragraphs` tetap berupa teks polos tanpa tag HTML.\n"
            "- `table_html` hanya berisi markup tabel (tanpa tag <p>)."
        )
    incorrect_instructions = (
        "\n- Tambahkan paragraf khusus dengan teks 'Jawaban yang kurang tepat:' (bold) setelah penjelasan jawaban benar.\n"
        "- Tambahkan paragraf terpisah untuk setiap opsi salah dengan format '- <strong>...:</strong> penjelasan...'. "
        "Teks sebelum titik dua HARUS persis menyalin isi opsi tanpa perubahan atau sinonim. "
        "Setiap alasan minimal dua kalimat yang jelas.\n"
        "- Setelah titik dua pada opsi salah, lanjutkan kalimat dengan huruf kecil kecuali untuk nama diri atau kata 'Anda'. "
        "Hindari mengawali dengan kata 'adalah'.\n"
        "- Jangan menulis ulang opsi yang benar di bagian opsi salah.\n"
        "- Nilai dalam `incorrect_reasons` adalah penjelasan mendalam (minimal dua kalimat) untuk tiap opsi salah tanpa menyalin ulang teks opsi.\n"
        "- Nilai `correct_summary` hanya berisi penjelasan singkat (tanpa kembali menuliskan frasa 'Jawaban yang tepat'). "
        "Jika tidak ada penjelasan tambahan, kosongkan string tersebut.\n"
        "- Gunakan style 'text-align:justify' pada setiap tag <p> dan <strong> untuk penekanan."
    )
    if not omit_incorrect:
        instructions += incorrect_instructions

    option_instructions = "\n".join(option_instruction_rows)
    if not option_instructions:
        option_instructions = "(Tidak ada pilihan)"

    correct_indices = _infer_correct_indices(row, option_map)
    if correct_indices:
        textful = [idx for idx in correct_indices if idx < len(option_map) and option_map[idx]]
        if textful:
            leftovers = [idx for idx in correct_indices if idx not in textful]
            correct_indices = textful + leftovers

    incorrect_indices = [
        idx for idx in range(len(option_map))
        if idx not in correct_indices and idx < len(option_map) and option_map[idx]
    ]

    correct_option_display = "\n".join(
        f"Opsi {idx + 1}: {option_map[idx]}"
        for idx in correct_indices
        if idx < len(option_map) and option_map[idx]
    ) or "(Tidak diketahui)"

    schema_lines = [
        "{",
        "  \"correct_summary\": string,",
        "  \"detail_paragraphs\": [string, ...],",
    ]
    if is_verbal_analitis:
        schema_lines.append("  \"table_html\": string,")
    schema_lines.extend(
        [
            "  \"incorrect_reasons\": {",
            "       \"1\": string alasan (untuk opsi indeks 1 jika salah),",
            "       ...",
            "   }",
            "}",
        ]
    )
    schema = "\n".join(schema_lines)

    context = (
        instructions
        + "\n\n"
        + metadata
        + f"\n\nSoal:\n{question_text}\n\nDaftar opsi (gunakan apa adanya untuk bagian opsi salah):\n{option_instructions}\n"
        + f"Opsi benar (copy teksnya secara utuh ketika menyusun ringkasan):\n{correct_option_display}\n"
        + f"Opsi salah yang wajib dijelaskan: {', '.join(str(idx + 1) for idx in incorrect_indices)}\n"
        + "Kembalikan respons dalam format JSON PERSIS seperti berikut tanpa teks tambahan atau blok kode:\n"
        f"{schema}\n"
        "Output TIDAK boleh diawali atau diakhiri dengan karakter selain tanda kurung kurawal JSON."
    )

    return {
        "prompt": context,
        "option_map": option_map,
        "correct_indices": correct_indices,
        "incorrect_indices": incorrect_indices,
        "question_summary": question_summary,
    }


def generate_ai_explanations(
    df: pd.DataFrame,
    target_indices: Sequence[object],
    model_name: str,
    api_key: Optional[str] = None,
) -> List[int]:
    try:
        import google.generativeai as genai
    except ImportError:
        st.error(
            "Paket `google-generativeai` belum terpasang. Jalankan `pip install google-generativeai` terlebih dahulu."
        )
        return []

    effective_key = (api_key or GEMINI_API_KEY or "").strip()
    if not effective_key or "ISI_API_KEY" in effective_key:
        secret_key = st.secrets.get("gemini_api_key")
        if secret_key:
            effective_key = str(secret_key).strip()

    if not effective_key or "ISI_API_KEY" in effective_key:
        st.error(
            "API key Gemini belum dikonfigurasi. Set environment variable `GEMINI_API_KEY` "
            "atau simpan di `st.secrets['gemini_api_key']`."
        )
        return []

    try:
        genai.configure(api_key=effective_key)
        model = genai.GenerativeModel(model_name)
    except Exception as exc:  # pragma: no cover - konfigurasi gagal
        st.error(f"Gagal menginisialisasi model Gemini: {exc}")
        return []

    updated_rows: List[int] = []
    progress = st.progress(0.0)
    status_text = st.empty()

    for counter, row_idx in enumerate(target_indices, start=1):
        row = df.loc[row_idx]

        try:
            skip_reason = _should_skip_row(row.get("category"), row.get("sub_category"))
            if skip_reason:
                question_label = row.get("no") or row_idx
                st.info(f"Lewati nomor {question_label}: {skip_reason}")
                continue

            prompt_data = build_prompt(row)
            prompt = prompt_data["prompt"]
            option_map: List[str] = prompt_data["option_map"]
            correct_indices: List[int] = prompt_data["correct_indices"]
            incorrect_indices: List[int] = prompt_data["incorrect_indices"]
            question_summary: str = prompt_data.get("question_summary", "")
            option_scores = _extract_option_scores(row)
            include_scores = _normalize_label(row.get("category")) == "tkp"
            is_tiu_numerik = _is_tiu_numerik(
                row.get("category"),
                row.get("sub_category"),
            )
            is_verbal_analitis = _is_verbal_analitis(
                row.get("category"),
                row.get("sub_category"),
            )
            include_incorrect = not is_tiu_numerik

            correct_indices = _order_indices(correct_indices, option_scores, option_map)
            incorrect_indices = _order_indices(incorrect_indices, option_scores, option_map)

            if not correct_indices:
                question_label = row.get("no") or row_idx
                st.warning(
                    f"Lewati nomor {question_label}: tidak menemukan jawaban benar pada data sumber."
                )
                continue

            try:
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "response_mime_type": "application/json",
                        "max_output_tokens": 2048,
                    },
                )
            except Exception:
                response = model.generate_content(prompt)
            raw_text = (response.text or "").strip()
            raw_text = raw_text.replace("```,", "```")
            if raw_text.startswith("```"):
                raw_text = raw_text.strip()
                if raw_text.startswith("```json"):
                    raw_text = raw_text[len("```json"):]
                raw_text = raw_text.strip("`").strip()

            data = _parse_response(raw_text)
            if data is None:
                question_label = row.get("no") or row_idx
                st.warning(
                    f"Format respons tidak valid untuk nomor {question_label}. Mengabaikan pembaruan."
                )
                if raw_text:
                    with st.expander(f"Detail respons AI untuk nomor {question_label}"):
                        st.code(_truncate_text(raw_text), language="text")
                continue

            if is_tiu_numerik:
                candidate_paragraphs = _normalize_detail_paragraphs(
                    data.get("detail_paragraphs")
                )
                candidate_paragraphs = _dedupe_paragraphs(candidate_paragraphs)
                if len(candidate_paragraphs) < MIN_NUMERIC_PARAGRAPHS:
                    retry_prompt = (
                        prompt
                        + "\n\nPERBAIKI: Outputkan JSON valid dan "
                        f"minimal {MIN_NUMERIC_PARAGRAPHS} detail_paragraphs yang berbeda."
                    )
                    try:
                        retry_response = model.generate_content(
                            retry_prompt,
                            generation_config={
                                "response_mime_type": "application/json",
                                "max_output_tokens": 2048,
                            },
                        )
                    except Exception:
                        retry_response = model.generate_content(retry_prompt)
                    retry_text = (retry_response.text or "").strip()
                    retry_text = retry_text.replace("```,", "```")
                    if retry_text.startswith("```"):
                        retry_text = retry_text.strip()
                        if retry_text.startswith("```json"):
                            retry_text = retry_text[len("```json"):]
                        retry_text = retry_text.strip("`").strip()
                    retry_data = _parse_response(retry_text)
                    if retry_data is not None:
                        data = retry_data

            correct_summary = str(data.get("correct_summary", "")).strip()
            detail_paragraphs = _normalize_detail_paragraphs(
                data.get("detail_paragraphs")
            )
            incorrect_reasons = data.get("incorrect_reasons") or {}

            if is_tiu_numerik:
                detail_paragraphs = _dedupe_paragraphs(detail_paragraphs)

            html_parts: List[str] = []

            if not correct_summary and correct_indices:
                first_idx = correct_indices[0]
                if first_idx < len(option_map):
                    correct_summary = option_map[first_idx]

            main_option = ""
            main_score: Optional[float] = None
            if correct_indices:
                idx0 = correct_indices[0]
                if idx0 < len(option_map):
                    main_option = option_map[idx0].strip()
                    if idx0 < len(option_scores):
                        main_score = option_scores[idx0]

            primary_text = ""
            explanation = ""

            use_style = True

            if correct_summary or main_option:
                correct_text = _sanitize_text(correct_summary)
                correct_text = re.sub(
                    r"^jawaban yang tepat[\s:,-]*", "", correct_text, flags=re.IGNORECASE
                ).strip()

                if not main_option and correct_text:
                    lowered = correct_text.lower()
                    for text in option_map:
                        if text and text.lower() == lowered:
                            main_option = text
                            break

                if main_option:
                    primary_text = main_option
                    explanation_core = _strip_option_echo(correct_text, main_option)
                else:
                    primary_text = correct_text
                    explanation_core = ""

                primary_text = primary_text.strip()
                explanation_core = explanation_core.strip()

                if explanation_core and explanation_core.lower() == primary_text.lower():
                    explanation_core = ""

                if primary_text:
                    primary_label = _label_with_score(primary_text, main_score, include_scores)
                    correct_line = f"Jawaban yang tepat: {primary_label}"
                    if not is_tiu_numerik:
                        correct_line = _ensure_trailing_period(correct_line)
                    if is_tiu_numerik:
                        correct_line = _wrap_math_tex(correct_line)
                    html_parts.append(
                        _format_paragraph(f"<strong>{correct_line}</strong>", styled=use_style)
                    )

                if explanation_core and not detail_paragraphs:
                    explanation_sentence = explanation_core.strip()
                    if explanation_sentence:
                        explanation_sentence = explanation_sentence[0].upper() + explanation_sentence[1:]
                        if explanation_sentence[-1] not in ".!?":
                            explanation_sentence += "."
                        detail_paragraphs = [explanation_sentence]

            explanation_paragraphs_added = 0
            for paragraph in detail_paragraphs:
                raw_paragraph = str(paragraph).strip()
                if not raw_paragraph:
                    continue
                if is_verbal_analitis and "<table" in raw_paragraph.lower():
                    html_parts.append(raw_paragraph)
                    explanation_paragraphs_added += 1
                    continue
                plain_check = _sanitize_text(raw_paragraph)
                if not plain_check:
                    continue
                lowered = plain_check.lower()
                if (
                    lowered.startswith("jawaban yang tepat")
                    or lowered.startswith("jawaban yang kurang tepat")
                    or lowered.startswith("- opsi")
                    or lowered.startswith("opsi")
                    or lowered.startswith("- pilihan")
                    or lowered.startswith("- ")
                ):
                    continue
                if is_tiu_numerik:
                    numeric_parts = _split_numeric_paragraphs(raw_paragraph)
                    for part in numeric_parts:
                        part_wrapped = _wrap_math_tex(part)
                        html_parts.append(_format_paragraph(part_wrapped, styled=use_style))
                        explanation_paragraphs_added += 1
                    continue
                html_parts.append(_format_paragraph(plain_check, styled=use_style))
                explanation_paragraphs_added += 1

            if (
                main_option
                and explanation_paragraphs_added == 0
                and not is_tiu_numerik
                and not is_verbal_analitis
            ):
                default_explanation = (
                    f"{main_option} mencerminkan penghormatan terhadap seni dan budaya lokal. "
                    "Jelaskan bagaimana unsur-unsur budaya dalam opsi tersebut muncul pada konteks soal."
                )
                html_parts.append(_format_paragraph(default_explanation, styled=use_style))

            if is_verbal_analitis:
                table_html = str(data.get("table_html", "")).strip()
                if table_html and "<table" in table_html.lower():
                    html_parts.append(table_html)

            if incorrect_indices and include_incorrect:
                html_parts.append(
                    _format_paragraph("<strong>Jawaban yang kurang tepat:</strong>", styled=True)
                )
                for idx in incorrect_indices:
                    if idx >= len(option_map):
                        continue
                    option_text = option_map[idx]
                    if not option_text:
                        continue
                    if main_option and option_text.strip().lower() == main_option.strip().lower():
                        continue
                    reason = str(incorrect_reasons.get(str(idx + 1), "")).strip()
                    reason = _sanitize_text(reason)
                    reason = _strip_reason_prefix(reason)
                    reason = _strip_option_echo(reason, option_text)
                    reason = _enrich_reason(option_text, reason, main_option, question_summary)
                    if reason:
                        preserve_tokens = _extract_proper_tokens(option_text, main_option)
                        reason = _normalize_reason_capital(reason, preserve_tokens)
                    option_score = option_scores[idx] if idx < len(option_scores) else None
                    option_label = _label_with_score(option_text, option_score, include_scores)
                    html_parts.append(
                        _format_paragraph(f"- <strong>{option_label}:</strong> {reason}", styled=True)
                    )

            explanation_text = "\n".join(html_parts)

            df.at[row_idx, "explanation_ai"] = explanation_text
            updated_rows.append(row_idx)
        except Exception as exc:  # pragma: no cover - ditampilkan ke pengguna
            question_label = row.get("no") or row_idx
            st.warning(f"Gagal membuat pembahasan untuk nomor {question_label}: {exc}")
        finally:
            progress.progress(counter / len(target_indices))
            status_text.info(f"Memproses {counter}/{len(target_indices)}")

    progress.progress(1.0)
    status_text.success("Selesai memproses seluruh permintaan.")
    return updated_rows
