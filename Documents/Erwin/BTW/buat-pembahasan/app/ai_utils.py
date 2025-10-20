"""Utilitas untuk membangun prompt dan memanggil model AI."""

import html
import json
import re
from typing import Dict, List, Optional, Sequence

import pandas as pd
import streamlit as st

from .config import GEMINI_API_KEY, OPTION_COLUMNS


def _sanitize_text(value: str) -> str:
    """Remove HTML tags, entities, and collapse whitespace without altering wording."""

    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


_REASON_PREFIXES = [
    re.compile(r"^\s*jawaban\s+yang\s+kurang\s+tepat\s*[:\-]*\s*", re.IGNORECASE),
    re.compile(r"^\s*jawaban\s+kurang\s+tepat\s*[:\-]*\s*", re.IGNORECASE),
    re.compile(r"^\s*jawaban\s+salah\s*[:\-]*\s*", re.IGNORECASE),
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

    lowered_reason = cleaned_reason.lower()
    lowered_option = option_clean.lower()

    if lowered_reason.startswith(lowered_option):
        trimmed = cleaned_reason[len(option_clean) :].lstrip(" :.-").strip()
        if trimmed:
            cleaned_reason = trimmed
        else:
            cleaned_reason = ""

    return cleaned_reason


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
        if correct_text:
            supplements.append(
                f"Opsi ini tidak menegaskan aspek utama '{correct_text}' sehingga kurang relevan dengan tujuan soal."
            )
        if not supplements:
            supplements.append(
                "Penalaran perlu menunjukkan mengapa opsi ini tidak menutup kebutuhan soal sehingga tidak layak dipilih."
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
    for idx, (label, (text_col, _)) in enumerate(OPTION_COLUMNS.items()):
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

    format_template = (
        "<p style=\"text-align:justify\"><strong>Jawaban yang tepat: ...</strong></p>\n"
        "<p style=\"text-align:justify\">Paragraf penjelasan lanjutan...</p>\n"
        "<p style=\"text-align:justify\"><strong>Jawaban yang kurang tepat:</strong></p>\n"
        "<p style=\"text-align:justify\"><strong>Opsi salah 1:</strong> alasan singkat...</p>\n"
        "<p style=\"text-align:justify\"><strong>Opsi salah 2:</strong> alasan singkat...</p>"
    )

    instructions = (
        "Tulis pembahasan dalam bahasa Indonesia menggunakan HTML tanpa menambahkan <!DOCTYPE>, <html>, <head>, atau <body>. "
        "Pembahasan harus bernas dan mudah dipahami siswa. Jelaskan alasan jawaban benar secara menyeluruh (minimal 3 kalimat) dan uraikan mengapa tiap opsi salah tidak memenuhi kriteria. "
        "Output harus berupa rangkaian tag <p> seperti contoh berikut dan tidak boleh memiliki teks di luar tag tersebut.\n\n"
        f"Format wajib diikuti:\n{format_template}\n\n"
        "Aturan tambahan:\n"
        "- Paragraf pertama harus menyatakan jawaban benar dengan awalan 'Jawaban yang tepat:' diikuti penjelasan singkat. Sertakan teks opsi benar secara utuh sebelum penjelasan.\n"
        "- Paragraf kedua (dan tambahan bila perlu) menjelaskan alasan jawaban benar secara detail (minimal 2 kalimat).\n"
        "- Gunakan paragraf ketiga dengan teks 'Jawaban yang kurang tepat:' (bold).\n"
        "- Tambahkan paragraf terpisah untuk setiap opsi salah dengan format '<strong>...:</strong> penjelasan...'. Teks sebelum titik dua HARUS persis menyalin isi opsi tanpa perubahan atau sinonim. Setiap alasan minimal dua kalimat yang jelas.\n"
        "- Jangan menuliskan label huruf seperti A/B/C di dalam isi jawaban. Fokus pada isi opsi saja.\n"
        "- Jangan menulis ulang opsi yang benar di bagian opsi salah.\n"
        "- Nilai `correct_summary` hanya berisi penjelasan singkat (tanpa kembali menuliskan frasa 'Jawaban yang tepat'). Jika tidak ada penjelasan tambahan, kosongkan string tersebut.\n"
        "- Nilai dalam `incorrect_reasons` adalah penjelasan mendalam (minimal dua kalimat) untuk tiap opsi salah tanpa menyalin ulang teks opsi.\n"
        "- Semua nilai string dalam JSON harus berupa teks polos tanpa tag HTML.\n"
        "- Gunakan style 'text-align:justify' pada setiap tag <p> dan <strong> untuk penekanan."
    )

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

    context = (
        instructions
        + "\n\n"
        + metadata
        + f"\n\nSoal:\n{question_text}\n\nDaftar opsi (gunakan apa adanya untuk bagian opsi salah):\n{option_instructions}\n"
        + f"Opsi benar (copy teksnya secara utuh ketika menyusun ringkasan):\n{correct_option_display}\n"
        + f"Opsi salah yang wajib dijelaskan: {', '.join(str(idx + 1) for idx in incorrect_indices)}\n"
        + "Kembalikan respons dalam format JSON PERSIS seperti berikut tanpa teks tambahan atau blok kode:\n"
        "{\n"
        "  \"correct_summary\": string,\n"
        "  \"detail_paragraphs\": [string, ...],\n"
        "  \"incorrect_reasons\": {\n"
        "       \"1\": string alasan (untuk opsi indeks 1 jika salah),\n"
        "       ...\n"
        "   }\n"
        "}\n"
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

    effective_key = api_key or GEMINI_API_KEY
    if not effective_key or "ISI_API_KEY" in effective_key:
        st.error(
            "API key Gemini belum dikonfigurasi. Set variabel lingkungan `GEMINI_API_KEY` atau perbarui `GEMINI_API_KEY` di `app/config.py`."
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
        prompt_data = build_prompt(row)
        prompt = prompt_data["prompt"]
        option_map: List[str] = prompt_data["option_map"]
        correct_indices: List[int] = prompt_data["correct_indices"]
        incorrect_indices: List[int] = prompt_data["incorrect_indices"]
        question_summary: str = prompt_data.get("question_summary", "")

        if not correct_indices:
            question_label = row.get("id") or row.get("no") or row_idx
            st.warning(
                f"Lewati baris {question_label}: tidak menemukan jawaban benar pada data sumber."
            )
            continue
        try:
            response = model.generate_content(prompt)
            raw_text = (response.text or "").strip()
            raw_text = raw_text.replace("```,", "```")
            if raw_text.startswith("```"):
                raw_text = raw_text.strip()
                if raw_text.startswith("```json"):
                    raw_text = raw_text[len("```json"):]
                raw_text = raw_text.strip("`").strip()

            parsed = None
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError:
                start = raw_text.find("{")
                end = raw_text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    candidate = raw_text[start : end + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        parsed = None
            if parsed is None:
                st.warning(
                    f"Format respons tidak valid untuk baris {row_idx}. Mengabaikan pembaruan."
                )
                continue

            data = parsed

            correct_summary = str(data.get("correct_summary", "")).strip()
            detail_paragraphs_raw = data.get("detail_paragraphs") or []
            if isinstance(detail_paragraphs_raw, str):
                detail_paragraphs = [detail_paragraphs_raw]
            else:
                detail_paragraphs = list(detail_paragraphs_raw)
            incorrect_reasons = data.get("incorrect_reasons") or {}

            html_parts: List[str] = []

            if not correct_summary and correct_indices:
                first_idx = correct_indices[0]
                if first_idx < len(option_map):
                    correct_summary = option_map[first_idx]

            main_option = ""
            if correct_indices:
                idx0 = correct_indices[0]
                if idx0 < len(option_map):
                    main_option = option_map[idx0].strip()

            primary_text = ""
            explanation = ""

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
                    explanation = _strip_option_echo(correct_text, main_option)
                else:
                    primary_text = correct_text
                    explanation = ""

                primary_text = primary_text.strip()
                if primary_text:
                    html_parts.append(
                        f"<p style=\"text-align:justify\"><strong>Jawaban yang tepat: {primary_text}</strong></p>"
                    )

                explanation = explanation.strip()
                if explanation and explanation.lower() == primary_text.lower():
                    explanation = ""

                if explanation:
                    detail_paragraphs.insert(0, explanation)

            explanation_paragraphs_added = 0
            for paragraph in detail_paragraphs:
                paragraph = _sanitize_text(paragraph)
                if not paragraph:
                    continue
                lowered = paragraph.lower()
                if (
                    lowered.startswith("jawaban yang tepat")
                    or lowered.startswith("jawaban yang kurang tepat")
                    or lowered.startswith("- opsi")
                    or lowered.startswith("opsi")
                    or lowered.startswith("- pilihan")
                    or lowered.startswith("- ")
                ):
                    continue
                html_parts.append(
                    f"<p style=\"text-align:justify\">{paragraph}</p>"
                )
                explanation_paragraphs_added += 1

            if main_option and explanation_paragraphs_added == 0:
                default_explanation = (
                    f"{main_option} mencerminkan penghormatan terhadap seni dan budaya lokal. "
                    "Jelaskan bagaimana unsur-unsur budaya dalam opsi tersebut muncul pada konteks soal."
                )
                html_parts.append(
                    f"<p style=\"text-align:justify\">{default_explanation}</p>"
                )

            if incorrect_indices:
                html_parts.append(
                    "<p style=\"text-align:justify\"><strong>Jawaban yang kurang tepat:</strong></p>"
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
                    html_parts.append(
                        f"<p style=\"text-align:justify\"><strong>{option_text}:</strong> {reason}</p>"
                    )

            explanation_text = "\n".join(html_parts)

            df.at[row_idx, "explanation_ai"] = explanation_text
            updated_rows.append(row_idx)
        except Exception as exc:  # pragma: no cover - ditampilkan ke pengguna
            st.warning(f"Gagal membuat pembahasan untuk baris {row_idx}: {exc}")
        finally:
            progress.progress(counter / len(target_indices))
            status_text.info(f"Memproses {counter}/{len(target_indices)}")

    progress.progress(1.0)
    status_text.success("Selesai memproses seluruh permintaan.")
    return updated_rows
