# Alat Pembahasan Soal

Aplikasi Streamlit yang membantu mengisi kolom `explanation_ai` secara otomatis menggunakan Gemini 2.0 Flash langsung dari Google Sheets.

## Menjalankan Aplikasi

1. Buat dan aktifkan lingkungan Python (opsional).
2. Instal dependensi:

   ```bash
   pip install streamlit pandas
   ```

   atau

   ```bash
   pip install -r requirements.txt
   ```

3. Masukkan API key Gemini:

   ```bash
   export GEMINI_API_KEY="key-anda"
   ```

   Atau edit nilai `GEMINI_API_KEY` di `app/config.py` secara langsung.

4. Siapkan kredensial Service Account Google (file JSON) dan pastikan spreadsheet sudah di-share ke service account tersebut sebagai editor.

5. Jalankan aplikasi:

   ```bash
   streamlit run streamlit_app.py
   ```

## Fitur

- Sambungkan aplikasi langsung ke Google Sheets via link/ID dan file service account JSON.
- Pilih worksheet yang ingin diproses dan generate pembahasan AI untuk kolom `explanation_ai`.
- Mode generate: hanya baris kosong atau seluruh baris.
- Simpan hasil kembali ke Google Sheets dengan satu tombol.

## Struktur Kode

- `streamlit_app.py` – titik masuk Streamlit, alur ambil Worksheet Google Sheet → generate → simpan kembali.
- `app/config.py` – daftar kolom wajib, pemetaan opsi jawaban, serta konfigurasi Gemini.
- `app/data_utils.py` – utilitas membaca workbook dan validasi kolom.
- `app/ai_utils.py` – penyusunan prompt HTML dan pemanggilan model Gemini.

Pastikan kolom berikut tersedia pada berkas CSV:

```
no, id, category, question_categories_id, sub_category, sub_category_id, program,
parent_id, question, answer_header_true, answer_header_false, option_a_text,
option_a_score, option_b_text, option_b_score, option_c_text, option_c_score,
option_d_text, option_d_score, option_e_text, option_e_score, option_number,
option_number_score, explanation, explanation_media, tags, answer_type,
question_keyword, question_type, type, bloom_level, in_modules, explanation_ai
```
