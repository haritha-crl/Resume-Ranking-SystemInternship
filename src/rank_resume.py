import os
import re
import pandas as pd
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POPPLER_PATH = os.path.normpath(os.path.join(BASE_DIR, "poppler", "Library", "bin"))

if not os.path.isdir(POPPLER_PATH):
    print(f"WARNING: Poppler directory not found at {POPPLER_PATH}")
elif not os.path.exists(os.path.join(POPPLER_PATH, "pdftoppm.exe")):
    print(f"WARNING: pdftoppm.exe not found in {POPPLER_PATH}")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

RESUMES_DIR = os.path.join(BASE_DIR, "data", "resumes")
JD_PATH = os.path.join(BASE_DIR, "data", "jd.txt")
OUTPUT_CSV = os.path.join(BASE_DIR, "output", "ranked_candidates.csv")

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"ERROR: Failed to read {os.path.basename(pdf_path)} with pdfplumber: {e}")

    if not text.strip():
        print(f"INFO: {os.path.basename(pdf_path)} scanned - OCR chestunna...")
        try:
            images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH, dpi=300)
            for image in images:
                text += pytesseract.image_to_string(image, lang='eng', config='--psm 6 --oem 3') + "\n"
        except Exception as e:
            print(f"ERROR: OCR failed for {os.path.basename(pdf_path)}: {e}")
    return text

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_assessment_scores():
    return {
        "resume1.pdf": 21.39,
        "resume2.pdf": 19.60,
        "resume3.pdf": 18.22,
        "resume4.pdf": 0.00,
        "resume5.pdf": 16.78,
        "resume6.pdf": 17.11,
        "resume7.pdf": 16.98,
        "resume8.pdf": 14.55,
        "resume9.pdf": 13.22,
        "resume10.pdf": 15.66,
        "resume11.pdf": 13.34,
        "resume12.pdf": 12.89,
        "resume13.pdf": 12.11,
        "resume14.pdf": 14.33,
        "resume15.pdf": 11.67,
        "resume16.pdf": 10.98,
        "resume17.pdf": 10.45,
        "resume18.pdf": 11.00,
        "resume19.pdf": 9.88,
        "resume20.pdf": 9.21,
        "resume21.pdf": 8.76,
        "resume22.pdf": 8.12,
        "resume23.pdf": 7.89,
        "resume24.pdf": 7.45,
        "resume25.pdf": 7.00,
    }

def calculate_similarity(resume_text, jd_text):
    processed_resume = preprocess_text(resume_text)
    processed_jd = preprocess_text(jd_text)

    if not processed_resume.strip():
        return 0.0

    documents = [processed_resume, processed_jd]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(similarity[0][0] * 100, 2)

def rank_resumes():
    if not os.path.exists(JD_PATH):
        print(f"ERROR: JD file not found at {JD_PATH}")
        return

    with open(JD_PATH, 'r', encoding='utf-8') as f:
        jd_text = f.read()

    if not jd_text.strip():
        print("ERROR: JD file is empty")
        return

    assessment_scores = get_assessment_scores()
    results = []

    if not os.path.isdir(RESUMES_DIR):
        print(f"ERROR: Resumes folder not found at {RESUMES_DIR}")
        return

    pdf_files = [f for f in os.listdir(RESUMES_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        print("ERROR: No PDF resumes found in data/resumes folder")
        return

    print(f"INFO: Found {len(pdf_files)} resumes. Processing...")

    for filename in pdf_files:
        filepath = os.path.join(RESUMES_DIR, filename)
        resume_text = extract_text_from_pdf(filepath)

        if not resume_text.strip():
            print(f"WARNING: No text extracted from {filename}. Skipping.")
            continue

        similarity_score = calculate_similarity(resume_text, jd_text)
        assessment_score = assessment_scores.get(filename, 0)

        final_score = (similarity_score * 0.6) + (assessment_score * 0.4)

        results.append({
            "Resume": filename,
            "Assessment Score": assessment_score,
            "Similarity Score (%)": similarity_score,
            "Final Score": round(final_score, 2)
        })

    if not results:
        print("\nERROR: No resumes were processed. Please check the PDF files and Poppler installation.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values(by="Final Score", ascending=False)
    df['Rank'] = range(1, len(df) + 1)

    df = df[["Rank", "Resume", "Assessment Score", "Similarity Score (%)", "Final Score"]]

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("="*80)
    print("=== FINAL SUMMARY - RESUME CANDIDATES RANKING ===")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    rank_resumes()