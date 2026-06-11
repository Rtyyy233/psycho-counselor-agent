"""
Batch OCR extraction for Clinical Interviewing textbook.
Extracts key chapters for therapy skills integration.
Uses checkpoints to allow resuming.
"""
import fitz
import pytesseract
from PIL import Image
import io
import os
import json
import time

os.environ['TESSDATA_PREFIX'] = r'C:\Users\RTyyyy\tessdata'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

PDF_PATH = r'C:\Users\RTyyyy\Desktop\大部头书\人文\心理咨询面谈技术（第4版） ([美]萨默斯-弗拉纳根) (z-library.sk, 1lib.sk, z-lib.sk).pdf'
OUTPUT_DIR = r'D:\Agent\Counselor-Agent-main\src\_book_extract\interviewing'
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, '_checkpoint.json')

# Chapter page ranges (1-indexed, from TOC)
# Extract key chapters for therapy skills
CHAPTERS = {
    'ch02_Foundations': (40, 80),       # 咨询面谈基础
    'ch04_Listening_Skills': (95, 135),  # 注意与倾听技术
    'ch05_Questions_Action': (136, 175), # 指导性提问与行动技术
    'ch06_Therapeutic_Relationship': (176, 220), # 治疗关系
    'ch08_Mental_Status': (260, 310),   # 精神状态检查
    'ch09_Suicide_Assessment': (311, 350), # 自杀评估
    'ch10_Diagnosis_Treatment': (351, 400), # 诊断与治疗计划
    'ch11_Diverse_Populations': (430, 480), # 多元群体
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'completed_ranges': [], 'current_chapter': None, 'last_page': 0}

def save_checkpoint(state):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(state, f)

def ocr_page(doc, page_num, dpi=200):
    """OCR a single page and return text."""
    page = doc[page_num]  # 0-indexed
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes('png')))
    text = pytesseract.image_to_string(img, lang='chi_sim')
    return text

def extract_chapter(doc, ch_name, start_page, end_page, checkpoint):
    """Extract a chapter range."""
    out_file = os.path.join(OUTPUT_DIR, f'{ch_name}.txt')

    # Adjust for 0-indexed pages (PDF pages are 1-indexed in TOC)
    start_idx = start_page - 1
    end_idx = min(end_page, doc.page_count)

    print(f'Extracting {ch_name}: PDF pages {start_idx+1}-{end_idx+1}')

    all_text = []
    for pg in range(start_idx, end_idx):
        if pg % 10 == 0:
            print(f'  Page {pg+1}...')
            save_checkpoint(checkpoint)

        try:
            text = ocr_page(doc, pg)
            all_text.append(f'=== Page {pg+1} ===\n{text}\n')
        except Exception as e:
            print(f'  ERROR page {pg+1}: {e}')
            all_text.append(f'=== Page {pg+1} ===\n[ERROR: {e}]\n')

    with open(out_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_text))

    print(f'  Saved to {out_file} ({len(all_text)} pages)')
    return out_file

def main():
    print('Opening PDF...')
    doc = fitz.open(PDF_PATH)
    print(f'Total pages: {doc.page_count}')

    checkpoint = load_checkpoint()
    print(f'Checkpoint: {checkpoint}')

    for ch_name, (start, end) in CHAPTERS.items():
        if ch_name in checkpoint.get('completed_ranges', []):
            print(f'Skipping {ch_name} (already done)')
            continue

        extract_chapter(doc, ch_name, start, end, checkpoint)
        checkpoint.setdefault('completed_ranges', []).append(ch_name)
        save_checkpoint(checkpoint)

    doc.close()
    print('Done!')

if __name__ == '__main__':
    main()
