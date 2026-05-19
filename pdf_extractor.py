"""
Module 1: PDF Extractor
- Extracts text from PDFs using PyMuPDF (for digital PDFs)
- Falls back to Tesseract OCR (for scanned/image PDFs)
- Korean (kor) + English (eng) OCR support
"""

import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from pathlib import Path
from typing import Optional
import time


class PDFExtractor:
    def __init__(self, ocr_lang: str = "kor+eng", ocr_dpi: int = 200):
        self.ocr_lang = ocr_lang
        self.ocr_dpi = ocr_dpi

    def extract(self, pdf_path: str, force_ocr: bool = False) -> dict:
        """
        Extract text from a PDF. Returns dict with text, method used, pages, time.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        start = time.time()

        # Step 1: Try direct text extraction first
        if not force_ocr:
            digital_text = self._extract_digital(pdf_path)
            if digital_text and len(digital_text.strip()) > 50:
                return {
                    "text": digital_text,
                    "method": "digital",
                    "pages": digital_text.count("\n\n=== PAGE"),
                    "elapsed_sec": round(time.time() - start, 2),
                    "filename": path.name,
                }

        # Step 2: Fall back to OCR for scanned PDFs
        ocr_text, n_pages = self._extract_ocr(pdf_path)
        return {
            "text": ocr_text,
            "method": "ocr",
            "pages": n_pages,
            "elapsed_sec": round(time.time() - start, 2),
            "filename": path.name,
        }

    def _extract_digital(self, pdf_path: str) -> str:
        """Extract text layer from digital PDFs"""
        doc = fitz.open(pdf_path)
        parts = []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                parts.append(f"\n\n=== PAGE {i+1} ===\n{text}")
        doc.close()
        return "".join(parts)

    def _extract_ocr(self, pdf_path: str) -> tuple:
        """Extract text via OCR for scanned PDFs"""
        images = convert_from_path(pdf_path, dpi=self.ocr_dpi)
        parts = []
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img, lang=self.ocr_lang)
            if text.strip():
                parts.append(f"\n\n=== PAGE {i+1} ===\n{text}")
        return "".join(parts), len(images)


if __name__ == "__main__":
    extractor = PDFExtractor()
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "pdfs/1.pdf"
    result = extractor.extract(pdf)
    print(f"File: {result['filename']}")
    print(f"Method: {result['method']}")
    print(f"Pages: {result['pages']}")
    print(f"Time: {result['elapsed_sec']}s")
    print(f"Text length: {len(result['text'])}")
    print("=" * 60)
    print(result['text'][:1500])
