"""
PDF Text Extractor
==================
Extracts text from PDF files using PyMuPDF (fitz).

Currently handles digital PDFs only — for scanned/image PDFs,
OCR support will be added in a subsequent module.
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict


class PDFExtractor:
    """Extract text from PDF files."""

    def extract(self, pdf_path: str) -> Dict:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Dictionary containing:
                - text: Extracted text content
                - pages: Number of pages
                - filename: Source filename
                - method: Extraction method used
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        text = self._extract_digital(pdf_path)

        return {
            "text": text,
            "pages": text.count("\n\n=== PAGE"),
            "filename": path.name,
            "method": "digital",
        }

    def _extract_digital(self, pdf_path: str) -> str:
        """Extract text layer from a digital PDF."""
        doc = fitz.open(pdf_path)
        parts = []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                parts.append(f"\n\n=== PAGE {i+1} ===\n{text}")
        doc.close()
        return "".join(parts)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py <pdf_path>")
        sys.exit(1)

    extractor = PDFExtractor()
    result = extractor.extract(sys.argv[1])

    print(f"File: {result['filename']}")
    print(f"Method: {result['method']}")
    print(f"Pages: {result['pages']}")
    print(f"Text length: {len(result['text'])} chars")
    print("=" * 60)
    print(result['text'][:1500])