"""
Module 2: Document Metadata Parser
- Parses OCR'd Korean government document text
- Extracts structured metadata: title, amount, date, location, participants, budget, etc.
- Uses regex patterns tuned for Korean 공문서 format
"""

import re
from typing import Optional


class DocumentParser:
    """Parse OCR'd Korean government document text into structured metadata."""

    # Patterns for common 공문서 fields
    PATTERNS = {
        "title": [r"제\s*목[:\s]*(.+?)(?:\n|$)"],
        "amount_text": [
            r"지출금액\s*[:：]\s*([^\n]+)",
            r"지\s*출\s*금\s*액\s*[:：]\s*([^\n]+)",
        ],
        "amount_num": [r"금?\s*([\d,]{4,})\s*원"],
        "date": [
            r"일\s*시\s*[:：]\s*([\d]{4}[.\s]+[\d]{1,2}[.\s]+[\d]{1,2}[^\n]*)",
            r"(\d{4}[.-]\d{1,2}[.-]\d{1,2})",
        ],
        "location": [r"장\s*소\s*[:：]\s*([^\n]+)"],
        "payment_method": [r"지급방법\s*[:：]\s*([^\n]+)", r"지\s*급\s*방\s*법\s*[:：]\s*([^\n]+)"],
        "payee": [r"지\s*급\s*처\s*[:：]\s*([^\n]+)", r"거\s*래\s*처\s*[:：]\s*([^\n]+)"],
        "purpose": [r"주요내용\s*[:：]\s*([^\n]+)", r"주\s*요\s*내\s*용\s*[:：]\s*([^\n]+)"],
        "participants": [r"참\s*석\s*자\s*[:：]\s*([^\n]+(?:\n\s+[^\n가-힣:][^\n]+)*)"],
        "budget_category": [r"예산과목\s*[:：]\s*([^\n]+)"],
        "doc_number": [r"(경제산업연구실-\d+)", r"(\w+연구실-\d+)"],
        "project_name": [r"「([^」]+)」", r"『([^』]+)』"],
    }

    DOC_TYPE_KEYWORDS = {
        "자문회의 비용 지출": ["자문회의", "자문비", "자문수당"],
        "회의비 지출(식대)": ["식대", "회의 후 식대", "회의비 지출"],
        "세미나 비용 지출": ["세미나", "발표비"],
        "인쇄비용 지출": ["인쇄", "보고서 인쇄", "유인물"],
        "검증수수료 지출": ["검증수수료", "회계법인"],
        "출장비 지출": ["출장비", "출장"],
    }

    def parse(self, text: str, filename: str = "") -> dict:
        """Parse full document text into structured metadata."""
        result = {
            "filename": filename,
            "raw_text": text,
        }

        # Extract fields via regex patterns
        for field, patterns in self.PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    result[field] = match.group(1).strip()
                    break
            if field not in result:
                result[field] = ""

        # Parse numeric amount
        if result.get("amount_text"):
            nums = re.findall(r"[\d,]+", result["amount_text"])
            if nums:
                try:
                    result["amount_num"] = int(nums[0].replace(",", ""))
                except ValueError:
                    result["amount_num"] = 0
        else:
            result["amount_num"] = 0

        # Classify document type
        result["doc_type"] = self._classify(text)

        # Build summary chunk for embedding
        result["summary"] = self._build_summary(result)

        return result

    def _classify(self, text: str) -> str:
        """Classify doc type by keyword matching."""
        scores = {}
        for doc_type, keywords in self.DOC_TYPE_KEYWORDS.items():
            scores[doc_type] = sum(1 for kw in keywords if kw in text)
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "기타 비용 지출"

    def _build_summary(self, parsed: dict) -> str:
        """Build a concise summary for embedding."""
        parts = [
            f"문서유형: {parsed.get('doc_type', '')}",
            f"제목: {parsed.get('title', '')}",
            f"목적: {parsed.get('purpose', '')}",
            f"일시: {parsed.get('date', '')}",
            f"장소: {parsed.get('location', '')}",
            f"금액: {parsed.get('amount_text', '')}",
            f"지급방법: {parsed.get('payment_method', '')}",
            f"프로젝트: {parsed.get('project_name', '')}",
        ]
        return "\n".join(p for p in parts if p.split(": ", 1)[-1])


if __name__ == "__main__":
    import sys, json
    from pdf_extractor import PDFExtractor

    extractor = PDFExtractor()
    parser = DocumentParser()

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "pdfs/1.pdf"
    extracted = extractor.extract(pdf_path)
    parsed = parser.parse(extracted["text"], filename=extracted["filename"])

    print(json.dumps(
        {k: v for k, v in parsed.items() if k != "raw_text"},
        ensure_ascii=False, indent=2
    ))
