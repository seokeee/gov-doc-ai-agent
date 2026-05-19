"""
Module 5: Document Verifier
- Spell/terminology checks (rule-based)
- Structure checks: required fields present
- Numeric consistency: amount in text == amount computed from unit*count
- Standard administrative term enforcement

For production:
  - Replace rule-based spell check with py-hanspell (Naver) or hunspell-dict-ko
  - Add standard term dictionary from 행정안전부 공공데이터 공통표준용어
"""

import re
from typing import List, Dict, Optional


# Rule-based Korean spelling/term fixes for 공문서 style
SPELLING_RULES = [
    # (pattern, fix, description, severity)
    (r"됬", "됐", "맞춤법 오류 (됬→됐)", "error"),
    (r"되여", "되어", "맞춤법 오류 (되여→되어)", "error"),
    (r"몇일", "며칠", "맞춤법 오류 (몇일→며칠)", "error"),
    (r"데이타", "데이터", "외래어 표기 (데이타→데이터)", "warning"),
    (r"컴퓨타", "컴퓨터", "외래어 표기 (컴퓨타→컴퓨터)", "warning"),
    (r"\b가능한한\b", "가능한 한", "띄어쓰기 (가능한한→가능한 한)", "warning"),
    (r"할수있", "할 수 있", "띄어쓰기 (할수있→할 수 있)", "warning"),
    (r"될수있", "될 수 있", "띄어쓰기 (될수있→될 수 있)", "warning"),
    (r"인쇄물비", "유인물비", "행정 용어 (인쇄물비→유인물비)", "error"),
    (r"업무추진수당", "자문수당", "행정 용어 (일반적으로 자문수당 사용)", "info"),
]

# Required structural fields for 지출 공문
REQUIRED_FIELDS = {
    "지출금액": r"지\s*출\s*금\s*액",
    "지급방법": r"지\s*급\s*방\s*법",
    "예산과목": r"예\s*산\s*과\s*목",
}

# Document structure rules
STRUCTURE_RULES = [
    ("문서 끝 표시", r"끝\s*\.?\s*$", "공문서는 '끝.' 표시로 마무리해야 합니다", "info"),
]


class DocumentVerifier:
    def __init__(self):
        self.spelling_rules = SPELLING_RULES
        self.required_fields = REQUIRED_FIELDS

    def verify(
        self,
        text: str,
        expected_amount: Optional[int] = None,
        expected_participant_count: Optional[int] = None,
        expected_unit_cost: Optional[int] = None,
    ) -> Dict:
        """
        Run all verification checks.
        Returns dict with 'issues' list, 'score' (0-100), 'passed' bool.
        """
        issues = []

        issues.extend(self._check_spelling(text))
        issues.extend(self._check_required_fields(text))
        issues.extend(self._check_structure(text))
        issues.extend(self._check_numeric_consistency(
            text, expected_amount, expected_participant_count, expected_unit_cost
        ))

        # Compute score
        error_count = sum(1 for i in issues if i["severity"] == "error")
        warning_count = sum(1 for i in issues if i["severity"] == "warning")
        info_count = sum(1 for i in issues if i["severity"] == "info")

        score = max(0, 100 - error_count * 20 - warning_count * 5 - info_count * 1)

        return {
            "issues": issues,
            "summary": {
                "total": len(issues),
                "errors": error_count,
                "warnings": warning_count,
                "info": info_count,
            },
            "score": score,
            "passed": error_count == 0,
        }

    def _check_spelling(self, text: str) -> List[Dict]:
        issues = []
        for pattern, fix, desc, severity in self.spelling_rules:
            for match in re.finditer(pattern, text):
                issues.append({
                    "type": "spelling",
                    "severity": severity,
                    "found": match.group(0),
                    "suggestion": fix,
                    "description": desc,
                    "position": match.start(),
                })
        return issues

    def _check_required_fields(self, text: str) -> List[Dict]:
        issues = []
        for field_name, pattern in self.required_fields.items():
            if not re.search(pattern, text):
                issues.append({
                    "type": "structure",
                    "severity": "error",
                    "found": None,
                    "suggestion": f"'{field_name}' 항목을 추가하세요",
                    "description": f"필수 항목 누락: {field_name}",
                    "position": -1,
                })
        return issues

    def _check_structure(self, text: str) -> List[Dict]:
        issues = []
        for name, pattern, desc, severity in STRUCTURE_RULES:
            if not re.search(pattern, text.strip()):
                issues.append({
                    "type": "structure",
                    "severity": severity,
                    "found": None,
                    "suggestion": desc,
                    "description": f"구조 규칙: {name}",
                    "position": -1,
                })
        return issues

    def _check_numeric_consistency(
        self,
        text: str,
        expected_amount: Optional[int],
        expected_count: Optional[int],
        expected_unit: Optional[int],
    ) -> List[Dict]:
        """Check that 지출금액 in the draft matches user-provided amount."""
        issues = []

        # Extract numeric amount after "지출금액"
        m = re.search(r"지\s*출\s*금\s*액[^\d]*([\d,]+)\s*원", text)
        if m:
            try:
                found_amount = int(m.group(1).replace(",", ""))
                if expected_amount is not None and found_amount != expected_amount:
                    issues.append({
                        "type": "numeric",
                        "severity": "error",
                        "found": f"{found_amount:,}원",
                        "suggestion": f"{expected_amount:,}원",
                        "description": f"지출금액 불일치: 본문은 {found_amount:,}원, 입력값은 {expected_amount:,}원",
                        "position": m.start(),
                    })
            except ValueError:
                pass

        # Check 산출기초 calculation: unit * count = total
        m = re.search(r"([\d,]+)\s*원?\s*[xX×]\s*([\d]+)\s*명?\s*=\s*([\d,]+)", text)
        if m:
            try:
                unit = int(m.group(1).replace(",", ""))
                count = int(m.group(2))
                declared_total = int(m.group(3).replace(",", ""))
                computed = unit * count
                if computed != declared_total:
                    issues.append({
                        "type": "numeric",
                        "severity": "error",
                        "found": f"{unit:,} × {count} = {declared_total:,}",
                        "suggestion": f"{unit:,} × {count} = {computed:,}",
                        "description": "산출기초 계산 오류",
                        "position": m.start(),
                    })
            except ValueError:
                pass

        return issues


if __name__ == "__main__":
    verifier = DocumentVerifier()

    # Test with a bad draft containing errors
    bad_draft = """협약과제 회의비 지출

1. 지출금액 : 250,000원
2. 산출기초 : 25,000원 X 8명 = 250,000원
3. 개요
   가. 일시: 2026.1.15.
   나. 주요내용: 데이타 분석 회의 관련 논의가 될수있도록
4. 지급방법: 카드결제

"""
    result = verifier.verify(
        bad_draft,
        expected_amount=200000,  # User said 8*25k=200k but draft says 250k
        expected_participant_count=8,
        expected_unit_cost=25000,
    )
    print(f"Score: {result['score']}/100")
    print(f"Passed: {result['passed']}")
    print(f"Summary: {result['summary']}\n")
    for issue in result["issues"]:
        severity = issue["severity"].upper()
        print(f"  [{severity}] {issue['description']}")
        if issue["found"]:
            print(f"           발견: {issue['found']}")
        print(f"           제안: {issue['suggestion']}")
