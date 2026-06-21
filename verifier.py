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
# NOTE: '끝.' detection is handled specially in _check_structure (see _has_closing_mark)
# because Korean official documents place '끝.' at the END OF THE BODY, after which
# 시행정보(시행 ...), 주소/연락처, 결재라인 등이 따라올 수 있다.
STRUCTURE_RULES = [
]

# Markers that legitimately appear AFTER the body's '끝.' in real 공문서.
# If '끝.' is followed only by these, the closing mark is considered valid.
POST_BODY_MARKERS = [
    "시행", "접수", "우편번호", "우 ", "전화", "전송", "이메일", "공개",
    "예상 결재라인", "결재라인", "기안", "검토", "협조", "결재",
    "www.", "http", "@", "연구실-", "재가",
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

        # Generic structure rules (currently none beyond closing mark)
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

        # Closing mark ('끝.') — special handling
        if not self._has_closing_mark(text):
            issues.append({
                "type": "structure",
                "severity": "info",
                "found": None,
                "suggestion": "공문서는 본문 마지막에 '끝.' 표시로 마무리해야 합니다",
                "description": "구조 규칙: 문서 끝 표시",
                "position": -1,
            })

        return issues

    def _has_closing_mark(self, text: str) -> bool:
        """
        Check whether the document has a valid '끝.' closing mark.

        Korean official documents place '끝.' at the end of the BODY. After it,
        post-body elements (시행정보, 주소/연락처, 결재라인 등) may follow.
        So '끝.' is valid if:
          (a) it appears at the very end of the text, OR
          (b) everything after the LAST '끝.' consists only of post-body markers.
        """
        # Find all '끝' followed by optional period
        matches = list(re.finditer(r"끝\s*\.?", text))
        if not matches:
            return False

        # Use the last occurrence as the candidate closing mark
        last = matches[-1]
        tail = text[last.end():].strip()

        # (a) Nothing meaningful after '끝.' -> valid
        if not tail:
            return True

        # (b) Everything after '끝.' is post-body content -> still valid
        for line in tail.splitlines():
            line = line.strip()
            if not line:
                continue
            if not any(marker in line for marker in POST_BODY_MARKERS):
                # Found a content line that isn't a recognized post-body marker.
                # Be lenient: short lines (likely names/dates in 결재라인) pass too.
                if len(line) > 25:
                    return False
        return True

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
