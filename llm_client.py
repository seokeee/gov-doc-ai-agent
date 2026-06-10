"""
Module 4: LLM Client (v2)
=========================
Google Gemini API wrapper for RAG-based draft generation.

v2 upgrades:
- New `google-genai` SDK (replaces deprecated `google-generativeai`)
- Model fallback chain: tries multiple models when quota/availability fails
- Retry with exponential backoff on transient errors (429, 5xx)
- Truncation detection via finish_reason -> `truncated` flag in result
- Input validation: rejects empty/insufficient user input early
- Backward-compatible interface (agent.py / api_server.py unchanged)

Env vars (.env supported):
- GOOGLE_API_KEY   : required for LLM generation
- GEMINI_MODEL     : optional, overrides the default model chain head
"""

import os
import time
from typing import List, Dict, Optional

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# New SDK: google-genai  (pip install google-genai)
try:
    from google import genai
    from google.genai import types as genai_types
    from google.genai import errors as genai_errors
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


def number_to_korean(num: int) -> str:
    """Convert integer to Korean number text (e.g., 225000 -> '금이십이만오천원').

    Korean gov-doc convention:
    - '일십' is abbreviated to '십'
    - '일백', '일천' are kept (e.g. 1,040,000 -> 금일백사만원)
    """
    if num == 0:
        return "금영원"
    units = ["", "만", "억", "조"]
    digits = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
    sub_units = ["", "십", "백", "천"]

    result = ""
    unit_idx = 0
    while num > 0:
        chunk = num % 10000
        if chunk > 0:
            chunk_str = ""
            c = chunk
            for i in range(4):
                if c <= 0:
                    break
                d = c % 10
                if d > 0:
                    if d == 1 and i == 1:  # 십 position only
                        prefix = ""
                    else:
                        prefix = digits[d]
                    chunk_str = prefix + sub_units[i] + chunk_str
                c //= 10
            result = chunk_str + units[unit_idx] + result
        num //= 10000
        unit_idx += 1
    return "금" + result + "원"


class LLMClient:
    """Gemini client with model fallback chain + retry + truncation detection."""

    # Tried in order until one succeeds. Head can be overridden via GEMINI_MODEL.
    MODEL_CHAIN = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]
    MAX_OUTPUT_TOKENS = 4500
    TEMPERATURE = 0.4
    MAX_RETRIES = 2          # per model, on transient errors
    RETRY_BASE_DELAY = 2.0   # seconds; doubles each retry

    # Minimum fields required to generate a meaningful draft
    REQUIRED_FIELDS = ["purpose", "date"]

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        # Build the model chain: explicit arg > env var > default chain
        env_model = os.getenv("GEMINI_MODEL")
        head = model or env_model
        if head:
            # Put requested model first, keep the rest as fallback
            chain = [head] + [m for m in self.MODEL_CHAIN if m != head]
        else:
            chain = list(self.MODEL_CHAIN)
        self.model_chain = chain

        # The model that most recently succeeded (starts as chain head)
        self.model_name = self.model_chain[0]
        self.model = self.model_name  # alias for backward compatibility

        self.client = None
        if _GENAI_AVAILABLE and self.api_key:
            self.client = genai.Client(api_key=self.api_key)

    def is_available(self) -> bool:
        return self.client is not None

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────
    def generate_draft(
        self,
        doc_type: str,
        user_input: Dict,
        retrieved_docs: List[Dict],
    ) -> Dict:
        """
        Generate a 공문 draft using RAG context.

        Returns dict:
            draft          : generated text
            model          : model actually used ('template-fallback' if LLM unused)
            retrieved_ids  : filenames of RAG context docs
            fallback       : True if template fallback was used
            truncated      : True if output hit token limit (LLM only)
            total_amount   : computed amount (int)
            korean_amount  : Korean text amount
            error          : error message if LLM failed (optional)
            warnings       : list of validation warnings (optional)
        """
        total_amount = self._compute_total(user_input)
        korean_amount = number_to_korean(total_amount) if total_amount else ""

        # ── Input validation ──
        validation_warnings = self._validate_input(doc_type, user_input)

        # ── No API -> template fallback ──
        if not self.is_available():
            draft = self._template_fallback(doc_type, user_input, total_amount, korean_amount)
            return {
                "draft": draft,
                "model": "template-fallback",
                "retrieved_ids": [d["doc"].get("filename", "") for d in retrieved_docs],
                "fallback": True,
                "truncated": False,
                "total_amount": total_amount,
                "korean_amount": korean_amount,
                "warnings": validation_warnings,
            }

        prompt = self._build_prompt(doc_type, user_input, retrieved_docs,
                                    total_amount, korean_amount)

        # ── Model fallback chain with retry ──
        last_error = None
        for model_name in self.model_chain:
            result = self._try_model(model_name, prompt)
            if result is not None:
                draft_text, truncated = result
                self.model_name = model_name
                self.model = model_name
                return {
                    "draft": draft_text,
                    "model": model_name,
                    "retrieved_ids": [d["doc"].get("filename", "") for d in retrieved_docs],
                    "fallback": False,
                    "truncated": truncated,
                    "total_amount": total_amount,
                    "korean_amount": korean_amount,
                    "warnings": validation_warnings,
                }
            last_error = self._last_error

        # ── All models failed -> template fallback ──
        print(f"[LLM] All models failed ({last_error}). Falling back to template.")
        draft = self._template_fallback(doc_type, user_input, total_amount, korean_amount)
        return {
            "draft": draft,
            "model": "template-fallback (API error)",
            "retrieved_ids": [d["doc"].get("filename", "") for d in retrieved_docs],
            "fallback": True,
            "truncated": False,
            "total_amount": total_amount,
            "korean_amount": korean_amount,
            "error": str(last_error),
            "warnings": validation_warnings,
        }

    # ──────────────────────────────────────────────────────────────
    # Internal: model call with retry
    # ──────────────────────────────────────────────────────────────
    def _try_model(self, model_name: str, prompt: str):
        """Try one model with retries. Returns (text, truncated) or None."""
        self._last_error = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        max_output_tokens=self.MAX_OUTPUT_TOKENS,
                        temperature=self.TEMPERATURE,
                    ),
                )

                text = response.text or ""
                if not text.strip():
                    self._last_error = f"{model_name}: empty response"
                    return None

                # Truncation detection
                truncated = False
                try:
                    fr = response.candidates[0].finish_reason
                    truncated = (str(fr).upper().find("MAX_TOKENS") >= 0)
                except (IndexError, AttributeError):
                    pass

                if truncated:
                    print(f"[LLM] WARNING: {model_name} output hit token limit "
                          f"({self.MAX_OUTPUT_TOKENS}). Draft may be incomplete.")
                return (text, truncated)

            except genai_errors.APIError as e:
                code = getattr(e, "code", None)
                self._last_error = f"{model_name}: {code} {e}"
                # 429 = quota; retry with backoff, then move to next model
                if code == 429 and attempt < self.MAX_RETRIES:
                    delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"[LLM] {model_name} rate-limited. Retry in {delay:.0f}s "
                          f"({attempt + 1}/{self.MAX_RETRIES})")
                    time.sleep(delay)
                    continue
                # 5xx transient errors: retry once
                if code is not None and 500 <= int(code) < 600 and attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_BASE_DELAY)
                    continue
                # 404 (model not found), persistent 429 etc -> next model in chain
                print(f"[LLM] {model_name} failed: {self._last_error}. Trying next model.")
                return None
            except Exception as e:
                self._last_error = f"{model_name}: {e}"
                print(f"[LLM] {model_name} unexpected error: {e}. Trying next model.")
                return None
        return None

    # ──────────────────────────────────────────────────────────────
    # Internal: prompt construction
    # ──────────────────────────────────────────────────────────────
    def _build_prompt(self, doc_type, user_input, retrieved_docs,
                      total_amount, korean_amount) -> str:
        context_blocks = []
        for i, r in enumerate(retrieved_docs, 1):
            d = r["doc"]
            context_blocks.append(
                f"[참고 문서 {i}] (유사도: {r.get('hybrid_score', r.get('score', 0)):.3f})\n"
                f"파일: {d.get('filename', '')}\n"
                f"유형: {d.get('doc_type', '')}\n"
                f"제목: {d.get('title', '')}\n"
                f"본문:\n{d.get('raw_text', '')[:1500]}"
            )
        context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(참고 문서 없음)"

        input_summary = self._format_input(user_input, total_amount, korean_amount)

        return f"""당신은 울산연구원(URI) 공문서 작성 보조 AI입니다.
아래 [참고 문서]의 서식과 표현을 최대한 참고하여 [입력 정보]에 맞는 공문 초안을 작성하세요.

[참고 문서]
{context}

[입력 정보]
문서 유형: {doc_type}
{input_summary}

[작성 규칙]
1. 참고 문서의 서식과 문장 구조를 따를 것 (동일한 항목 번호, 표현 방식)
2. 수치와 날짜는 입력 정보를 정확히 반영할 것
3. 예산과목은 참고 문서에서 가장 적절한 것을 선택할 것
4. 문서 끝에 '끝.' 표시를 붙일 것
5. 결재라인은 참고 문서의 패턴을 따를 것 (본문 작성 후 별도 섹션으로 [예상 결재라인] 표시)
6. 초안 본문만 출력하고 설명이나 메타 코멘트는 넣지 말 것
7. 한국어 공문서 표준 표현 및 맞춤법을 엄격히 준수할 것

초안을 작성하세요:"""

    # ──────────────────────────────────────────────────────────────
    # Internal: validation + helpers
    # ──────────────────────────────────────────────────────────────
    def _validate_input(self, doc_type: str, user_input: Dict) -> List[str]:
        """Return list of human-readable warnings for missing/weak input."""
        warnings = []
        if not doc_type or not str(doc_type).strip():
            warnings.append("문서 유형이 비어 있습니다.")
        missing = [f for f in self.REQUIRED_FIELDS
                   if not str(user_input.get(f, "") or "").strip()]
        if missing:
            label = {"purpose": "주요 내용", "date": "일시"}
            names = ", ".join(label.get(m, m) for m in missing)
            warnings.append(f"필수 입력이 비어 있습니다: {names}. "
                            f"입력이 부족하면 초안 품질이 낮아질 수 있습니다.")
        return warnings

    def _compute_total(self, user_input: Dict) -> int:
        if "total_amount" in user_input and user_input["total_amount"]:
            try:
                return int(user_input["total_amount"])
            except (ValueError, TypeError):
                pass
        unit = user_input.get("unit_cost", 0) or 0
        count = user_input.get("participant_count", 0) or 0
        try:
            return int(unit) * int(count)
        except (ValueError, TypeError):
            return 0

    def _format_input(self, u: Dict, total: int, korean: str) -> str:
        lines = []
        if u.get("purpose"):           lines.append(f"주요 내용: {u['purpose']}")
        if u.get("date"):              lines.append(f"일시: {u['date']}")
        if u.get("location"):          lines.append(f"장소: {u['location']}")
        if u.get("participants"):      lines.append(f"참석자: {u['participants']}")
        if u.get("participant_count"): lines.append(f"참석 인원: 총 {u['participant_count']}명")
        if u.get("unit_cost"):         lines.append(f"1인당 단가: {int(u['unit_cost']):,}원")
        if total:                      lines.append(f"총 지출금액: {total:,}원 ({korean})")
        if u.get("payment_method"):    lines.append(f"지급방법: {u['payment_method']}")
        if u.get("project"):           lines.append(f"프로젝트: {u['project']}")
        return "\n".join(lines)

    def _template_fallback(self, doc_type: str, u: Dict, total: int, korean: str) -> str:
        """Pattern-based generation when API is unavailable."""
        project = u.get("project", "2025년 울산 빅데이터센터 운영")
        date = u.get("date", "")
        location = u.get("location", "")
        participants = u.get("participants", "")
        count = u.get("participant_count", "")
        unit_cost = int(u.get("unit_cost") or 0)
        purpose = u.get("purpose", "")
        payment = u.get("payment_method", "카드결제")
        payee = u.get("payee", "회의장소 인근 음식점" if payment == "카드결제" else "지급처")

        calc = f"{unit_cost:,}원 X {count}명 = {total:,}원" if (unit_cost and count) else f"{total:,}원"

        return f"""협약과제 「{project}」 관련하여 아래와 같이 개최하고 비용을 지출하고자 합니다.

1. 지출금액 : {total:,}원({korean})
2. 산출기초 : {calc}
3. 개요
   가. 일  시 : {date}
   나. 장  소 : {location}
   다. 참 석 자 : {participants}, 총 {count}명
   라. 주요내용 : {purpose}
4. 지급방법 : {payment}
5. 지 급 처 : {payee}
6. 예산과목 : 사업비용, 영업비용, 연구사업, 대행사업비, {project}사업(회의비)

끝.

[예상 결재라인]
연구원 → 연구위원 → 경제산업연구실장

※ 이 초안은 API 연결 없이 템플릿 기반으로 생성되었습니다. GOOGLE_API_KEY를 설정하면 Gemini 기반 생성이 활성화됩니다.
"""


if __name__ == "__main__":
    client = LLMClient()
    print(f"API available : {client.is_available()}")
    print(f"Model chain   : {' -> '.join(client.model_chain)}")
    print(f"Korean number : 225000 -> {number_to_korean(225000)}")
    print(f"                1040000 -> {number_to_korean(1040000)}")
    print(f"                150000 -> {number_to_korean(150000)}")

    # Validation test: empty input should produce warnings
    empty_result = client.generate_draft("회의비 지출(식대)", {}, [])
    print(f"\n[Validation test] warnings: {empty_result.get('warnings')}")

    # Full test
    result = client.generate_draft(
        doc_type="회의비 지출(식대)",
        user_input={
            "purpose": "AI 에이전트 개발 논의",
            "date": "2026.1.15.(수) 12:00",
            "location": "연구원 3층 회의실",
            "participants": "김상락, 이상일, 차민규",
            "participant_count": 8,
            "unit_cost": 25000,
            "payment_method": "카드결제",
            "project": "2025년 울산 빅데이터센터 운영",
        },
        retrieved_docs=[],
    )
    print("\n" + "=" * 60)
    print(f"Mode      : {result['model']}")
    print(f"Truncated : {result.get('truncated')}")
    print(result["draft"][:800])
