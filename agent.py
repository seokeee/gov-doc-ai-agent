"""
Module 6: Agent Orchestrator
- End-to-end pipeline: ingest PDFs → parse → embed → search → generate → verify
- Exposes a clean API for UI/server integration
- Reports step-by-step progress (for pipeline visualization)
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Callable

from pdf_extractor import PDFExtractor
from doc_parser import DocumentParser
from vector_search import VectorSearchEngine
from llm_client import LLMClient
from verifier import DocumentVerifier


class GovDocAgent:
    """
    Main agent orchestrating the full 공문서 작성 support pipeline.

    Two workflows:
      1. ingest(pdf_dir)   — process PDFs into searchable index (one-time setup)
      2. generate(input)   — RAG search + draft + verify (per request)

    Reference-document workflow (v2):
      - get_doc_fields(filename) — analyze a selected document and return
        the input fields needed for that doc type + reference values
      - generate(reference_filename=...) — pin the selected doc as the
        primary RAG context document
    """

    # ─── Field schemas per doc type ───────────────────────────────────
    # Defines which input fields the UI should ask for, per document type.
    # key   : UserInput field name (matches api_server.py schema)
    # label : Korean label shown in UI
    # required : minimal fields for a meaningful draft
    FIELD_SCHEMAS = {
        "회의비 지출(식대)": [
            {"key": "purpose",           "label": "주요 내용 / 목적", "required": True},
            {"key": "date",              "label": "일시",             "required": True},
            {"key": "location",          "label": "장소",             "required": False},
            {"key": "participants",      "label": "참석자",           "required": False},
            {"key": "participant_count", "label": "참석 인원 (명)",   "required": True},
            {"key": "unit_cost",         "label": "1인당 단가 (원)",  "required": True},
            {"key": "payment_method",    "label": "지급방법",         "required": False},
        ],
        "자문회의 비용 지출": [
            {"key": "purpose",           "label": "자문 주제 / 내용", "required": True},
            {"key": "date",              "label": "일시",             "required": True},
            {"key": "location",          "label": "장소",             "required": False},
            {"key": "participants",      "label": "자문위원 / 참석자","required": True},
            {"key": "participant_count", "label": "참석 인원 (명)",   "required": False},
            {"key": "unit_cost",         "label": "1인당 자문료 (원)","required": False},
            {"key": "total_amount",      "label": "총 지출금액 (원)", "required": False},
            {"key": "payment_method",    "label": "지급방법",         "required": False},
        ],
        "세미나 비용 지출": [
            {"key": "purpose",           "label": "세미나 주제",      "required": True},
            {"key": "date",              "label": "일시",             "required": True},
            {"key": "location",          "label": "장소",             "required": False},
            {"key": "participants",      "label": "발표자 / 참석자",  "required": False},
            {"key": "total_amount",      "label": "총 지출금액 (원)", "required": True},
            {"key": "payment_method",    "label": "지급방법",         "required": False},
        ],
        "인쇄비용 지출": [
            {"key": "purpose",           "label": "인쇄 내역 (보고서명 등)", "required": True},
            {"key": "total_amount",      "label": "총 지출금액 (원)", "required": True},
            {"key": "payee",             "label": "지급처 (인쇄업체)","required": True},
            {"key": "payment_method",    "label": "지급방법",         "required": False},
        ],
        "검증수수료 지출": [
            {"key": "purpose",           "label": "검증 대상 (정산보고서 등)", "required": True},
            {"key": "total_amount",      "label": "총 지출금액 (원)", "required": True},
            {"key": "payee",             "label": "지급처 (회계법인)","required": True},
            {"key": "payment_method",    "label": "지급방법",         "required": False},
        ],
        "출장비 지출": [
            {"key": "purpose",           "label": "출장 목적",        "required": True},
            {"key": "date",              "label": "출장 기간",        "required": True},
            {"key": "location",          "label": "출장지",           "required": True},
            {"key": "participants",      "label": "출장자",           "required": True},
            {"key": "total_amount",      "label": "총 지출금액 (원)", "required": False},
            {"key": "payment_method",    "label": "지급방법",         "required": False},
        ],
        "기타 비용 지출": [
            {"key": "purpose",           "label": "지출 내역 / 목적", "required": True},
            {"key": "date",              "label": "일시",             "required": False},
            {"key": "total_amount",      "label": "총 지출금액 (원)", "required": True},
            {"key": "payee",             "label": "지급처",           "required": False},
            {"key": "payment_method",    "label": "지급방법",         "required": False},
        ],
    }

    def __init__(
        self,
        pdf_dir: str = "pdfs",
        index_dir: str = "vector_store",
        data_dir: str = "data",
        vector_backend: str = "auto",
        llm_model: Optional[str] = None,
    ):
        self.pdf_dir = Path(pdf_dir)
        self.index_dir = Path(index_dir)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Initialize modules
        self.extractor = PDFExtractor()
        self.parser = DocumentParser()
        self.vector = VectorSearchEngine(backend=vector_backend, index_dir=index_dir)
        self.llm = LLMClient(model=llm_model)
        self.verifier = DocumentVerifier()

        self.parsed_docs: List[Dict] = []

    # ─── Workflow 1: Ingestion ────────────────────────────────────────
    def ingest(
        self,
        pdf_dir: Optional[str] = None,
        save: bool = True,
        on_progress: Optional[Callable[[Dict], None]] = None,
    ) -> Dict:
        """
        Process all PDFs in a directory → parse → build vector index.
        Call on_progress(step_dict) after each step if provided.
        """
        pdf_dir = Path(pdf_dir) if pdf_dir else self.pdf_dir
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        report = {"pdfs_processed": 0, "docs": [], "elapsed_sec": 0, "errors": []}

        t0 = time.time()
        self.parsed_docs = []

        for i, pdf_path in enumerate(pdf_files):
            step = {"stage": "ingest", "step": i + 1, "total": len(pdf_files),
                    "filename": pdf_path.name, "status": "processing"}
            if on_progress: on_progress(step)

            try:
                # 1. Extract text
                extracted = self.extractor.extract(str(pdf_path))

                # 2. Parse structured metadata
                parsed = self.parser.parse(extracted["text"], filename=pdf_path.name)
                parsed["extraction_method"] = extracted["method"]
                parsed["extraction_time"] = extracted["elapsed_sec"]

                self.parsed_docs.append(parsed)
                report["docs"].append({
                    "filename": pdf_path.name,
                    "title": parsed.get("title", ""),
                    "doc_type": parsed.get("doc_type", ""),
                    "amount": parsed.get("amount_text", ""),
                    "method": extracted["method"],
                })
                step["status"] = "done"
                step["doc_type"] = parsed.get("doc_type", "")
                if on_progress: on_progress(step)

            except Exception as e:
                report["errors"].append({"filename": pdf_path.name, "error": str(e)})
                step["status"] = "error"
                step["error"] = str(e)
                if on_progress: on_progress(step)

        # 3. Build vector index
        step = {"stage": "ingest", "step": "index", "status": "processing",
                "message": f"벡터 인덱스 구축 중 ({len(self.parsed_docs)} 문서)"}
        if on_progress: on_progress(step)

        if self.parsed_docs:
            self.vector.build_index(self.parsed_docs, text_field="summary")
            if save:
                self.vector.save(name="main")
                # Also save parsed docs as JSON for inspection
                with open(self.data_dir / "parsed_docs.json", "w", encoding="utf-8") as f:
                    json.dump(
                        [{k: v for k, v in d.items() if k != "raw_text"}
                         for d in self.parsed_docs],
                        f, ensure_ascii=False, indent=2,
                    )

        step["status"] = "done"
        if on_progress: on_progress(step)

        report["pdfs_processed"] = len(self.parsed_docs)
        report["elapsed_sec"] = round(time.time() - t0, 2)
        report["vector_backend"] = self.vector.backend
        return report

    def load_index(self, name: str = "main") -> bool:
        """Load previously-built index from disk."""
        try:
            self.vector.load(name=name)
            self.parsed_docs = self.vector.documents
            return True
        except (FileNotFoundError, IOError):
            return False

    # ─── Reference document workflow (v2) ─────────────────────────────
    def find_doc(self, filename: str) -> Optional[Dict]:
        """Find a parsed document by filename."""
        for doc in self.parsed_docs:
            if doc.get("filename") == filename:
                return doc
        return None

    def get_doc_fields(self, filename: str) -> Optional[Dict]:
        """
        Analyze a selected reference document and return:
          - doc_type      : classified document type
          - fields        : input field schema for this doc type
          - reference     : the reference doc's parsed values (for pre-fill)
        Returns None if document not found.
        """
        doc = self.find_doc(filename)
        if doc is None:
            return None

        doc_type = doc.get("doc_type", "기타 비용 지출")
        schema = self.FIELD_SCHEMAS.get(doc_type, self.FIELD_SCHEMAS["기타 비용 지출"])

        # Reference values for pre-fill (parsed from the actual PDF)
        reference = {
            "title": doc.get("title", ""),
            "purpose": doc.get("purpose", ""),
            "date": doc.get("date", ""),
            "location": doc.get("location", ""),
            "participants": doc.get("participants", ""),
            "payment_method": doc.get("payment_method", ""),
            "payee": doc.get("payee", ""),
            "amount_text": doc.get("amount_text", ""),
            "amount_num": doc.get("amount_num", 0),
            "budget_category": doc.get("budget_category", ""),
            "project_name": doc.get("project_name", ""),
        }

        return {
            "filename": filename,
            "doc_type": doc_type,
            "fields": schema,
            "reference": reference,
        }

    # ─── Workflow 2: Draft generation ─────────────────────────────────
    def generate(
        self,
        doc_type: str,
        user_input: Dict,
        top_k: int = 3,
        verify: bool = True,
        reference_filename: Optional[str] = None,
        on_progress: Optional[Callable[[Dict], None]] = None,
    ) -> Dict:
        """
        Full pipeline for drafting a new 공문서:
          1. Retrieve similar docs (semantic + keyword hybrid search)
             — if reference_filename given, that doc is pinned as context #1
          2. Generate draft via LLM with RAG context
          3. Verify draft (spelling, structure, numerics)
        """
        def progress(stage, status, **kwargs):
            if on_progress:
                on_progress({"stage": stage, "status": status, **kwargs})

        result = {
            "doc_type": doc_type,
            "user_input": user_input,
            "steps": [],
        }

        # Step 1: Retrieval
        progress("retrieval", "running", message="유사 문서 검색 중...")
        t = time.time()

        # Build a rich query from doc_type + purpose + key fields
        query_parts = [doc_type]
        if user_input.get("purpose"):
            query_parts.append(user_input["purpose"])
        if user_input.get("payment_method"):
            query_parts.append(user_input["payment_method"])
        query = " ".join(query_parts)

        # Reference doc pinning: selected doc becomes context #1,
        # remaining slots filled from hybrid search (excluding the pinned doc)
        pinned = None
        if reference_filename:
            ref_doc = self.find_doc(reference_filename)
            if ref_doc is not None:
                pinned = {"doc": ref_doc, "score": 1.0, "hybrid_score": 1.0,
                          "pinned": True}

        if pinned:
            searched = self.vector.hybrid_search(query, top_k=top_k + 1)
            others = [r for r in searched
                      if r["doc"].get("filename") != reference_filename]
            retrieved = [pinned] + others[: max(top_k - 1, 0)]
        else:
            retrieved = self.vector.hybrid_search(query, top_k=top_k)

        elapsed_retrieval = round(time.time() - t, 3)

        result["reference_filename"] = reference_filename
        result["steps"].append({
            "name": "유사 문서 검색" + (" (참조 문서 고정)" if pinned else ""),
            "elapsed_sec": elapsed_retrieval,
            "backend": self.vector.backend,
            "retrieved": [
                {
                    "filename": r["doc"].get("filename", ""),
                    "title": r["doc"].get("title", ""),
                    "doc_type": r["doc"].get("doc_type", ""),
                    "score": round(r["score"], 4),
                    "hybrid_score": round(r.get("hybrid_score", r["score"]), 4),
                    "pinned": r.get("pinned", False),
                }
                for r in retrieved
            ],
        })
        progress("retrieval", "done",
                 found=len(retrieved), elapsed=elapsed_retrieval)

        # Step 2: Generation
        progress("generation", "running", message="LLM 초안 생성 중...")
        t = time.time()
        gen_result = self.llm.generate_draft(
            doc_type=doc_type,
            user_input=user_input,
            retrieved_docs=retrieved,
        )
        elapsed_gen = round(time.time() - t, 3)

        result["steps"].append({
            "name": "LLM 초안 생성",
            "elapsed_sec": elapsed_gen,
            "model": gen_result["model"],
            "fallback": gen_result["fallback"],
        })
        result["draft"] = gen_result["draft"]
        result["model"] = gen_result["model"]
        result["total_amount"] = gen_result["total_amount"]
        result["korean_amount"] = gen_result["korean_amount"]
        progress("generation", "done",
                 model=gen_result["model"], elapsed=elapsed_gen)

        # Step 3: Verification
        if verify:
            progress("verification", "running", message="자동 검증 중...")
            t = time.time()

            expected_amount = gen_result["total_amount"] or None
            try:
                ep_count = int(user_input.get("participant_count") or 0) or None
                ep_unit = int(user_input.get("unit_cost") or 0) or None
            except (ValueError, TypeError):
                ep_count = ep_unit = None

            verify_result = self.verifier.verify(
                gen_result["draft"],
                expected_amount=expected_amount,
                expected_participant_count=ep_count,
                expected_unit_cost=ep_unit,
            )
            elapsed_v = round(time.time() - t, 3)

            result["verification"] = verify_result
            result["steps"].append({
                "name": "자동 검증",
                "elapsed_sec": elapsed_v,
                "score": verify_result["score"],
                "passed": verify_result["passed"],
                "issues_summary": verify_result["summary"],
            })
            progress("verification", "done",
                     score=verify_result["score"], issues=verify_result["summary"]["total"])

        return result

    # ─── Utility ──────────────────────────────────────────────────────
    def get_documents(self) -> List[Dict]:
        """Return list of all parsed documents (without raw text for brevity)."""
        return [
            {k: v for k, v in d.items() if k != "raw_text"}
            for d in self.parsed_docs
        ]

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Direct search interface for the UI."""
        results = self.vector.hybrid_search(query, top_k=top_k)
        return [
            {
                "filename": r["doc"].get("filename", ""),
                "title": r["doc"].get("title", ""),
                "doc_type": r["doc"].get("doc_type", ""),
                "amount": r["doc"].get("amount_text", ""),
                "date": r["doc"].get("date", ""),
                "purpose": r["doc"].get("purpose", ""),
                "score": round(r["score"], 4),
                "hybrid_score": round(r.get("hybrid_score", r["score"]), 4),
                "summary": r["doc"].get("summary", ""),
                "raw_text": r["doc"].get("raw_text", ""),
            }
            for r in results
        ]


if __name__ == "__main__":
    agent = GovDocAgent()

    def progress_print(step):
        print(f"  [{step.get('stage')}] {step.get('status', '')} — {step}")

    print("▶ Ingesting PDFs...")
    report = agent.ingest(on_progress=progress_print)
    print(f"\n✓ Ingested {report['pdfs_processed']} PDFs in {report['elapsed_sec']}s")
    print(f"  Backend: {report['vector_backend']}")
    print(f"  Errors: {len(report['errors'])}")

    print("\n▶ Testing search...")
    results = agent.search("식대", top_k=3)
    for r in results:
        print(f"  [{r['hybrid_score']:.3f}] {r['filename']} — {r['title']}")

    print("\n▶ Testing generation...")
    gen = agent.generate(
        doc_type="회의비 지출(식대)",
        user_input={
            "purpose": "AI 에이전트 개발 킥오프 미팅",
            "date": "2026.1.20.(화) 12:00",
            "location": "연구원 3층 회의실",
            "participants": "김상락, 이상일, 차민규, 문정현, 이세연",
            "participant_count": 8,
            "unit_cost": 25000,
            "payment_method": "카드결제",
            "project": "2025년 울산 빅데이터센터 운영",
        },
        on_progress=progress_print,
    )
    print("\n=== DRAFT ===")
    print(gen["draft"])
    print("\n=== VERIFICATION ===")
    print(f"Score: {gen['verification']['score']}/100  Passed: {gen['verification']['passed']}")
    for issue in gen["verification"]["issues"]:
        print(f"  [{issue['severity']}] {issue['description']}")
