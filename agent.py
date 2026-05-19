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
    """

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

    # ─── Workflow 2: Draft generation ─────────────────────────────────
    def generate(
        self,
        doc_type: str,
        user_input: Dict,
        top_k: int = 3,
        verify: bool = True,
        on_progress: Optional[Callable[[Dict], None]] = None,
    ) -> Dict:
        """
        Full pipeline for drafting a new 공문서:
          1. Retrieve similar docs (semantic + keyword hybrid search)
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

        retrieved = self.vector.hybrid_search(query, top_k=top_k)
        elapsed_retrieval = round(time.time() - t, 3)

        result["steps"].append({
            "name": "유사 문서 검색",
            "elapsed_sec": elapsed_retrieval,
            "backend": self.vector.backend,
            "retrieved": [
                {
                    "filename": r["doc"].get("filename", ""),
                    "title": r["doc"].get("title", ""),
                    "doc_type": r["doc"].get("doc_type", ""),
                    "score": round(r["score"], 4),
                    "hybrid_score": round(r.get("hybrid_score", r["score"]), 4),
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
