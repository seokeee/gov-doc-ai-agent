"""
Module 7: FastAPI Server
- REST API exposing the agent
- Serves frontend at /
- Endpoints:
    POST /api/ingest     — (re)build vector index from PDFs
    GET  /api/documents  — list all ingested documents
    POST /api/search     — semantic search
    POST /api/generate   — RAG draft generation + verification
    POST /api/verify     — standalone verification
"""

import os
from pathlib import Path
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from agent import GovDocAgent


# ─── Pydantic Schemas ────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class UserInput(BaseModel):
    purpose: Optional[str] = None
    date: Optional[str] = None
    location: Optional[str] = None
    participants: Optional[str] = None
    participant_count: Optional[int] = None
    unit_cost: Optional[int] = None
    total_amount: Optional[int] = None
    payment_method: Optional[str] = None
    payee: Optional[str] = None
    project: Optional[str] = "2025년 울산 빅데이터센터 운영"


class GenerateRequest(BaseModel):
    doc_type: str
    user_input: UserInput
    top_k: int = 3
    verify: bool = True
    reference_filename: Optional[str] = None  # v2: pin selected doc as RAG context #1


class VerifyRequest(BaseModel):
    text: str
    expected_amount: Optional[int] = None
    expected_participant_count: Optional[int] = None
    expected_unit_cost: Optional[int] = None


# ─── App Initialization ──────────────────────────────────────────────
app = FastAPI(
    title="공문서 AI 작성 에이전트 API",
    description="RAG 기반 공공행정 문서 작성 지원 시스템",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = GovDocAgent()
_ingested = False


@app.on_event("startup")
async def startup():
    """Try to load existing index on startup."""
    global _ingested
    if agent.load_index():
        _ingested = True
        print(f"[startup] Loaded index with {len(agent.parsed_docs)} docs")
    else:
        print("[startup] No existing index. Run POST /api/ingest first.")


# ─── API Endpoints ───────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "ingested": _ingested,
        "document_count": len(agent.parsed_docs),
        "vector_backend": agent.vector.backend,
        "llm_available": agent.llm.is_available(),
        "llm_model": agent.llm.model if agent.llm.is_available() else None,
    }


@app.post("/api/ingest")
async def ingest():
    """Run full ingestion pipeline on pdfs/ directory."""
    global _ingested
    report = agent.ingest()
    _ingested = report["pdfs_processed"] > 0
    return report


@app.get("/api/documents")
async def list_documents():
    """List all ingested documents (without raw text)."""
    if not _ingested:
        raise HTTPException(status_code=400, detail="No documents ingested. Run POST /api/ingest first.")
    return {
        "count": len(agent.parsed_docs),
        "documents": agent.get_documents(),
    }


@app.get("/api/documents/{filename}")
async def get_document(filename: str):
    """Get full document including raw text."""
    for doc in agent.parsed_docs:
        if doc.get("filename") == filename:
            return doc
    raise HTTPException(status_code=404, detail=f"Document not found: {filename}")


@app.post("/api/search")
async def search(req: SearchRequest):
    if not _ingested:
        raise HTTPException(status_code=400, detail="No index loaded. Run POST /api/ingest first.")
    results = agent.search(req.query, top_k=req.top_k)
    return {
        "query": req.query,
        "count": len(results),
        "backend": agent.vector.backend,
        "results": results,
    }


@app.get("/api/doc-fields/{filename}")
async def doc_fields(filename: str):
    """
    v2: Analyze a selected reference document.
    Returns the input field schema for its doc type + reference values for pre-fill.
    """
    if not _ingested:
        raise HTTPException(status_code=400, detail="No index loaded. Run POST /api/ingest first.")
    result = agent.get_doc_fields(filename)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {filename}")
    return result


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    if not _ingested:
        raise HTTPException(status_code=400, detail="No index loaded. Run POST /api/ingest first.")

    user_input = req.user_input.model_dump(exclude_none=True)
    result = agent.generate(
        doc_type=req.doc_type,
        user_input=user_input,
        top_k=req.top_k,
        verify=req.verify,
        reference_filename=req.reference_filename,
    )
    return result


@app.post("/api/verify")
async def verify(req: VerifyRequest):
    result = agent.verifier.verify(
        req.text,
        expected_amount=req.expected_amount,
        expected_participant_count=req.expected_participant_count,
        expected_unit_cost=req.expected_unit_cost,
    )
    return result


# ─── Frontend Mount (if built) ───────────────────────────────────────
static_dir = Path(__file__).parent / "frontend"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def frontend_root():
        index = static_dir / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return JSONResponse({"message": "Frontend not built. See /docs for API."})
else:
    @app.get("/")
    async def api_root():
        return {
            "message": "공문서 AI 작성 에이전트 API",
            "docs": "/docs",
            "endpoints": [
                "GET  /api/health",
                "POST /api/ingest",
                "GET  /api/documents",
                "POST /api/search",
                "POST /api/generate",
                "POST /api/verify",
            ],
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
