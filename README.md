# 공문서 AI 작성 에이전트 · Gov-Doc AI Agent

**Phase 2 Prototype** — RAG 기반 공공행정 문서 작성 지원 시스템

울산연구원 공문서 8건을 학습 데이터로 사용하여, 유사 문서 검색 → LLM 초안 생성 → 자동 검증까지의 end-to-end 파이프라인을 구현한 실제 작동 시스템입니다.

---

## 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│ Frontend (HTML/CSS/Vanilla JS · served by FastAPI)          │
│  · 유사 문서 검색 │ 초안 생성 │ 자동 검증 │ 문서 DB          │
└────────────────────────────┬────────────────────────────────┘
                             │ REST
┌────────────────────────────▼────────────────────────────────┐
│ FastAPI Server (api_server.py)                              │
│  /api/health /api/ingest /api/documents                     │
│  /api/search /api/generate /api/verify                      │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│ GovDocAgent Orchestrator (agent.py)                         │
└──┬──────┬──────┬──────┬──────┬──────────────────────────────┘
   │      │      │      │      │
   ▼      ▼      ▼      ▼      ▼
 PDF   Parser  Vector  LLM   Verifier
Extractor      Search  Client
   │      │      │      │      │
   ▼      ▼      ▼      ▼      ▼
 PyMuPDF  정규식 FAISS  Claude 규칙기반
 +        기반  +      API    스펠/
 Tesseract     SBERT          용어/
 OCR           or TFIDF       수치 검증
              (fallback)
```

---

## 디렉터리 구조

```
gov_doc_agent/
├── pdf_extractor.py     # Module 1: PDF → text (digital + OCR)
├── doc_parser.py         # Module 2: text → structured metadata
├── vector_search.py      # Module 3: FAISS/TFIDF 벡터 검색 엔진
├── llm_client.py         # Module 4: Claude API 래퍼 + 한국어 숫자
├── verifier.py           # Module 5: 맞춤법·용어·수치 검증
├── agent.py              # Module 6: 전체 파이프라인 오케스트레이터
├── api_server.py         # Module 7: FastAPI REST 서버
├── frontend/
│   └── index.html        # 단일 HTML 프론트엔드 (바닐라 JS)
├── pdfs/                 # 처리할 원본 PDF 8건
├── vector_store/         # 빌드된 벡터 인덱스 (자동 생성)
├── data/                 # 파싱 결과 JSON (자동 생성)
├── requirements.txt
└── README.md
```

---

## 설치

### 1. 시스템 패키지 (OCR용)

**Ubuntu/Debian:**
```bash
sudo apt-get install -y tesseract-ocr tesseract-ocr-kor poppler-utils
```

**macOS:**
```bash
brew install tesseract tesseract-lang poppler
```

**Windows:** [Tesseract 설치 가이드](https://tesseract-ocr.github.io/tessdoc/Installation.html) 참조. 한국어 데이터(`kor.traineddata`)를 tessdata 폴더에 배치.

### 2. Python 의존성

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Claude API 키 설정 (선택사항, 권장)

API 키가 없어도 템플릿 기반 fallback으로 동작하지만, 실제 LLM 기반 초안 생성을 위해 권장:

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...
```

---

## 실행

### 빠른 시작

```bash
# 1. PDF를 pdfs/ 폴더에 넣기 (샘플 8개 포함됨)

# 2. 서버 실행
python3 -m uvicorn api_server:app --host 0.0.0.0 --port 8000

# 3. 브라우저에서 열기
open http://localhost:8000
```

첫 실행 시 문서 DB 탭에서 **[재인덱싱 실행]**을 클릭하여 PDF를 처리하세요 (8건 OCR, 약 2-3분 소요). 이후에는 저장된 인덱스가 자동 로드됩니다.

### CLI 단독 실행

각 모듈을 개별로 테스트 가능:

```bash
python3 pdf_extractor.py pdfs/1.pdf      # OCR 테스트
python3 doc_parser.py pdfs/1.pdf         # 메타데이터 파싱
python3 vector_search.py                 # 벡터 검색 데모
python3 verifier.py                      # 검증 데모
python3 agent.py                         # 전체 파이프라인 end-to-end
```

---

## API 엔드포인트

| Method | Path              | 설명                                        |
|--------|-------------------|---------------------------------------------|
| GET    | `/api/health`     | 서버 상태 · 인덱스 · 백엔드 · LLM 가용성    |
| POST   | `/api/ingest`     | PDFs → 파싱 → 벡터 인덱스 재빌드 (OCR 수행) |
| GET    | `/api/documents`  | 인덱싱된 모든 문서 메타데이터 리스트         |
| GET    | `/api/documents/{filename}` | 특정 문서 전체 (원문 포함)       |
| POST   | `/api/search`     | 의미 기반 유사 문서 검색 (하이브리드)        |
| POST   | `/api/generate`   | RAG 초안 생성 + 자동 검증                   |
| POST   | `/api/verify`     | 독립적 문서 검증                            |

Swagger UI: `http://localhost:8000/docs`

---

## 핵심 특징

### 1. 실제 PDF 처리 (Dual-mode)

- **디지털 PDF**: `PyMuPDF`로 텍스트 레이어 직접 추출 (즉시)
- **스캔/이미지 PDF**: `Tesseract OCR (kor+eng)` 자동 폴백 (문서당 약 15-20초)

### 2. 하이브리드 벡터 검색

- **프로덕션**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` + FAISS `IndexFlatIP` (코사인 유사도)
- **오프라인**: `scikit-learn TfidfVectorizer` + 한국어 토크나이저 자동 폴백
- **쿼리**: `search = semantic_score + 0.2 × keyword_overlap` (하이브리드 부스팅)

Korean-specialized 업그레이드: `model_name="jhgan/ko-sroberta-multitask"` 로 교체 시 768차원 한국어 특화 임베딩 사용 가능.

### 3. 한국어 숫자 변환 (검증 완료)

| 금액 | 변환 결과 | 실제 문서 일치 |
|------|-----------|---------------|
| 225,000 | 금이십이만오천원 | ✓ 1.pdf |
| 1,040,000 | 금일백사만원 | ✓ 2.pdf |
| 150,000 | 금십오만원 | ✓ 7.pdf |
| 300,000 | 금삼십만원 | ✓ 8.pdf |
| 525,000 | 금오십이만오천원 | ✓ 3.pdf |
| 567,000 | 금오십육만칠천원 | ✓ 4.pdf |
| 1,650,000 | 금일백육십오만원 | ✓ 5.pdf |

### 4. 자동 검증 (0-100 점수)

- **맞춤법**: 외래어 표기 (데이타→데이터), 띄어쓰기 (할수있→할 수 있), 공문 용어 (인쇄물비→유인물비) 등 10+ 규칙
- **필수 항목**: 지출금액 · 지급방법 · 예산과목 누락 검출
- **구조**: `끝.` 표시 여부
- **수치 정합성**: 본문 지출금액 vs. 사용자 입력값 대조
- **산출기초**: `단가 × 인원 = 총액` 계산 검증

점수 산정: `100 - 오류×20 - 경고×5 - 안내×1`

---

## 확장 계획

- **디자인**: `hunspell-dict-ko` 통합으로 전체 한글 맞춤법 검사 강화
- **데이터**: 행정안전부 공통표준용어 DB 연결하여 행정 용어 표준화
- **모델**: Korean-SBERT로 임베딩 교체, re-rank 모델 (ko-reranker) 도입
- **평가**: `AI Hub 행정문서 기계독해 데이터셋`으로 벤치마크

---

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `ANTHROPIC_API_KEY` | 없음 | Claude API 키. 미설정 시 템플릿 폴백 |

---

## 테스트된 환경

- Python 3.12
- Ubuntu 24.04
- Tesseract 5.3 (kor traineddata)
- 8 PDFs: 울산연구원 「2025년 울산 빅데이터센터 운영」 지출 결재문서

실측 데이터:
- 8개 PDF 전체 인덱싱: **137초** (OCR 포함)
- 검색 응답: **1ms** (인덱스 로드 후)
- 초안 생성 (템플릿): **즉시**
- 초안 생성 (Claude API): **3-8초**
- 검증: **1ms**

---

## 라이선스

Phase 2 Prototype. 학술 연구 및 과제 제출용.
