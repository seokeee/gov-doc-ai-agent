# 공문서 AI 작성 에이전트

> RAG (Retrieval-Augmented Generation) + HITL (Human-in-the-Loop) 기반 공공행정 문서 초안 생성 및 검증 시스템

## 프로젝트 개요

지방자치단체 및 공공기관의 공문서 기안 업무에서 발생하는 비효율을 해소하기 위한 AI 에이전트 시스템입니다.

과거 공문서를 의미 기반으로 검색하여 LLM에 맥락으로 제공하고, 자동 검증을 거친 후 작성자가 최종 검토하는 인간-AI 협업 구조를 갖습니다.

## 해결하려는 문제

- 과거 사례 탐색에 과다한 시간 소요
- 작성자별 문서 품질 편차 발생
- 오탈자 및 행정 용어 비일관성
- 결재 라인·서식 확인 부담 누적

## 기술 스택

- **Backend**: Python, FastAPI
- **LLM**: Google Gemini API
- **Embedding**: sentence-transformers (multilingual)
- **Vector Search**: FAISS
- **OCR**: Tesseract (Korean)
- **PDF Processing**: PyMuPDF

## 프로젝트 상태

현재 초기 개발 단계입니다.

