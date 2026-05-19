#!/bin/bash
# Start the gov-doc AI agent server
# Usage: ./run.sh [port]

PORT=${1:-8000}
cd "$(dirname "$0")"

# Load .env if exists
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " 공문서 AI 작성 에이전트"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "⚠  ANTHROPIC_API_KEY 미설정 → 템플릿 fallback 모드로 동작"
  echo "   실제 LLM 생성을 원하면: export ANTHROPIC_API_KEY=sk-ant-..."
else
  echo "✓  ANTHROPIC_API_KEY 설정됨 → Claude API 활성화"
fi

echo ""
echo "▶  서버 시작 중..."
echo "   Frontend:  http://localhost:$PORT"
echo "   API Docs:  http://localhost:$PORT/docs"
echo ""

python3 -m uvicorn api_server:app --host 0.0.0.0 --port "$PORT" --reload
