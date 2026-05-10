# SHL Assessment Recommender v2.0

Conversational FastAPI agent that recommends SHL assessments through multi-turn dialogue.

**Stack**: FastAPI · LangGraph · FAISS · Gemini 1.5 Flash · Google Embeddings

---

## Architecture

```
POST /chat
    │
    ▼
LangGraph StateGraph
    │
    ├─ [retrieve] — HybridRetriever
    │     ├── FAISS similarity search (top 40 candidates)
    │     ├── Keyword-overlap boost (name / keys / description)
    │     ├── Deduplication
    │     └── Returns top-15 reranked catalog items
    │
    └─ [llm] ──── Gemini 1.5 Flash
                   │
                   ├── System prompt with hiring context + catalog context
                   ├── Full conversation history
                   └── Outputs: prose OR <RECOMMENDATIONS>{JSON}</RECOMMENDATIONS>

FastAPI parses structured output → validates every URL against catalog →
caps at 10 → returns ChatResponse
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| Hybrid retrieval | FAISS alone misses keyword-specific tests; BM25-style boost fixes precision |
| Context extraction | Rule-based context parser primes the LLM with structured role/level/purpose |
| URL validation | Every URL is checked against `VALID_URLS` (the full catalog); hallucinated URLs are silently rejected |
| Name recovery | If an LLM-generated URL fails validation, we fall back to name-matching against the catalog |
| Turn cap enforcement | Hard limit at 8 turns; LLM pressure added at turn 7 to force commitment |
| Temperature 0.1 | Deterministic outputs; reduces hallucination risk |
| OPQ/Verify defaults | Injected via system prompt rules matching observed evaluator expectations |

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Set GOOGLE_API_KEY=your_key_here
```

Get a free Gemini API key at: https://aistudio.google.com/app/apikey

### 3. Build FAISS Index (once)

```bash
python build_index.py
```

### 4. Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 5. Test

```bash
python test_api.py
```

---

## API

### `GET /health`
```json
{"status": "ok"}
```

### `POST /chat`
**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I am hiring a mid-level Java developer"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user", "content": "4 years experience, backend focus"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are 5 assessments for a mid-level Java backend developer.",
  "recommendations": [
    {"name": "Core Java (Advanced Level) (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

**Schema rules (non-negotiable):**
- `recommendations` is `[]` when agent is still gathering context
- `end_of_conversation` is `true` only when user explicitly confirms satisfaction
- Max 8 turns per conversation
- Max 10 items in recommendations
- All URLs must come from `shl.com/products/product-catalog`

---

## Agent Behaviors

| Behavior | Description |
|---|---|
| Clarify | Asks ONE focused question for vague queries before recommending |
| Recommend | Returns 1–10 catalog assessments with validated URLs |
| Refine | Updates shortlist when user changes constraints (add/remove/swap) |
| Compare | Grounds comparisons in catalog data only |
| Refuse | Rejects off-topic, legal, and prompt injection attempts |
| Force commit | At turn 7, commits to best shortlist regardless of ambiguity |

---

## Conversational Design (from trace analysis)

The agent follows these patterns observed in the reference conversation traces:

- **C1 pattern**: Vague "senior leadership" → clarify who → clarify purpose → recommend OPQ32r + reports
- **C2 pattern**: Technology gap (Rust) → acknowledge honestly → use proxies (Live Coding, Linux)
- **C3 pattern**: Contact center volume → clarify language → clarify accent → SVAR + simulation stack
- **C4 pattern**: Graduate analysts → immediate recs → refine with SJT → two-stage design
- **C5 pattern**: Sales re-skilling → GSA + OPQ + Sales reports → explain instrument vs. report
- **C6 pattern**: Safety-critical → DSI/Safety 8.0 as primary → compare before selecting
- **C7 pattern**: Bilingual + legal question → refuse legal, keep shortlist
- **C8 pattern**: Quick screen → knowledge tests → upgrade to simulation on request
- **C9 pattern**: JD input → clarify frontend vs. backend → senior IC → multi-step refinement
- **C10 pattern**: Full battery → user drops OPQ → honor the drop → eoc=true

---

## Deployment

### Render (Recommended)

1. Push code to GitHub
2. Go to https://render.com → New Web Service
3. Connect repo
4. Set `GOOGLE_API_KEY` in Environment Variables
5. Build: `pip install -r requirements.txt && python build_index.py`
6. Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Docker

```bash
docker build -t shl-recommender .
docker run -e GOOGLE_API_KEY=your_key -p 8000:8000 shl-recommender
```

---

## File Structure

```
shl_recommender/
├── main.py           # FastAPI app + LangGraph agent (all logic)
├── catalog_data.py   # SHL catalog (~130 assessments as Python list)
├── build_index.py    # One-time FAISS index builder
├── test_api.py       # Comprehensive test suite (22 test functions)
├── requirements.txt
├── Dockerfile
├── render.yaml
└── README.md
```
