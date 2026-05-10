# SHL Assessment Recommender — Architecture Reference

> **Stack:** FastAPI · LangGraph · FAISS · Gemini 2.5 Flash · Google Embeddings (`gemini-embedding-001`)

---

## Table of Contents

1. [High-Level Flow](#high-level-flow)
2. [Component Breakdown](#component-breakdown)
3. [FAISS Index](#faiss-index)
4. [Hybrid Retriever](#hybrid-retriever)
5. [LangGraph Agent](#langgraph-agent)
6. [Hiring Context Extractor](#hiring-context-extractor)
7. [Response Validation Pipeline](#response-validation-pipeline)
8. [Conversational Rules](#conversational-rules)
9. [API Contract](#api-contract)
10. [Configuration Reference](#configuration-reference)
11. [File Structure](#file-structure)

---

## High-Level Flow

```
POST /chat (full conversation history)
        │
        ▼
  Turn-cap guard  (>8 turns → hard stop)
        │
        ▼
  Convert messages → LangChain HumanMessage / AIMessage
        │
        ▼
  LangGraph StateGraph
  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │  [retrieve]                                         │
  │    ├─ extract_context_from_messages()               │
  │    │    └─ returns HiringContext (role, seniority,  │
  │    │         purpose, skills, languages, industry)  │
  │    ├─ hybrid_retrieve(vectorstore, query, k=15)     │
  │    │    ├─ FAISS similarity_search (top 30 docs)    │
  │    │    ├─ keyword_boost_score per doc              │
  │    │    └─ score = 0.7×position + 0.3×boost        │
  │    └─ formats catalog_context string for LLM       │
  │                                                     │
  │  [llm]                                              │
  │    ├─ build system prompt with hiring_context       │
  │    │   and catalog_context                          │
  │    ├─ inject turn-pressure hint at turn ≥7          │
  │    ├─ invoke Gemini 2.5 Flash (temp=0.1)            │
  │    └─ parse <RECOMMENDATIONS>{JSON}</RECOMMENDATIONS>│
  │                                                     │
  └─────────────────────────────────────────────────────┘
        │
        ▼
  assemble_response()
    ├─ validate every URL against VALID_URLS set
    ├─ recover hallucinated URLs via name-match fallback
    ├─ deduplicate, cap at 10
    └─ strip <RECOMMENDATIONS> block from reply text
        │
        ▼
  ChatResponse { reply, recommendations[], end_of_conversation }
```

---

## Component Breakdown

| Component | File | Responsibility |
|---|---|---|
| FastAPI app | `main.py` | HTTP layer, startup, CORS, exception handler |
| LangGraph graph | `main.py` | Two-node DAG: `retrieve → llm → END` |
| Hybrid retriever | `main.py` | FAISS + keyword boost, dedup, top-k |
| Context extractor | `main.py` | Rule-based parser for role/seniority/purpose/skills |
| FAISS index | `build_index.py` + `main.py` | Embedding + persistence; lazy-loaded at startup |
| Catalog | `catalog_data.py` | ~130 Individual Test Solutions; reports/bundles filtered at data layer |
| Validator | `main.py` | URL whitelist + name-match fallback; enforces MAX_RECS=10 |
| Test suite | `test_api.py` | 22 test functions covering all agent behaviors |

---

## FAISS Index

### Build once, load forever

```
python build_index.py
```

This calls `_build_catalog_indexes()` (populates `CATALOG_LOOKUP`, `CATALOG_BY_NAME`, `VALID_URLS`) then `build_faiss_index()`, which:

1. Iterates `get_catalog()` → list of ~130 filtered items
2. Converts each to a `Document` via `_make_doc()`:
   - `page_content` = Name + Description + Test Types + Job Levels + Duration + Adaptive + Remote
   - `metadata` = `entity_id`, `name`, `link`, `keys`, `primary_code`
3. Embeds with `GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")`
4. Saves to `./faiss_shl_index/` (two files: `index.faiss`, `index.pkl`)

### Load path at startup

```python
load_or_build_faiss()
  ├─ if ./faiss_shl_index/ exists → FAISS.load_local(allow_dangerous_deserialization=True)
  └─ else → build_faiss_index()
```

> **Note:** `allow_dangerous_deserialization=True` is required by LangChain's FAISS wrapper when loading a pickled index. The pickle originates from your own `build_index.py` run — it is not user-supplied.

### Committed index files

The repo ships `index.faiss` and `index.pkl` at the root so Render / Docker can skip the embedding API call at deploy time. The build command in `render.yaml` still runs `build_index.py` to regenerate them fresh; you can remove that step if you prefer to rely on the committed files.

---

## Hybrid Retriever

`hybrid_retrieve(vectorstore, query, k=15)`

### Step 1 — FAISS similarity search

Fetches `k*2 = 30` candidates via cosine similarity on `gemini-embedding-001` vectors.

### Step 2 — Keyword boost

For each candidate, `_keyword_boost_score(item, query_lower)` computes token-overlap ratios:

```
boost = 0.5 × name_overlap + 0.3 × key_overlap + 0.2 × desc_overlap
```

All ratios are `|tokens ∩ field_tokens| / |query_tokens|`.

### Step 3 — Combined score

```
final_score = 0.7 × position_score + 0.3 × boost
```

`position_score` is the normalized rank from FAISS (1st result = ~1.0, last = ~0.0).

### Step 4 — Dedup + cap

Tracks `entity_id` to prevent duplicate items. Returns top `k=15` after sorting by `final_score` descending.

### Optional filters (passed from context extractor)

- `job_level_filter` — drops items whose `job_levels` list doesn't include the target level (only applied if item has a non-empty levels list)
- `key_filter` — drops items whose `keys` don't overlap with the requested test type categories

---

## LangGraph Agent

### State shape

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # full conversation (LangChain messages)
    catalog_context: str                       # formatted retrieval results
    hiring_context_summary: str                # "Role: X | Seniority: Y | ..."
    structured_output: Optional[dict]          # parsed JSON from <RECOMMENDATIONS>
    turn_count: int                            # number of user turns so far
```

### Graph topology

```
START → retrieve → llm → END
```

Both nodes are pure functions; the graph is stateless across HTTP requests (all state is passed in per-request).

### `retrieve` node

Builds composite query from the last 4 user messages, runs `hybrid_retrieve`, formats results as a bullet list with URL + description snippet, writes to `catalog_context` and `hiring_context_summary`.

### `llm` node

1. Formats `SYSTEM_PROMPT` with `hiring_context` and `catalog_context`
2. Appends turn-pressure text if `turn_count ≥ 7`
3. Prepends `SystemMessage` to full conversation history
4. Calls `ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)`
5. Extracts `<RECOMMENDATIONS>…</RECOMMENDATIONS>` with regex, strips code fences, parses JSON
6. Writes `AIMessage` + `structured_output` back to state

---

## Hiring Context Extractor

`extract_context_from_messages(messages)` returns a `HiringContext` dataclass populated by rule-based regex over all user messages concatenated:

| Field | Detection method |
|---|---|
| `seniority` | Regex patterns for Executive / Director / Manager / Mid-Professional / Graduate / Entry-Level |
| `purpose` | Keywords: select/recruit/hire → `"selection"`; develop/360/upskill → `"development"` |
| `languages` | Named language list (Spanish, French, German, …) |
| `industry` | Domain keywords: healthcare, manufacturing, finance, sales |
| `skills` | ~50 tech/role keywords (java, python, sql, react, devops, …) |
| `role` | Inferred from skills if not explicitly stated |

`HiringContext.is_sufficient` is `True` when both `role` and `seniority` are set — used by the LLM system prompt to decide whether to clarify or recommend.

---

## Response Validation Pipeline

`assemble_response(state) → ChatResponse`

```
structured_output dict
        │
        ▼
  for each rec in recommendations[]:
    validate_recommendation(rec)
      ├─ url ∈ VALID_URLS?  → use it
      ├─ url not in set?    → try CATALOG_BY_NAME[name.lower()]
      │     ├─ found → recover URL, log INFO
      │     └─ not found → drop (log WARNING)
      └─ cross-check: canonical name + test_type from catalog
        │
        ▼
  deduplicate by url (seen_urls set)
        │
        ▼
  cap at MAX_RECS = 10
        │
        ▼
  strip <RECOMMENDATIONS> block from reply text
        │
        ▼
  ChatResponse(reply, recommendations, end_of_conversation)
```

`VALID_URLS` is a `set[str]` built at startup from `get_catalog()` — O(1) lookup per validation.

---

## Conversational Rules

These are enforced jointly by the system prompt and post-processing code:

| Rule | Enforcement |
|---|---|
| Ask ONE clarifying question for vague queries | System prompt instruction |
| Need role + seniority before recommending | `HiringContext.missing_fields` injected into prompt |
| Max 8 turns per conversation | Hard cap in `/chat` handler (`_is_over_cap`) |
| Force shortlist at turn 7 | Turn-pressure text appended to system prompt |
| `end_of_conversation=true` only on explicit confirmation | System prompt + LLM judgment |
| Max 10 recommendations | `assemble_response` hard cap |
| All URLs from `shl.com/products/product-catalog` | `VALID_URLS` whitelist validation |
| OPQ32r default for senior/professional hires | System prompt rule 11 |
| Cognitive test default for analytical roles | System prompt rule 12 |
| Refuse off-topic, legal, DEI, compensation queries | System prompt rule 1 |
| Honor refinements without restarting | System prompt rule 7 |

---

## API Contract

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /chat`

**Request**
```json
{
  "messages": [
    { "role": "user",      "content": "I need to hire a mid-level Java developer" },
    { "role": "assistant", "content": "Are you screening for backend or full-stack?" },
    { "role": "user",      "content": "Backend, 4 years experience" }
  ]
}
```

**Response**
```json
{
  "reply": "Here are my recommendations for a mid-level Java backend developer.",
  "recommendations": [
    {
      "name": "Core Java (Advanced Level) (New)",
      "url": "https://www.shl.com/products/product-catalog/...",
      "test_type": "K"
    },
    {
      "name": "OPQ32r",
      "url": "https://www.shl.com/products/product-catalog/...",
      "test_type": "P"
    }
  ],
  "end_of_conversation": false
}
```

**`test_type` codes**

| Code | Category |
|---|---|
| A | Ability & Aptitude |
| K | Knowledge & Skills |
| P | Personality & Behavior |
| B | Biodata & Situational Judgment |
| S | Simulations |
| C | Competencies |
| D | Development & 360 |
| E | Assessment Exercises |

**Schema invariants**
- `recommendations` is `[]` when agent is still gathering context
- `end_of_conversation` is `true` only on explicit user confirmation
- Max 8 turns per session; client must start a new session after the cap
- Max 10 items in `recommendations`

---

## Configuration Reference

All configuration is via environment variables (loaded from `.env` by `python-dotenv`):

| Variable | Required | Default | Description |
|---|---|---|---|
| `GOOGLE_API_KEY` | Yes | — | Gemini API key from [aistudio.google.com](https://aistudio.google.com/app/apikey) |

Tuning constants in `main.py`:

| Constant | Value | Description |
|---|---|---|
| `FAISS_INDEX_PATH` | `"faiss_shl_index"` | Directory for persisted index |
| `MAX_TURNS` | `8` | Hard conversation turn cap |
| `FORCE_REC_TURN` | `7` | Turn at which LLM is pressured to commit |
| `RETRIEVAL_K` | `20` | FAISS candidates before reranking |
| `RERANK_TOP` | `15` | Items passed to LLM context after reranking |
| `MAX_RECS` | `10` | Hard cap on recommendations returned |

---

## File Structure

```
shl_recommender/
├── main.py              # FastAPI app + LangGraph agent (all core logic)
├── catalog_data.py      # SHL catalog (~130 Individual Test Solutions)
│                        # Reports, bundles, and packaged solutions filtered here
├── build_index.py       # One-time FAISS index builder (run before first start)
├── test_api.py          # 22-function test suite covering all agent behaviors
├── requirements.txt     # Python dependencies
├── Dockerfile           # python:3.11-slim; index built at runtime via CMD
├── render.yaml          # Render.com deploy config (free tier)
├── .env.example         # Environment variable template
├── setup.sh             # Helper: venv + install + build index
├── faiss_shl_index/     # Generated by build_index.py (committed for fast deploys)
│   ├── index.faiss
│   └── index.pkl
└── ARCHITECTURE.md      # This document
```
