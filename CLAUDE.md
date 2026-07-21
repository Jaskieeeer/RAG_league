# CLAUDE.md — LoL Knowledge RAG

## What this project is

A production-grade RAG application over League of Legends data (Data Dragon JSON +
LoL wiki via MediaWiki API). Portfolio project: the goal is code the author fully
understands and can defend in an interview. Built in phases; do not implement a
later phase before the current one is complete and tested.

Phases (infrastructure):

1. Core RAG package (`lolrag/`): ingestion, indexing, retrieval, generation. Tests.
2. FastAPI backend exposing the pipeline. Security hardening (see below).
3. React chat frontend (single small app, no framework sprawl).
4. Docker + compose for local dev.
5. GCP Cloud Run deploy + GitHub Actions CI/CD.
6. Optional: GKE migration.

## RAG evolution roadmap

The pipeline itself evolves separately from the infrastructure phases, one
technique at a time. Rules of the roadmap:

- v1 is a deliberately NAIVE baseline: single dense retriever, grounded prompt,
  nothing clever. It must work end to end before anything below is attempted.
- v2 is the EVALUATION HARNESS, built before any retrieval upgrade: a
  checked-in golden dataset of question/expected-source pairs over the corpus,
  retrieval metrics (hit rate @k, MRR), answer metrics (faithfulness /
  groundedness via LLM-as-judge), runnable as `uv run pytest -m eval` and
  traced in LangSmith.
- Every upgrade after v2 is a separate branch + eval run. An upgrade is kept
  only if the numbers justify it; either way the result goes in the README
  ablation table (technique, hit rate, MRR, latency, verdict). Negative
  results stay in the table — they are evidence of method, not failure.
- Upgrade queue, roughly in order: multi-query / RAG-fusion with RRF;
  hybrid search (BM25 + dense, fused with RRF); metadata query construction
  (patch/champion/class filters via structured output); logical routing
  between collections (gameplay vs lore vs patches); multi-representation
  indexing for long lore pages; cross-encoder re-ranking; HyDE if evals
  show question/passage vocabulary mismatch.
- One technique per iteration. Never stack two unevaluated changes.

## Stack (do not substitute without asking)

- Python 3.13, uv for dependency management (`uv add`, never pip install)
- LangChain 1.x: imports come from `langchain_core`, `langchain_classic`,
  `langchain_community`, `langchain_chroma`, `langchain_huggingface`,
  `langchain_google_genai`, `langchain_text_splitters`. Never import from
  legacy paths like `langchain.prompts` — they do not exist in 1.x.
- LLM: Gemini via `ChatGoogleGenerativeAI` (default `gemini-3.5-flash`,
  fallback `gemini-3.1-flash-lite`). Model name always from config, never hardcoded.
- Embeddings: `HuggingFaceEmbeddings` (all-MiniLM-L6-v2) unless config says otherwise.
- Vector store: Chroma (persistent client). Deterministic document ids derived from
  content+source hash so re-ingestion upserts instead of duplicating.
- Backend: FastAPI + pydantic v2. Frontend: React + Vite. Tests: pytest.

## The author is learning

The human is rebuilding hands-on coding skill. Therefore:

- Explain non-obvious design decisions in the PR/commit description or chat,
  not in code comments.
- Prefer plans and small diffs over large generated dumps.
- When asked a question, answer it; do not silently implement.
- If the human's suggestion has a flaw, say so directly before implementing.

## Code style — strict

- No inline commentary comments, no filler comments, no emojis anywhere
  (code, comments, commit messages, docs).
- Allowed comments, only these two kinds:
  1. Docstrings on every public function/class: one-line summary, then
     Args/Returns/Raises with types and meaning.
  2. Section separator comments in long modules:
     `# ---------- ingestion ----------`
- Type hints on all function signatures.
- pydantic models for all API request/response bodies and all structured LLM output.
- No `print` in library code; use `logging`.
- Config via pydantic-settings from environment; `.env` is gitignored, a
  `.env.example` documents every variable.

## Security requirements (Phase 2, non-negotiable)

- Rate limiting on every public endpoint (slowapi): per-IP request caps,
  stricter caps on the LLM-backed endpoint.
- API key auth for the query endpoint (simple header key for v1; document the
  upgrade path to OAuth if this were multi-user).
- Hard daily budget cap on LLM calls; the service degrades to 429 with a clear
  message when exceeded, it never silently keeps spending.
- Input validation: max question length, reject non-text payloads, strip control
  characters. Retrieved context and user input are clearly separated in prompts.
- CORS locked to the deployed frontend origin.
- Secrets only via environment / GCP Secret Manager. If a key ever appears in
  code or git history, stop and tell the human immediately.
- Request/response logging without logging secrets or full user IPs (truncate).

## Data sources

- Data Dragon (versioned JSON: champions, items, runes) — primary, no scraping.
- LoL wiki via MediaWiki `api.php` — polite rate limiting, User-Agent set,
  respect robots and ToS. Content is CC BY-SA: attribution goes in the README
  and the frontend footer, along with Riot's fan-project disclaimer.
- Raw HTML scraping is a last resort and requires explicit human approval.

## Commands

- `uv run pytest` — run tests (must pass before any commit)
- `uv run ruff check . && uv run ruff format --check .` — lint/format gate
- `uv run uvicorn app.main:app --reload` — dev server (Phase 2+)

## Git conventions

- Small, single-purpose commits; imperative mood messages ("Add rate limiter",
  not "Added"/"adding stuff").
- Never commit: `.env`, Chroma persistence directories, model caches,
  `node_modules`, build artifacts.
- The human reviews every diff before it is committed. Do not auto-commit
  unless explicitly told to.
