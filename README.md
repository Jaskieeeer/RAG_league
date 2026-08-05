# LoL Knowledge RAG

[![CI](https://github.com/Jaskieeeer/RAG_league/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/Jaskieeeer/RAG_league/actions/workflows/ci.yml)

A retrieval-augmented generation system for answering League of Legends questions, built over Riot's first-party JSON APIs: Data Dragon, Community Dragon, and the Riot Universe.

**Status:** The core RAG package works end to end on Postgres with pgvector: fetch, entity load, document and chunk build, dense retrieval, grounded generation. The evaluation harness that gates every retrieval upgrade is in place and scores the pipeline against a no-retrieval control, but it has not been run against the rebuilt corpus yet, so the ablation table below carries headers and no numbers. Next: a FastAPI backend.

**Stack:** Python 3.13, uv, LangChain 1.x, Gemini (`ChatGoogleGenerativeAI`), HuggingFace embeddings (all-MiniLM-L6-v2, 384 dimensions), PostgreSQL with pgvector via SQLAlchemy 2.0 and Alembic. Structured entity rows and embeddings live in one store, so metadata filters and vector ranking can compose in a single SQL statement. Retrieval is a hand-written SQLAlchemy Core query ranked by cosine distance against an HNSW index rather than a vector store wrapper, so the ranking is one readable SELECT that can be explained and profiled. FastAPI and React planned for later phases.

## Setup

```
uv sync
docker compose up -d
uv run alembic upgrade head
```

Copy `.env.example` to `.env` and set `GOOGLE_API_KEY` (from [Google AI Studio](https://aistudio.google.com/apikey)). The remaining variables have working defaults.

`docker compose up -d` starts the Postgres/pgvector container that both the entity store and the database-backed tests run against, and `alembic upgrade head` creates its schema. The credentials in `docker-compose.yml` are a local-development default and are never reused outside a developer machine.

## Usage

```
uv run python -m lolrag ingest
uv run python -m lolrag index
uv run python -m lolrag ask "What does Jinx's ultimate ability do?"
```

`ingest` warms the on-disk response cache for the pinned Data Dragon patch, loads every entity, association and numeric value row into Postgres, then builds the retrieval documents and embeds their chunks. `index` reruns that last stage alone against the entity rows already stored, which is what to use after changing how a document is rendered. Both are upserts keyed on each entity's own identity, and a document whose content has not changed is not re-chunked or re-embedded, so a rerun over a static corpus embeds nothing. `ask` retrieves the nearest chunks and generates an answer grounded in them, citing every chunk it used.

## Corpus

Every document is generated from a stored entity row and belongs to one of four collections:

- `abilities` - one document per champion ability, carrying the description, the substituted tooltip, and the per-rank numbers Community Dragon's raw game data publishes: base damages, ratios, cooldowns and costs, each rendered with the champion stat it scales off and the damage type it deals.
- `champion_stats` - one document per playable champion's base statistics, each stat given as its level-1 value and the per-level growth figure the source publishes. Held apart from the biography on purpose, because a block of bare numbers inside lore prose degrades lore retrieval.
- `equipment` - purchasable items with their cost, tags, stat values and the game modes they are sold in, plus runes and summoner spells.
- `lore` - champion biographies, long-form Riot Universe stories, and faction overviews.

Nothing in the corpus is truncated or summarised. Content is dropped only where the source publishes none: a tooltip whose substitution a token blocked is omitted entirely rather than shipped half-resolved, and a rune's long description is dropped only when it repeats the short one verbatim.

Items are filtered down to the ones a player can actually buy: purchasable, listed in the shop, available on at least one map, and not declared a mode variant of another item. What survives is then checked for copies that would put two different prices on one name in one game mode, and the copy the source itself names loses. Every filtered row stays in the `items` table as a graph node with its components intact, because this is a document-build filter and not a deletion.

## Tests

```
uv run pytest
uv run pytest -m corpus
uv run pytest -m integration
uv run pytest -m eval
```

The default run needs no network access and no API key, but it does need the Postgres container from Setup to be running with migrations applied, because the schema and retrieval tests assert against a real database rather than a mock. The `corpus` run loads the whole corpus from the warm cache in `data/cache` and asserts its per-table and per-collection row counts; it skips itself when that cache is absent, so a fresh clone has to run `ingest` once first. It expects an empty database and rolls its transaction back, leaving one exactly as empty as it found it. The `integration` run additionally needs network access and a configured `GOOGLE_API_KEY`. The `eval` run needs both of those plus an indexed corpus, and spends real tokens.

Passing `-m` on the command line replaces the default marker filter rather than narrowing it, so `uv run pytest` and `uv run pytest -m corpus` are two separate gates and both are run before a commit.

## Evaluation

The harness scores two systems against the checked-in golden dataset in `lolrag/eval/golden_dataset.json`: the pipeline, and a no-retrieval control that answers from pretraining alone. The control holds everything but retrieval constant - same model, same fallback, same temperature, same assistant persona - so a gap between the two is attributable to the retrieved context and to nothing else. The one thing that cannot be held constant is the grounding clause, since "answer using only the context below" with no context would measure obedience rather than knowledge.

Questions are authored into three blocks, and the block decides which figure a question contributes to:

- **retrieval** - questions with a supported answer in the corpus. They name the `doc_key`s that answer them, and they alone produce hit rate at 1, 3 and k and MRR. They are also the only block the control is run against.
- **refusal** - questions naming a detail the corpus does not carry. Scored only on whether the answer declined, and never against the control, because declining is correct behaviour for a grounded system and merely unhelpful for a general one.
- **limitation** - list and aggregation questions that dense retrieval cannot answer in v1 but the entity tables could answer with SQL. Reported unscored, so a known architectural gap neither flatters nor drags a headline number, and each one converts to a scored question on the routing upgrade.

Answers are graded by the check their question declares. Number questions go through a deterministic numeric check with no model in the loop, because a general-knowledge model asked to grade numbers it also knows would quietly grade its own recall. Prose questions go to an LLM entailment judge that compares against the dataset's reference answer and is forbidden its own knowledge of the game, since the corpus is deliberately narrower than the live game. Groundedness is scored separately, by an LLM faithfulness judge on a 1 to 5 scale. A sample of entailment judgements is written out each run for hand-scoring, and the judge-human agreement over the ones scored by hand is folded into the next report: the judge is an instrument, and an unvalidated instrument measures nothing.

Run it with `uv run pytest -m eval`. Reports are written to `EVAL_REPORT_DIR` as JSON and Markdown, with a headline comparison of the two systems, the refusal and limitation blocks on their own lines, and a per-category stratification. Both systems generate with `EVAL_MODEL_NAME` rather than the product's `LLM_MODEL_NAME`, so a run can be priced and rate-limited independently of what `ask` answers with, and runs are traced in LangSmith when `LANGSMITH_TRACING` is on.

### Ablation table

Every retrieval upgrade is a separate branch and a separate eval run, kept only if the numbers justify it. The result is recorded here either way, because a negative result is evidence about the method rather than a failure to hide.

| Technique | Hit rate @k | MRR | Latency | Verdict |
| --- | --- | --- | --- | --- |

No evaluation has been run yet: the v1 dense baseline row lands with the first run of the harness against the rebuilt corpus, and each upgrade adds one row beneath it.

## Attribution

LoL Knowledge RAG isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone officially involved in producing or managing Riot Games properties. Riot Games and all associated properties are trademarks or registered trademarks of Riot Games, Inc.

All corpus content comes from Riot's first-party JSON APIs: Data Dragon, Community Dragon, and the Riot Universe. Nothing is scraped.
