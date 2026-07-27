# LoL Knowledge RAG

[![CI](https://github.com/Jaskieeeer/RAG_league/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/Jaskieeeer/RAG_league/actions/workflows/ci.yml)

A retrieval-augmented generation system for answering League of Legends champion questions, built over Riot's first-party JSON APIs: Data Dragon, Community Dragon, and the Riot Universe.

**Status:** The core RAG package works end to end as a naive v1 baseline, and the evaluation harness that gates every retrieval upgrade after it is in place. Current work is rebuilding the corpus onto Postgres/pgvector as a multi-collection entity store, so that metadata filters and vector ranking compose in a single SQL statement. Next: a FastAPI backend.

**Stack:** Python 3.13, uv, LangChain 1.x, Gemini (`ChatGoogleGenerativeAI`), HuggingFace embeddings, PostgreSQL with pgvector via SQLAlchemy 2.0 and Alembic. Chroma remains the live vector store until the migration completes. FastAPI and React planned for later phases.

## Setup

```
uv sync
docker compose up -d
uv run alembic upgrade head
```

Copy `.env.example` to `.env` and set `GOOGLE_API_KEY` (from [Google AI Studio](https://aistudio.google.com/apikey)). The remaining variables have working defaults.

`docker compose up -d` starts the Postgres/pgvector container the entity store and the database-backed tests run against, and `alembic upgrade head` creates its schema. The credentials in `docker-compose.yml` are a local-development default and are never reused outside a developer machine.

## Usage

```
uv run python -m lolrag ingest
uv run python -m lolrag ask "What does Jinx's ultimate ability do?"
```

`ingest` fetches every champion for the pinned Data Dragon patch and builds a persistent Chroma index in `./data/chroma`. `ask` retrieves the most relevant champions and generates a grounded answer with cited sources.

The corpus covers champion lore, roles, and ability descriptions as prose, plus ability cooldowns, costs, and ranges, which Data Dragon publishes as real values.

It does not yet include ability damage numbers or scaling ratios. The endpoints currently ingested ship these as unresolved placeholders: Data Dragon tooltips read `{{ qdamage }}` with empty `vars` and `datavalues` across all 692 spells, and Community Dragon's champion records repeat the gap as `@QDamage@` with zeroed `effectAmounts`.

The values do exist in Community Dragon's raw game data (`/game/data/characters/{champion}/{champion}.bin.json`), which exposes per-rank arrays such as Aatrox's `QBaseDamage` of 10/25/40/55/70 and `QTotalADRatio` of 60-90%. Ingesting them is part of the corpus rebuild now in progress; rendering a finished tooltip string additionally requires evaluating Riot's `GameCalculation` formula graph, so the cheaper first step is to expose the named per-rank values as structured facts.

Until then, questions about ability damage have no supporting context in the corpus, and the intended behaviour is to answer that the sources do not specify rather than to invent a value.

## Tests

```
uv run pytest
uv run pytest -m integration
uv run pytest -m eval
```

The default run needs no network access and no API key, but it does need the Postgres container from Setup to be running with migrations applied, because the schema tests assert against a real database rather than a mock. The `integration` run additionally needs network access and a configured `GOOGLE_API_KEY`. The `eval` run needs both of those plus a built index, and spends real tokens.

Passing `-m` on the command line replaces the default marker filter rather than narrowing it, so `uv run pytest` with no arguments is the gate to run before a commit.

## Evaluation

The harness scores retrieval and generation separately against a hand-authored golden dataset of 20 questions in `lolrag/eval/golden_dataset.json`, each pairing a question with the champion whose document should be retrieved. Retrieval is measured by hit rate at 1, 3 and k, and by MRR. Generation is measured by an LLM-as-judge faithfulness score from 1 to 5. Runs are traced in LangSmith.

The dataset deliberately mixes two categories. `factual` questions have a supported answer in the corpus. `refusal` questions name a champion the retriever should still find, but ask for a detail the corpus does not carry, so a faithful system reports that the sources do not specify it instead of inventing a number.

Every retrieval upgrade is a separate branch and a separate eval run, kept only if the numbers justify it. The result is recorded here either way, because a negative result is evidence about the method rather than a failure to hide.

| Technique | Hit rate @k | MRR | Latency | Verdict |
| --- | --- | --- | --- | --- |
| v1 dense baseline | | | | baseline |

No upgrade has been measured yet. The first rows land once the corpus rebuild is indexed and the harness is re-run against it.

**Attribution:** This is an unofficial fan-made project and isn't endorsed by Riot Games. All data is sourced from Riot's first-party JSON APIs: Data Dragon, Community Dragon, and the Riot Universe. League of Legends is a trademark of Riot Games, Inc.