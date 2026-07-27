# LoL Knowledge RAG

[![CI](https://github.com/Jaskieeeer/RAG_league/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/Jaskieeeer/RAG_league/actions/workflows/ci.yml)

A retrieval-augmented generation system for answering League of Legends champion questions, built over Riot's first-party JSON APIs: Data Dragon, Community Dragon, and the Riot Universe.

**Status:** Phase 1 complete — the core RAG package (ingestion, indexing, retrieval, generation) works end to end as a naive v1 baseline. Next: evaluation harness, then a FastAPI backend.

**Stack:** Python 3.13, uv, LangChain 1.x, Gemini (`ChatGoogleGenerativeAI`), HuggingFace embeddings, Chroma. FastAPI and React planned for later phases.

## Setup

```
uv sync
```

Copy `.env.example` to `.env` and set `GOOGLE_API_KEY` (from [Google AI Studio](https://aistudio.google.com/apikey)). The remaining variables have working defaults.

## Usage

```
uv run python -m lolrag ingest
uv run python -m lolrag ask "What does Jinx's ultimate ability do?"
```

`ingest` fetches every champion for the pinned Data Dragon patch and builds a persistent Chroma index in `./data/chroma`. `ask` retrieves the most relevant champions and generates a grounded a
nswer with cited sources.

The corpus covers champion lore, roles, and ability descriptions as prose, plus ability cooldowns, costs, and ranges, which Data Dragon publishes as real values.

It does not yet include ability damage numbers or scaling ratios. The endpoints currently ingested ship these as unresolved placeholders: Data Dragon tooltips read `{{ qdamage }}` with empty `vars` and `datavalues` across all 692 spells, and Community Dragon's champion records repeat the gap as `@QDamage@` with zeroed `effectAmounts`.

The values do exist in Community Dragon's raw game data (`/game/data/characters/{champion}/{champion}.bin.json`), which exposes per-rank arrays such as Aatrox's `QBaseDamage` of 10/25/40/55/70 and `QTotalADRatio` of 60-90%. Ingesting them is a candidate for a later iteration; rendering a finished tooltip string additionally requires evaluating Riot's `GameCalculation` formula graph, so the cheaper first step is to expose the named per-rank values as structured facts.

Until then, questions about ability damage have no supporting context in the corpus, and the intended behaviour is to answer that the sources do not specify rather than to invent a value.

## Tests

```
uv run pytest
uv run pytest -m integration
```

The default run is fast and offline. The `integration` run needs network access and a configured `GOOGLE_API_KEY`.

**Attribution:** This is an unofficial fan-made project and isn't endorsed by Riot Games. All data is sourced from Riot's first-party JSON APIs: Data Dragon, Community Dragon, and the Riot Universe. League of Legends is a trademark of Riot Games, Inc.