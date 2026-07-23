# LoL Knowledge RAG

[![CI](https://github.com/Jaskieeeer/RAG_league/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/Jaskieeeer/RAG_league/actions/workflows/ci.yml)

A retrieval-augmented generation system for answering League of Legends champion questions, built over Riot's Data Dragon data and the LoL wiki.

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

The corpus currently covers champion lore, roles, and ability descriptions as prose. It does not include ability damage numbers, scaling ratios, or cooldowns: Data Dragon's tooltip variables have been unpopulated by Riot for years, so reliable numeric data will arrive with the planned LoL wiki ingestion.

## Tests

```
uv run pytest
uv run pytest -m integration
```

The default run is fast and offline. The `integration` run needs network access and a configured `GOOGLE_API_KEY`.

**Attribution:** This is an unofficial fan-made project and isn't endorsed by Riot Games. Champion data is sourced from Riot's Data Dragon and the League of Legends Wiki (CC BY-SA). League of
Legends is a trademark of Riot Games, Inc.