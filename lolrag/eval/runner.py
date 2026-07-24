import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, Field

from lolrag import pipeline
from lolrag.config import Settings
from lolrag.eval.dataset import GoldenDataset, load_golden_dataset
from lolrag.eval.judge import judge_faithfulness
from lolrag.eval.metrics import hit_at_k, reciprocal_rank
from lolrag.indexing import get_vector_store

logger = logging.getLogger(__name__)


class QuestionResult(BaseModel):
    """Per-question evaluation outcome for a single golden question.

    Args:
        id: Golden question identifier.
        question: The evaluated question text.
        category: Golden question category, factual or refusal.
        expected_champion_ids: Champion ids expected to be retrieved.
        retrieved_champion_ids: Champion ids actually retrieved, in rank order.
        hit_at_1: Whether a relevant id was retrieved at rank 1.
        hit_at_3: Whether a relevant id was retrieved within the first 3.
        hit_at_k: Whether a relevant id was retrieved within the first k.
        reciprocal_rank: Reciprocal rank of the first relevant retrieved id.
        faithfulness_score: Judge groundedness score from 1 to 5.
        faithfulness_reasoning: Judge justification for the faithfulness score.
        latency_seconds: Wall-clock seconds for retrieval plus generation.
    """

    id: str = Field(description="Golden question identifier.")
    question: str = Field(description="The evaluated question text.")
    category: str = Field(description="Golden question category.")
    expected_champion_ids: list[str] = Field(description="Champion ids expected to be retrieved.")
    retrieved_champion_ids: list[str] = Field(
        description="Champion ids actually retrieved, in rank order."
    )
    hit_at_1: bool = Field(description="Whether a relevant id was retrieved at rank 1.")
    hit_at_3: bool = Field(description="Whether a relevant id was retrieved within the first 3.")
    hit_at_k: bool = Field(description="Whether a relevant id was retrieved within the first k.")
    reciprocal_rank: float = Field(description="Reciprocal rank of the first relevant id.")
    faithfulness_score: int = Field(description="Judge groundedness score from 1 to 5.")
    faithfulness_reasoning: str = Field(description="Judge justification for the score.")
    latency_seconds: float = Field(description="Seconds for retrieval plus generation.")


class EvalReport(BaseModel):
    """Aggregate evaluation report across the whole golden dataset.

    Args:
        dataset_version: Version of the golden dataset evaluated.
        num_questions: Number of questions evaluated.
        k: Retriever cutoff used for hit_at_k and the report.
        hit_rate_at_1: Mean hit_at_1 across questions.
        hit_rate_at_3: Mean hit_at_3 across questions.
        hit_rate_at_k: Mean hit_at_k across questions.
        mrr: Mean reciprocal rank across questions.
        mean_faithfulness: Mean faithfulness score across questions.
        mean_latency_seconds: Mean per-question latency in seconds.
        timestamp: UTC ISO-8601 timestamp of when the run completed.
        results: Per-question results.
    """

    dataset_version: int = Field(description="Version of the golden dataset evaluated.")
    num_questions: int = Field(description="Number of questions evaluated.")
    k: int = Field(description="Retriever cutoff used for hit_at_k.")
    hit_rate_at_1: float = Field(description="Mean hit_at_1 across questions.")
    hit_rate_at_3: float = Field(description="Mean hit_at_3 across questions.")
    hit_rate_at_k: float = Field(description="Mean hit_at_k across questions.")
    mrr: float = Field(description="Mean reciprocal rank across questions.")
    mean_faithfulness: float = Field(description="Mean faithfulness score across questions.")
    mean_latency_seconds: float = Field(description="Mean per-question latency in seconds.")
    timestamp: str = Field(description="UTC ISO-8601 timestamp of the run.")
    results: list[QuestionResult] = Field(description="Per-question results.")


# ---------- tracing ----------


def _enable_langsmith(settings: Settings) -> None:
    """Copy LangSmith settings into the environment so auto-tracing activates.

    Args:
        settings: Application settings providing langsmith_tracing,
            langsmith_api_key, langsmith_project.

    Returns:
        None. Sets LANGSMITH_* environment variables when tracing is configured,
        otherwise leaves the environment unchanged.
    """
    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        logger.info("LangSmith tracing enabled for project %s", settings.langsmith_project)
    else:
        logger.info("LangSmith tracing disabled")


# ---------- preconditions ----------


def _ensure_index_populated(settings: Settings) -> None:
    """Verify the vector store collection contains documents before evaluating.

    Args:
        settings: Application settings providing vector store configuration.

    Raises:
        RuntimeError: If the collection is empty, instructing the caller to
            ingest the corpus first.
    """
    vector_store = get_vector_store(settings)
    count = vector_store._collection.count()
    if count == 0:
        raise RuntimeError(
            "The vector store is empty. Run 'uv run python -m lolrag ingest' to build "
            "the index before running the evaluation."
        )


# ---------- evaluation ----------


def run_evaluation(settings: Settings, dataset: GoldenDataset | None = None) -> EvalReport:
    """Run the full evaluation harness over the golden dataset.

    Args:
        settings: Application settings for retrieval, generation, judging and
            tracing.
        dataset: Golden dataset to evaluate, or None to load the packaged one.

    Returns:
        EvalReport aggregating retrieval and faithfulness metrics over every
        golden question.

    Raises:
        RuntimeError: If the vector store is empty.
    """
    if dataset is None:
        dataset = load_golden_dataset()
    _enable_langsmith(settings)
    _ensure_index_populated(settings)

    k = settings.retriever_k
    results: list[QuestionResult] = []
    for question in dataset.questions:
        relevant_ids = set(question.expected_champion_ids)
        start = time.perf_counter()
        documents = pipeline.retrieve(question.question, settings)
        answer = pipeline.generate(question.question, documents, settings)
        latency_seconds = time.perf_counter() - start

        retrieved_champion_ids = [doc.metadata["champion_id"] for doc in documents]
        context = pipeline.format_context(documents)
        verdict = judge_faithfulness(question.question, context, answer, settings)

        results.append(
            QuestionResult(
                id=question.id,
                question=question.question,
                category=question.category,
                expected_champion_ids=question.expected_champion_ids,
                retrieved_champion_ids=retrieved_champion_ids,
                hit_at_1=hit_at_k(retrieved_champion_ids, relevant_ids, 1),
                hit_at_3=hit_at_k(retrieved_champion_ids, relevant_ids, 3),
                hit_at_k=hit_at_k(retrieved_champion_ids, relevant_ids, k),
                reciprocal_rank=reciprocal_rank(retrieved_champion_ids, relevant_ids),
                faithfulness_score=verdict.score,
                faithfulness_reasoning=verdict.reasoning,
                latency_seconds=latency_seconds,
            )
        )

    return EvalReport(
        dataset_version=dataset.version,
        num_questions=len(results),
        k=k,
        hit_rate_at_1=mean(r.hit_at_1 for r in results) if results else 0.0,
        hit_rate_at_3=mean(r.hit_at_3 for r in results) if results else 0.0,
        hit_rate_at_k=mean(r.hit_at_k for r in results) if results else 0.0,
        mrr=mean(r.reciprocal_rank for r in results) if results else 0.0,
        mean_faithfulness=mean(r.faithfulness_score for r in results) if results else 0.0,
        mean_latency_seconds=mean(r.latency_seconds for r in results) if results else 0.0,
        timestamp=datetime.now(UTC).isoformat(),
        results=results,
    )


# ---------- reporting ----------


def _render_markdown(report: EvalReport) -> str:
    """Render an evaluation report as a human-readable Markdown summary.

    Args:
        report: The evaluation report to render.

    Returns:
        Markdown text with an ablation-style summary row and a per-question table.
    """
    lines = [
        "# Evaluation report",
        "",
        f"- Dataset version: {report.dataset_version}",
        f"- Questions: {report.num_questions}",
        f"- k: {report.k}",
        f"- Timestamp: {report.timestamp}",
        "",
        "## Ablation summary",
        "",
        "| Technique | hit@1 | hit@3 | hit@k | MRR | faithfulness | mean latency (s) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        (
            f"| v1 naive baseline | {report.hit_rate_at_1:.3f} | {report.hit_rate_at_3:.3f} | "
            f"{report.hit_rate_at_k:.3f} | {report.mrr:.3f} | {report.mean_faithfulness:.3f} | "
            f"{report.mean_latency_seconds:.3f} |"
        ),
        "",
        "## Per-question results",
        "",
        (
            "| id | category | expected | retrieved | hit@1 | hit@3 | hit@k | RR | "
            "faithfulness | latency (s) |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in report.results:
        expected = ", ".join(r.expected_champion_ids)
        retrieved = ", ".join(r.retrieved_champion_ids)
        lines.append(
            f"| {r.id} | {r.category} | {expected} | {retrieved} | {r.hit_at_1} | "
            f"{r.hit_at_3} | {r.hit_at_k} | {r.reciprocal_rank:.3f} | "
            f"{r.faithfulness_score} | {r.latency_seconds:.3f} |"
        )
    return "\n".join(lines) + "\n"


def write_report(report: EvalReport, report_dir: str) -> tuple[Path, Path]:
    """Write an evaluation report as JSON and Markdown into a directory.

    Args:
        report: The evaluation report to persist.
        report_dir: Directory to write latest.json and latest.md into; created
            with parents if it does not exist.

    Returns:
        Tuple of the JSON path and the Markdown path written.
    """
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "latest.json"
    markdown_path = directory / "latest.md"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path, markdown_path
