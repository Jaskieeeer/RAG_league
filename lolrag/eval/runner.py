"""Runs the golden dataset through two systems and reports the result stratified.

Two things make this more than a loop over questions.

First, the scoring mode decides which figure a question contributes to. Retrieval
metrics are computed over the retrieval block alone: a refusal question has no
correct document, so counting it as a retrieval miss would drag the headline
number down for a reason that has nothing to do with the retriever. Refusal
accuracy is its own figure and the limitation block is reported unscored.

Second, there are two systems. The pipeline retrieves and answers; the baseline
answers from pretraining with everything else held constant. Only the retrieval
block is run against both, because declining is correct behaviour for a grounded
system and merely unhelpful for a general one, so comparing them on refusals
would compare two different notions of success.
"""

import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from lolrag import pipeline
from lolrag.config import Settings
from lolrag.db.session import get_session
from lolrag.eval.agreement import (
    SAMPLE_FILENAME,
    AgreementSummary,
    EntailmentSample,
    compute_agreement,
    dump_entailment_sample,
    load_entailment_sample,
    select_sample,
)
from lolrag.eval.baseline import generate_without_context
from lolrag.eval.dataset import (
    CHECK_NUMERIC,
    SCORING_LIMITATION,
    SCORING_REFUSAL,
    SCORING_RETRIEVAL,
    GoldenDataset,
    GoldenQuestion,
    load_golden_dataset,
)
from lolrag.eval.judge import (
    ENTAILMENT_SCORES,
    EntailmentVerdict,
    judge_entailment,
    judge_faithfulness,
    judge_refusal,
)
from lolrag.eval.metrics import hit_at_k, reciprocal_rank
from lolrag.eval.numeric import NumericScore, score_numeric

logger = logging.getLogger(__name__)

SYSTEM_PIPELINE = "pipeline"
SYSTEM_BASELINE = "baseline"


class QuestionResult(BaseModel):
    """One system's outcome on one golden question.

    Args:
        system: System under test, pipeline or baseline.
        id: Golden question identifier.
        question: The evaluated question text.
        category: Golden question category, the unit of stratification.
        scoring: Scoring mode of the question, retrieval, refusal or limitation.
        check: How the answer was graded, numeric, entailment, refusal or none.
        answer: The answer the system produced.
        expected_doc_keys: Document keys expected to be retrieved, empty outside
            the retrieval block.
        retrieved_doc_keys: Document keys actually retrieved, in rank order, one
            per retrieved chunk, so a document matched by two chunks appears
            twice. Always empty for the baseline, which retrieves nothing.
        hit_at_1: Whether a relevant key was retrieved at rank 1, or None where
            retrieval is not scored.
        hit_at_3: Whether a relevant key was retrieved within the first 3, or
            None where retrieval is not scored.
        hit_at_k: Whether a relevant key was retrieved within the first k, or
            None where retrieval is not scored.
        reciprocal_rank: Reciprocal rank of the first relevant retrieved key, or
            None where retrieval is not scored.
        answer_score: Answer quality on a 0 to 1 scale, comparable across the
            numeric, entailment and refusal checks, or None for unscored
            limitation questions.
        numeric: Deterministic numeric grading, or None when the check was not
            numeric.
        entailment: Judge verdict against the reference answer, or None when the
            check was not entailment.
        declined: Whether the answer declined, or None outside the refusal block.
        faithfulness_score: Judge groundedness score from 1 to 5, or None for
            the baseline, which has no context to be grounded in.
        faithfulness_reasoning: Judge justification for the faithfulness score.
        latency_seconds: Wall-clock seconds for retrieval plus generation,
            excluding every judge call.
    """

    system: str = Field(description="System under test, pipeline or baseline.")
    id: str = Field(description="Golden question identifier.")
    question: str = Field(description="The evaluated question text.")
    category: str = Field(description="Golden question category.")
    scoring: str = Field(description="Scoring mode of the question.")
    check: str = Field(description="How the answer was graded.")
    answer: str = Field(description="The answer the system produced.")
    expected_doc_keys: list[str] = Field(description="Document keys expected to be retrieved.")
    retrieved_doc_keys: list[str] = Field(
        description="Document keys actually retrieved, in rank order."
    )
    hit_at_1: bool | None = Field(description="Whether a relevant key was retrieved at rank 1.")
    hit_at_3: bool | None = Field(description="Whether a relevant key was in the first 3.")
    hit_at_k: bool | None = Field(description="Whether a relevant key was in the first k.")
    reciprocal_rank: float | None = Field(description="Reciprocal rank of the first relevant key.")
    answer_score: float | None = Field(description="Answer quality from 0 to 1.")
    numeric: NumericScore | None = Field(default=None, description="Deterministic numeric grading.")
    entailment: EntailmentVerdict | None = Field(
        default=None, description="Judge verdict against the reference answer."
    )
    declined: bool | None = Field(default=None, description="Whether the answer declined.")
    faithfulness_score: int | None = Field(
        default=None, description="Judge groundedness score from 1 to 5."
    )
    faithfulness_reasoning: str | None = Field(
        default=None, description="Judge justification for the faithfulness score."
    )
    latency_seconds: float = Field(description="Seconds for retrieval plus generation.")


class RetrievalMetrics(BaseModel):
    """Retrieval quality over the retrieval block alone.

    Args:
        num_questions: Questions the metrics were computed over.
        hit_rate_at_1: Mean hit_at_1.
        hit_rate_at_3: Mean hit_at_3.
        hit_rate_at_k: Mean hit_at_k.
        mrr: Mean reciprocal rank.
    """

    num_questions: int = Field(description="Questions the metrics were computed over.")
    hit_rate_at_1: float = Field(description="Mean hit_at_1.")
    hit_rate_at_3: float = Field(description="Mean hit_at_3.")
    hit_rate_at_k: float = Field(description="Mean hit_at_k.")
    mrr: float = Field(description="Mean reciprocal rank.")


class AnswerMetrics(BaseModel):
    """Answer quality over the retrieval block, split by how it was checked.

    Args:
        num_questions: Retrieval-block questions scored.
        mean_answer_score: Mean answer_score across them, the single comparable
            figure.
        num_numeric: Questions graded by the deterministic numeric check.
        numeric_pass_rate: Fraction of those whose every required number
            appeared.
        mean_numeric_coverage: Mean fraction of required numeric tokens matched,
            a partial-credit view of the same block.
        num_entailment: Questions graded by the entailment judge.
        mean_entailment_score: Mean of 1.0 entails, 0.5 partial, 0.0 misses.
        entails: Count of entails verdicts.
        partial: Count of partial verdicts.
        misses: Count of misses verdicts.
    """

    num_questions: int = Field(description="Retrieval-block questions scored.")
    mean_answer_score: float = Field(description="Mean answer score from 0 to 1.")
    num_numeric: int = Field(description="Questions graded numerically.")
    numeric_pass_rate: float = Field(description="Fraction of numeric questions fully matched.")
    mean_numeric_coverage: float = Field(description="Mean fraction of required numbers matched.")
    num_entailment: int = Field(description="Questions graded by the entailment judge.")
    mean_entailment_score: float = Field(description="Mean entailment score from 0 to 1.")
    entails: int = Field(description="Count of entails verdicts.")
    partial: int = Field(description="Count of partial verdicts.")
    misses: int = Field(description="Count of misses verdicts.")


class SystemSummary(BaseModel):
    """Everything measured about one system.

    Args:
        system: System the summary describes.
        retrieval: Retrieval metrics, or None for a system that retrieves
            nothing.
        answers: Answer metrics over the retrieval block.
        mean_faithfulness: Mean groundedness over every question the system
            answered with context, or None for a system with no context.
        mean_latency_seconds: Mean retrieval plus generation latency.
    """

    system: str = Field(description="System the summary describes.")
    retrieval: RetrievalMetrics | None = Field(description="Retrieval metrics, None if it cannot.")
    answers: AnswerMetrics = Field(description="Answer metrics over the retrieval block.")
    mean_faithfulness: float | None = Field(description="Mean groundedness, None without context.")
    mean_latency_seconds: float = Field(description="Mean retrieval plus generation latency.")


class RefusalSummary(BaseModel):
    """The refusal block, scored on one axis and never against the baseline.

    Args:
        num_questions: Refusal questions asked.
        refusal_accuracy: Fraction of them the pipeline declined on.
        asserted_ids: Ids of the questions it answered instead of declining.
        mean_faithfulness: Mean groundedness over the block.
        mean_latency_seconds: Mean retrieval plus generation latency.
    """

    num_questions: int = Field(description="Refusal questions asked.")
    refusal_accuracy: float = Field(description="Fraction of them declined.")
    asserted_ids: list[str] = Field(description="Ids answered instead of declined.")
    mean_faithfulness: float | None = Field(description="Mean groundedness over the block.")
    mean_latency_seconds: float = Field(description="Mean latency over the block.")


class LimitationSummary(BaseModel):
    """The limitation block, reported on its own line and scored nowhere.

    Args:
        num_questions: Limitation questions asked.
        mean_faithfulness: Mean groundedness over the block, which says whether
            the pipeline at least stayed inside its context while failing.
        mean_latency_seconds: Mean retrieval plus generation latency.
    """

    num_questions: int = Field(description="Limitation questions asked.")
    mean_faithfulness: float | None = Field(description="Mean groundedness over the block.")
    mean_latency_seconds: float = Field(description="Mean latency over the block.")


class CategoryRow(BaseModel):
    """One stratum of the report, aggregating a single authoring category.

    Args:
        category: The category aggregated.
        scoring: Scoring mode shared by its questions.
        checks: The checks used within it, joined by "+" where a category mixes
            two.
        num_questions: Questions in the category.
        pipeline_score: Mean pipeline answer score, which is refusal accuracy in
            the refusal block, or None where nothing is scored.
        baseline_score: Mean baseline answer score, or None where the baseline
            does not run.
        delta: pipeline_score minus baseline_score, or None where either is
            absent.
        hit_rate_at_k: Pipeline hit rate at k, or None outside the retrieval
            block.
        mrr: Pipeline mean reciprocal rank, or None outside the retrieval block.
    """

    category: str = Field(description="The category aggregated.")
    scoring: str = Field(description="Scoring mode shared by its questions.")
    checks: str = Field(description="Checks used within the category.")
    num_questions: int = Field(description="Questions in the category.")
    pipeline_score: float | None = Field(description="Mean pipeline answer score.")
    baseline_score: float | None = Field(description="Mean baseline answer score.")
    delta: float | None = Field(description="Pipeline score minus baseline score.")
    hit_rate_at_k: float | None = Field(description="Pipeline hit rate at k.")
    mrr: float | None = Field(description="Pipeline mean reciprocal rank.")


class EvalReport(BaseModel):
    """The full result of one evaluation run over both systems.

    Args:
        dataset_version: Version of the golden dataset evaluated.
        dataset_description: The dataset's own description of itself.
        num_questions: Golden questions in the dataset evaluated.
        num_retrieval: Questions in the retrieval block.
        num_refusal: Questions in the refusal block.
        num_limitation: Questions in the limitation block.
        k: Retriever cutoff used for hit_at_k.
        generation_model_name: Model both systems generated their answers with,
            which is eval_model_name and not the product's llm_model_name. It is
            recorded because a report whose numbers came off a different model is
            not comparable with this one.
        judge_model_name: Model the judges ran on.
        pipeline: Summary of the retrieving system.
        baseline: Summary of the no-retrieval control.
        refusal: The refusal block, pipeline only.
        limitation: The limitation block, unscored.
        categories: Per-category stratification, the row order of the dataset.
        judge_agreement: Judge-human agreement over hand-scored entailment
            judgements, or None when none have been scored yet.
        entailment_sample: The entailment judgements drawn for hand-scoring,
            written out beside the report by write_report.
        timestamp: UTC ISO-8601 timestamp of when the run completed.
        results: Per-question results for both systems.
    """

    dataset_version: int = Field(description="Version of the golden dataset evaluated.")
    dataset_description: str = Field(description="The dataset's description of itself.")
    num_questions: int = Field(description="Golden questions evaluated.")
    num_retrieval: int = Field(description="Questions in the retrieval block.")
    num_refusal: int = Field(description="Questions in the refusal block.")
    num_limitation: int = Field(description="Questions in the limitation block.")
    k: int = Field(description="Retriever cutoff used for hit_at_k.")
    generation_model_name: str = Field(
        description="Model both systems generated their answers with."
    )
    judge_model_name: str = Field(description="Model the judges ran on.")
    pipeline: SystemSummary = Field(description="Summary of the retrieving system.")
    baseline: SystemSummary = Field(description="Summary of the no-retrieval control.")
    refusal: RefusalSummary = Field(description="The refusal block, pipeline only.")
    limitation: LimitationSummary = Field(description="The limitation block, unscored.")
    categories: list[CategoryRow] = Field(description="Per-category stratification.")
    judge_agreement: AgreementSummary | None = Field(
        default=None, description="Judge-human agreement, None until judgements are hand-scored."
    )
    entailment_sample: list[EntailmentSample] = Field(
        default_factory=list, description="Entailment judgements drawn for hand-scoring."
    )
    timestamp: str = Field(description="UTC ISO-8601 timestamp of the run.")
    results: list[QuestionResult] = Field(description="Per-question results for both systems.")


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


# ---------- per-question evaluation ----------


def _generation_settings(settings: Settings) -> Settings:
    """Derive the settings both systems generate their answers with.

    Evaluation generates with eval_model_name rather than the product's
    llm_model_name, so a run can be priced and rate-limited without changing what
    lolrag ask answers with. The substitution is made once, here, and handed to
    both systems: two systems generating with different models would measure the
    models rather than the retrieval, which is the one thing the comparison
    exists to isolate.

    Args:
        settings: Application settings carrying llm_model_name for the product
            and eval_model_name for the harness.

    Returns:
        A copy of settings whose llm_model_name and llm_fallback_model_name are
        both eval_model_name. The fallback is substituted too because a fallback
        left pointing at the product's model would let one failed call answer
        with a model the report's generation_model_name does not name, which is
        the one thing that field exists to guarantee. Every other field is
        untouched, so retrieval and the judges keep reading the values they
        already read.
    """
    return settings.model_copy(
        update={
            "llm_model_name": settings.eval_model_name,
            "llm_fallback_model_name": settings.eval_model_name,
        }
    )


def _grade_answer(
    question: GoldenQuestion, answer: str, settings: Settings
) -> tuple[float | None, NumericScore | None, EntailmentVerdict | None, bool | None]:
    """Grade one answer by the check its question declares.

    Args:
        question: The golden question, whose check field selects the path and
            whose expected_answer is the only reference used.
        answer: The answer to grade.
        settings: Application settings providing eval_judge_model_name and
            google_api_key for the judged paths.

    Returns:
        Tuple of the 0 to 1 answer score, the numeric grading, the entailment
        verdict and the refusal flag, all but the score being None on the paths
        that do not produce them. The numeric path calls no model at all.
    """
    if question.scoring == SCORING_LIMITATION:
        return None, None, None, None
    if question.scoring == SCORING_REFUSAL:
        verdict = judge_refusal(question.question, answer, settings)
        return float(verdict.declined), None, None, verdict.declined
    if question.check == CHECK_NUMERIC:
        numeric = score_numeric(question.expected_answer, answer)
        return float(numeric.passed), numeric, None, None
    entailment = judge_entailment(question.question, question.expected_answer, answer, settings)
    return ENTAILMENT_SCORES[entailment.verdict], None, entailment, None


def evaluate_pipeline_question(
    question: GoldenQuestion, session: Session, settings: Settings
) -> QuestionResult:
    """Run one golden question through the retrieving pipeline and score it.

    Args:
        question: The golden question to run.
        session: Open Session the retrieval query runs through.
        settings: Application settings for retrieval, generation and judging.
            Generation reads eval_model_name, not llm_model_name.

    Returns:
        QuestionResult carrying retrieval metrics for retrieval-block questions,
        an answer score for every scored question, and faithfulness for all of
        them, since every pipeline answer has a context to be grounded in.
    """
    generation = _generation_settings(settings)

    start = time.perf_counter()
    documents = pipeline.retrieve(question.question, session, settings)
    answer = pipeline.generate(question.question, documents, generation)
    latency_seconds = time.perf_counter() - start

    retrieved_doc_keys = [document.metadata["doc_key"] for document in documents]
    faithfulness = judge_faithfulness(
        question.question, pipeline.format_context(documents), answer, settings
    )
    score, numeric, entailment, declined = _grade_answer(question, answer, settings)

    scores_retrieval = question.scoring == SCORING_RETRIEVAL
    relevant = set(question.expected_doc_keys)
    return QuestionResult(
        system=SYSTEM_PIPELINE,
        id=question.id,
        question=question.question,
        category=question.category,
        scoring=question.scoring,
        check=question.check,
        answer=answer,
        expected_doc_keys=question.expected_doc_keys,
        retrieved_doc_keys=retrieved_doc_keys,
        hit_at_1=hit_at_k(retrieved_doc_keys, relevant, 1) if scores_retrieval else None,
        hit_at_3=hit_at_k(retrieved_doc_keys, relevant, 3) if scores_retrieval else None,
        hit_at_k=(
            hit_at_k(retrieved_doc_keys, relevant, settings.retriever_k)
            if scores_retrieval
            else None
        ),
        reciprocal_rank=reciprocal_rank(retrieved_doc_keys, relevant) if scores_retrieval else None,
        answer_score=score,
        numeric=numeric,
        entailment=entailment,
        declined=declined,
        faithfulness_score=faithfulness.score,
        faithfulness_reasoning=faithfulness.reasoning,
        latency_seconds=latency_seconds,
    )


def evaluate_baseline_question(question: GoldenQuestion, settings: Settings) -> QuestionResult:
    """Run one golden question through the no-retrieval control and score it.

    Args:
        question: The golden question to run, which must be a retrieval-block
            question; the control is not run on refusals or limitations.
        settings: Application settings for generation and judging. Generation
            reads eval_model_name, the same model the pipeline generated with.

    Returns:
        QuestionResult with no retrieval metrics and no faithfulness, both being
        undefined for a system that retrieves nothing, and an answer score from
        the same numeric or entailment path the pipeline was graded on.
    """
    generation = _generation_settings(settings)

    start = time.perf_counter()
    answer = generate_without_context(question.question, generation)
    latency_seconds = time.perf_counter() - start

    score, numeric, entailment, declined = _grade_answer(question, answer, settings)
    return QuestionResult(
        system=SYSTEM_BASELINE,
        id=question.id,
        question=question.question,
        category=question.category,
        scoring=question.scoring,
        check=question.check,
        answer=answer,
        expected_doc_keys=question.expected_doc_keys,
        retrieved_doc_keys=[],
        hit_at_1=None,
        hit_at_3=None,
        hit_at_k=None,
        reciprocal_rank=None,
        answer_score=score,
        numeric=numeric,
        entailment=entailment,
        declined=declined,
        faithfulness_score=None,
        faithfulness_reasoning=None,
        latency_seconds=latency_seconds,
    )


# ---------- aggregation ----------


def _mean(values: list[float]) -> float:
    """Average a list of numbers, treating the empty list as zero.

    Args:
        values: Numbers to average.

    Returns:
        The arithmetic mean, or 0.0 when there is nothing to average.
    """
    return mean(values) if values else 0.0


def _select(results: list[QuestionResult], system: str, scoring: str) -> list[QuestionResult]:
    """Select one system's results within one scoring block.

    Args:
        results: Every per-question result of a run.
        system: System to select, pipeline or baseline.
        scoring: Scoring mode to select.

    Returns:
        The matching results in run order.
    """
    return [result for result in results if result.system == system and result.scoring == scoring]


def _retrieval_metrics(results: list[QuestionResult]) -> RetrievalMetrics:
    """Aggregate retrieval quality over the retrieval block.

    Args:
        results: Retrieval-block results of one system, each carrying non-None
            hit flags.

    Returns:
        RetrievalMetrics over exactly those questions.
    """
    return RetrievalMetrics(
        num_questions=len(results),
        hit_rate_at_1=_mean([float(bool(result.hit_at_1)) for result in results]),
        hit_rate_at_3=_mean([float(bool(result.hit_at_3)) for result in results]),
        hit_rate_at_k=_mean([float(bool(result.hit_at_k)) for result in results]),
        mrr=_mean([result.reciprocal_rank or 0.0 for result in results]),
    )


def _answer_metrics(results: list[QuestionResult]) -> AnswerMetrics:
    """Aggregate answer quality over the retrieval block, split by check.

    Args:
        results: Retrieval-block results of one system.

    Returns:
        AnswerMetrics holding the comparable mean score plus the numeric and
        entailment blocks separately, since the two are measured by different
        instruments and averaging them alone would hide which one moved.
    """
    numeric = [result for result in results if result.numeric is not None]
    entailment = [result for result in results if result.entailment is not None]
    verdicts = [result.entailment.verdict for result in entailment if result.entailment]
    return AnswerMetrics(
        num_questions=len(results),
        mean_answer_score=_mean([result.answer_score or 0.0 for result in results]),
        num_numeric=len(numeric),
        numeric_pass_rate=_mean(
            [float(result.numeric.passed) for result in numeric if result.numeric]
        ),
        mean_numeric_coverage=_mean(
            [result.numeric.coverage for result in numeric if result.numeric]
        ),
        num_entailment=len(entailment),
        mean_entailment_score=_mean([ENTAILMENT_SCORES[verdict] for verdict in verdicts]),
        entails=verdicts.count("entails"),
        partial=verdicts.count("partial"),
        misses=verdicts.count("misses"),
    )


def _system_summary(
    system: str, results: list[QuestionResult], *, retrieves: bool
) -> SystemSummary:
    """Summarise one system across every block it ran.

    Args:
        system: System to summarise.
        results: Every per-question result of the run, both systems.
        retrieves: Whether the system retrieves, which decides whether retrieval
            metrics and faithfulness are defined for it at all.

    Returns:
        SystemSummary whose retrieval metrics and faithfulness are None for a
        system that retrieves nothing.
    """
    own = [result for result in results if result.system == system]
    retrieval_block = _select(results, system, SCORING_RETRIEVAL)
    faithfulness = [
        float(result.faithfulness_score) for result in own if result.faithfulness_score is not None
    ]
    return SystemSummary(
        system=system,
        retrieval=_retrieval_metrics(retrieval_block) if retrieves else None,
        answers=_answer_metrics(retrieval_block),
        mean_faithfulness=_mean(faithfulness) if faithfulness else None,
        mean_latency_seconds=_mean([result.latency_seconds for result in own]),
    )


def _refusal_summary(results: list[QuestionResult]) -> RefusalSummary:
    """Summarise the refusal block, which the pipeline alone runs.

    Args:
        results: Every per-question result of the run.

    Returns:
        RefusalSummary naming the questions the pipeline asserted on rather than
        declining, since those are the ones worth reading by hand.
    """
    block = _select(results, SYSTEM_PIPELINE, SCORING_REFUSAL)
    faithfulness = [
        float(result.faithfulness_score)
        for result in block
        if result.faithfulness_score is not None
    ]
    return RefusalSummary(
        num_questions=len(block),
        refusal_accuracy=_mean([float(bool(result.declined)) for result in block]),
        asserted_ids=[result.id for result in block if not result.declined],
        mean_faithfulness=_mean(faithfulness) if faithfulness else None,
        mean_latency_seconds=_mean([result.latency_seconds for result in block]),
    )


def _limitation_summary(results: list[QuestionResult]) -> LimitationSummary:
    """Summarise the limitation block, which is counted and never scored.

    Args:
        results: Every per-question result of the run.

    Returns:
        LimitationSummary over the pipeline's limitation questions.
    """
    block = _select(results, SYSTEM_PIPELINE, SCORING_LIMITATION)
    faithfulness = [
        float(result.faithfulness_score)
        for result in block
        if result.faithfulness_score is not None
    ]
    return LimitationSummary(
        num_questions=len(block),
        mean_faithfulness=_mean(faithfulness) if faithfulness else None,
        mean_latency_seconds=_mean([result.latency_seconds for result in block]),
    )


def _category_rows(dataset: GoldenDataset, results: list[QuestionResult]) -> list[CategoryRow]:
    """Stratify the run by authoring category.

    Args:
        dataset: The evaluated dataset, which fixes the row order.
        results: Every per-question result of the run.

    Returns:
        One CategoryRow per category, in first-appearance order. Aggregating
        these away is what would hide the finding: the control is expected to
        hold its own on lore prose and to collapse on ability numbers and item
        mode variants, and a single total shows neither.
    """
    order: list[str] = []
    for question in dataset.questions:
        if question.category not in order:
            order.append(question.category)

    rows: list[CategoryRow] = []
    for category in order:
        questions = [q for q in dataset.questions if q.category == category]
        scoring = questions[0].scoring
        checks = "+".join(sorted({question.check for question in questions}))
        in_category = [result for result in results if result.category == category]
        pipeline_results = [r for r in in_category if r.system == SYSTEM_PIPELINE]
        baseline_results = [r for r in in_category if r.system == SYSTEM_BASELINE]
        pipeline_scored = [r.answer_score for r in pipeline_results if r.answer_score is not None]
        baseline_scored = [r.answer_score for r in baseline_results if r.answer_score is not None]
        pipeline_score = _mean(pipeline_scored) if pipeline_scored else None
        baseline_score = _mean(baseline_scored) if baseline_scored else None
        retrieval_scored = [r for r in pipeline_results if r.hit_at_k is not None]
        rows.append(
            CategoryRow(
                category=category,
                scoring=scoring,
                checks=checks,
                num_questions=len(questions),
                pipeline_score=pipeline_score,
                baseline_score=baseline_score,
                delta=(
                    pipeline_score - baseline_score
                    if pipeline_score is not None and baseline_score is not None
                    else None
                ),
                hit_rate_at_k=(
                    _mean([float(bool(r.hit_at_k)) for r in retrieval_scored])
                    if retrieval_scored
                    else None
                ),
                mrr=(
                    _mean([r.reciprocal_rank or 0.0 for r in retrieval_scored])
                    if retrieval_scored
                    else None
                ),
            )
        )
    return rows


# ---------- judge-human agreement ----------


def entailment_samples(
    results: list[QuestionResult], dataset: GoldenDataset
) -> list[EntailmentSample]:
    """Collect every entailment judgement of a run as a hand-scorable record.

    Args:
        results: Every per-question result of the run.
        dataset: The evaluated dataset, read for each question's reference
            answer.

    Returns:
        One EntailmentSample per judged answer, across both systems.
    """
    references = {question.id: question.expected_answer for question in dataset.questions}
    return [
        EntailmentSample(
            id=result.id,
            system=result.system,
            question=result.question,
            reference=references.get(result.id, ""),
            answer=result.answer,
            judge_verdict=result.entailment.verdict,
        )
        for result in results
        if result.entailment is not None
    ]


def build_agreement_sample(
    results: list[QuestionResult], dataset: GoldenDataset, sample_size: int, seed: int = 0
) -> list[EntailmentSample]:
    """Draw the entailment judgements a human should hand-score.

    Args:
        results: Every per-question result of the run.
        dataset: The evaluated dataset, read for reference answers.
        sample_size: How many judgements to draw.
        seed: Seed for the deterministic draw.

    Returns:
        The drawn judgements, with human_verdict left null for hand-scoring.
    """
    return select_sample(entailment_samples(results, dataset), sample_size, seed)


# ---------- evaluation ----------


def run_evaluation(
    settings: Settings,
    dataset: GoldenDataset | None = None,
    human_scores_path: Path | None = None,
) -> EvalReport:
    """Run the golden dataset through the pipeline and the no-retrieval control.

    Args:
        settings: Application settings for retrieval, generation, judging and
            tracing.
        dataset: Golden dataset to evaluate, or None to load the packaged one.
            Pass a trimmed copy to smoke-test the harness without paying for the
            full run.
        human_scores_path: Path to a hand-scored entailment sample, or None. When
            it exists, its agreement with the judge is folded into the report.

    Returns:
        EvalReport holding both systems' per-question results and every
        aggregate, stratified by category.
    """
    if dataset is None:
        dataset = load_golden_dataset()
    _enable_langsmith(settings)

    retrieval_questions = dataset.by_scoring(SCORING_RETRIEVAL)
    results: list[QuestionResult] = []
    with get_session(settings) as session:
        for index, question in enumerate(dataset.questions, start=1):
            logger.info(
                "pipeline %d/%d %s [%s]",
                index,
                len(dataset.questions),
                question.id,
                question.scoring,
            )
            results.append(evaluate_pipeline_question(question, session, settings))

    for index, question in enumerate(retrieval_questions, start=1):
        logger.info("baseline %d/%d %s", index, len(retrieval_questions), question.id)
        results.append(evaluate_baseline_question(question, settings))

    agreement: AgreementSummary | None = None
    if human_scores_path is not None and human_scores_path.exists():
        agreement = compute_agreement(load_entailment_sample(human_scores_path))

    return EvalReport(
        dataset_version=dataset.version,
        dataset_description=dataset.description,
        num_questions=len(dataset.questions),
        num_retrieval=len(retrieval_questions),
        num_refusal=len(dataset.by_scoring(SCORING_REFUSAL)),
        num_limitation=len(dataset.by_scoring(SCORING_LIMITATION)),
        k=settings.retriever_k,
        generation_model_name=_generation_settings(settings).llm_model_name,
        judge_model_name=settings.eval_judge_model_name,
        pipeline=_system_summary(SYSTEM_PIPELINE, results, retrieves=True),
        baseline=_system_summary(SYSTEM_BASELINE, results, retrieves=False),
        refusal=_refusal_summary(results),
        limitation=_limitation_summary(results),
        categories=_category_rows(dataset, results),
        judge_agreement=agreement,
        entailment_sample=build_agreement_sample(
            results, dataset, settings.eval_agreement_sample_size
        ),
        timestamp=datetime.now(UTC).isoformat(),
        results=results,
    )


# ---------- reporting ----------


def _optional(value: float | None, digits: int = 3) -> str:
    """Render a number that may be undefined for the system or block at hand.

    Args:
        value: The number, or None where the figure does not apply.
        digits: Decimal places to render.

    Returns:
        The number rounded to digits places, or "-" when it is None, so an
        undefined figure never reads as a zero.
    """
    return "-" if value is None else f"{value:.{digits}f}"


def _headline_row(summary: SystemSummary) -> str:
    """Render one system's headline row of the ablation table.

    Args:
        summary: The system summary to render.

    Returns:
        A Markdown table row whose retrieval and faithfulness cells are dashes
        for a system that retrieves nothing.
    """
    retrieval = summary.retrieval
    return (
        f"| {summary.system} | {summary.answers.num_questions} | "
        f"{_optional(retrieval.hit_rate_at_1 if retrieval else None)} | "
        f"{_optional(retrieval.hit_rate_at_3 if retrieval else None)} | "
        f"{_optional(retrieval.hit_rate_at_k if retrieval else None)} | "
        f"{_optional(retrieval.mrr if retrieval else None)} | "
        f"{_optional(summary.answers.numeric_pass_rate)} | "
        f"{_optional(summary.answers.mean_entailment_score)} | "
        f"{_optional(summary.answers.mean_answer_score)} | "
        f"{_optional(summary.mean_faithfulness)} | "
        f"{_optional(summary.mean_latency_seconds)} |"
    )


def _render_markdown(report: EvalReport) -> str:
    """Render an evaluation report as a human-readable Markdown summary.

    Args:
        report: The evaluation report to render.

    Returns:
        Markdown text: a two-row headline table comparing the pipeline against
        the no-retrieval control, the refusal and limitation blocks on their own
        lines, the per-category stratification, judge agreement when it has been
        measured, and a per-question appendix.
    """
    lines = [
        "# Evaluation report",
        "",
        f"- Dataset version: {report.dataset_version}",
        (
            f"- Questions: {report.num_questions} "
            f"({report.num_retrieval} retrieval, {report.num_refusal} refusal, "
            f"{report.num_limitation} limitation)"
        ),
        f"- k: {report.k}",
        f"- Generation model, both systems: {report.generation_model_name}",
        f"- Judge model: {report.judge_model_name}",
        f"- Timestamp: {report.timestamp}",
        "",
        "## Headline, retrieval block only",
        "",
        (
            "| system | n | hit@1 | hit@3 | hit@k | MRR | numeric pass | entailment | "
            "answer score | faithfulness | latency (s) |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _headline_row(report.pipeline),
        _headline_row(report.baseline),
        "",
        (
            f"Numeric block: {report.pipeline.answers.num_numeric} questions, "
            f"pipeline coverage {report.pipeline.answers.mean_numeric_coverage:.3f}, "
            f"baseline coverage {report.baseline.answers.mean_numeric_coverage:.3f}."
        ),
        (
            f"Entailment block: {report.pipeline.answers.num_entailment} questions, "
            f"pipeline {report.pipeline.answers.entails}/{report.pipeline.answers.partial}/"
            f"{report.pipeline.answers.misses} entails/partial/misses, "
            f"baseline {report.baseline.answers.entails}/{report.baseline.answers.partial}/"
            f"{report.baseline.answers.misses}."
        ),
        "",
        "## Refusal block, pipeline only",
        "",
        "| questions | refusal accuracy | faithfulness | latency (s) | asserted instead |",
        "| ---: | ---: | ---: | ---: | --- |",
        (
            f"| {report.refusal.num_questions} | {report.refusal.refusal_accuracy:.3f} | "
            f"{_optional(report.refusal.mean_faithfulness)} | "
            f"{report.refusal.mean_latency_seconds:.3f} | "
            f"{', '.join(report.refusal.asserted_ids) or 'none'} |"
        ),
        "",
        "## Limitation block, unscored",
        "",
        "| questions | faithfulness | latency (s) |",
        "| ---: | ---: | ---: |",
        (
            f"| {report.limitation.num_questions} | "
            f"{_optional(report.limitation.mean_faithfulness)} | "
            f"{report.limitation.mean_latency_seconds:.3f} |"
        ),
        "",
        "## By category",
        "",
        "| category | scoring | checks | n | pipeline | baseline | delta | hit@k | MRR |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.categories:
        lines.append(
            f"| {row.category} | {row.scoring} | {row.checks} | {row.num_questions} | "
            f"{_optional(row.pipeline_score)} | {_optional(row.baseline_score)} | "
            f"{_optional(row.delta)} | {_optional(row.hit_rate_at_k)} | {_optional(row.mrr)} |"
        )

    lines.extend(["", "## Judge-human agreement", ""])
    if report.judge_agreement is None:
        lines.append(
            "Not measured. Hand-score the dumped entailment sample and rerun to fill this in."
        )
    else:
        agreement = report.judge_agreement
        lines.extend(
            [
                "| hand-scored | agreed | agreement | disagreements |",
                "| ---: | ---: | ---: | --- |",
                (
                    f"| {agreement.num_hand_scored} | {agreement.num_agreed} | "
                    f"{agreement.agreement_rate:.3f} | "
                    f"{', '.join(agreement.disagreement_ids) or 'none'} |"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Per-question results",
            "",
            "| system | id | category | check | score | hit@k | RR | faithfulness | latency (s) |",
            "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for result in report.results:
        score = result.faithfulness_score
        lines.append(
            f"| {result.system} | {result.id} | {result.category} | {result.check} | "
            f"{_optional(result.answer_score)} | "
            f"{'-' if result.hit_at_k is None else result.hit_at_k} | "
            f"{_optional(result.reciprocal_rank)} | "
            f"{_optional(float(score) if score is not None else None, 1)} | "
            f"{result.latency_seconds:.3f} |"
        )
    return "\n".join(lines) + "\n"


def write_report(report: EvalReport, report_dir: str) -> tuple[Path, Path, Path]:
    """Write an evaluation report as JSON and Markdown, with its judgement sample.

    Args:
        report: The evaluation report to persist.
        report_dir: Directory to write latest.json, latest.md and
            entailment_sample.json into; created with parents if it does not
            exist.

    Returns:
        Tuple of the JSON path, the Markdown path and the sample path written.
        The sample is written every run, always to entailment_sample.json and
        never to the entailment_scored.json the hand scores live in, so a rerun
        cannot overwrite work someone did by hand.
    """
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "latest.json"
    markdown_path = directory / "latest.md"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    sample_path = dump_entailment_sample(report.entailment_sample, directory / SAMPLE_FILENAME)
    return json_path, markdown_path, sample_path
