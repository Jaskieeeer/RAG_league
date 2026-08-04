from pathlib import Path

import pytest

from lolrag.config import get_settings
from lolrag.eval.agreement import SAMPLE_FILENAME, SCORED_FILENAME
from lolrag.eval.dataset import load_golden_dataset
from lolrag.eval.runner import (
    SYSTEM_BASELINE,
    SYSTEM_PIPELINE,
    run_evaluation,
    write_report,
)

pytestmark = pytest.mark.eval


def test_eval_harness_produces_valid_report():
    """Run the whole golden dataset through both systems and write the report.

    This is the expensive entry point: roughly 400 model calls for the pipeline
    and 180 for the no-retrieval control, so it is deselected from both gates and
    only ever run deliberately. It needs a GOOGLE_API_KEY and a database that has
    already been ingested.
    """
    settings = get_settings()
    if settings.google_api_key is None:
        pytest.skip("GOOGLE_API_KEY is not set; the eval harness needs a live LLM.")

    dataset = load_golden_dataset()
    report_dir = Path(settings.eval_report_dir)
    report = run_evaluation(settings, dataset, human_scores_path=report_dir / SCORED_FILENAME)

    assert report.num_questions == len(dataset.questions)
    assert len(report.results) == len(dataset.questions) + report.num_retrieval
    assert report.generation_model_name == settings.eval_model_name
    assert report.judge_model_name == settings.eval_judge_model_name

    retrieval_results = [
        result
        for result in report.results
        if result.system == SYSTEM_PIPELINE and result.scoring == "retrieval"
    ]
    assert len(retrieval_results) == report.num_retrieval
    for result in retrieval_results:
        assert isinstance(result.hit_at_k, bool)
        assert result.reciprocal_rank is not None
        assert 0.0 <= result.reciprocal_rank <= 1.0
        assert result.answer_score is not None

    assert report.pipeline.retrieval is not None
    assert 0.0 <= report.pipeline.retrieval.hit_rate_at_k <= 1.0
    assert 0.0 <= report.pipeline.retrieval.mrr <= 1.0
    assert report.pipeline.mean_faithfulness is not None
    assert 1.0 <= report.pipeline.mean_faithfulness <= 5.0

    assert report.baseline.retrieval is None
    assert report.baseline.mean_faithfulness is None
    baseline_results = [result for result in report.results if result.system == SYSTEM_BASELINE]
    assert {result.scoring for result in baseline_results} == {"retrieval"}

    assert 0.0 <= report.refusal.refusal_accuracy <= 1.0
    assert report.limitation.num_questions == report.num_limitation
    assert len(report.categories) == len({q.category for q in dataset.questions})

    assert len(report.entailment_sample) == min(
        settings.eval_agreement_sample_size, 2 * report.pipeline.answers.num_entailment
    )

    json_path, markdown_path, sample_path = write_report(report, str(report_dir))

    assert json_path.exists()
    assert markdown_path.exists()
    assert sample_path.exists()
    assert sample_path.name == SAMPLE_FILENAME
