import pytest

from lolrag.config import get_settings
from lolrag.eval.dataset import load_golden_dataset
from lolrag.eval.runner import (
    _ensure_index_populated,
    run_evaluation,
    write_report,
)

pytestmark = pytest.mark.eval


def test_eval_harness_produces_valid_report(tmp_path):
    settings = get_settings()
    if settings.google_api_key is None:
        pytest.skip("GOOGLE_API_KEY is not set; the eval harness needs a live LLM.")
    try:
        _ensure_index_populated(settings)
    except RuntimeError as error:
        pytest.skip(str(error))

    dataset = load_golden_dataset()
    report = run_evaluation(settings)

    assert len(report.results) == len(dataset.questions)
    for result in report.results:
        assert isinstance(result.hit_at_1, bool)
        assert isinstance(result.hit_at_3, bool)
        assert isinstance(result.hit_at_k, bool)
        assert 0.0 <= result.reciprocal_rank <= 1.0
        assert 1 <= result.faithfulness_score <= 5

    assert 0.0 <= report.hit_rate_at_1 <= 1.0
    assert 0.0 <= report.hit_rate_at_3 <= 1.0
    assert 0.0 <= report.hit_rate_at_k <= 1.0
    assert 0.0 <= report.mrr <= 1.0
    assert 1.0 <= report.mean_faithfulness <= 5.0

    json_path, markdown_path = write_report(report, str(tmp_path / "reports"))

    assert json_path.exists()
    assert markdown_path.exists()
