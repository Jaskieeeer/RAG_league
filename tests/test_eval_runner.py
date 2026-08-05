from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from langchain_core.documents import Document

from lolrag import pipeline
from lolrag.config import Settings, get_settings
from lolrag.eval import runner
from lolrag.eval.agreement import (
    SAMPLE_FILENAME,
    SCORED_FILENAME,
    EntailmentSample,
    compute_agreement,
    load_entailment_sample,
)
from lolrag.eval.dataset import GoldenDataset, GoldenQuestion
from lolrag.eval.judge import EntailmentVerdict, FaithfulnessVerdict, RefusalVerdict
from lolrag.eval.runner import (
    SYSTEM_BASELINE,
    SYSTEM_PIPELINE,
    EvalReport,
    build_agreement_sample,
    run_evaluation,
    write_report,
)

NUMERIC_REFERENCE = "120/100/80 seconds."
PROSE_REFERENCE = "Rocket Grab pulls the first enemy hit toward Blitzcrank and deals magic damage."

RIGHT_NUMERIC_ANSWER = "The cooldown is 120/100/80 seconds."
WRONG_NUMERIC_ANSWER = "The cooldown is 20/18/16 seconds."


def golden(**overrides: object) -> GoldenQuestion:
    """Build a golden question for the harness tests.

    Args:
        overrides: Fields to replace in the base question.

    Returns:
        A validated GoldenQuestion.
    """
    payload: dict[str, object] = {
        "id": "q-num",
        "question": "What is the cooldown of Darius's ultimate?",
        "category": "ability-numbers",
        "scoring": "retrieval",
        "check": "numeric",
        "expected_doc_keys": ["ability:darius:R"],
        "expected_answer": NUMERIC_REFERENCE,
        "collection": "abilities",
        "failure_mode": "numbers live in a tail chunk",
    }
    payload.update(overrides)
    return GoldenQuestion(**payload)


def dataset() -> GoldenDataset:
    """Build the four-question dataset the harness tests run over.

    Returns:
        A GoldenDataset holding one numeric, one entailment, one refusal and one
        limitation question, one of each scoring mode the runner branches on.
    """
    return GoldenDataset(
        description="fixture",
        version=2,
        questions=[
            golden(),
            golden(
                id="q-prose",
                question="What does Blitzcrank's hook do?",
                category="ability-mechanics",
                check="entailment",
                expected_doc_keys=["ability:blitzcrank:Q"],
                expected_answer=PROSE_REFERENCE,
            ),
            golden(
                id="q-refuse",
                question="Which champion has the highest win rate?",
                category="refusal-meta",
                scoring="refusal",
                check="refusal",
                expected_doc_keys=[],
                expected_answer="Should decline.",
                collection=None,
            ),
            golden(
                id="q-limit",
                question="Which champions are from Noxus?",
                category="list-aggregation",
                scoring="limitation",
                check="none",
                expected_doc_keys=[],
                expected_answer="Out of scope for v1.",
                collection=None,
            ),
        ],
    )


def chunk(doc_key: str) -> Document:
    """Build a retrieved chunk carrying the metadata the retriever produces.

    Args:
        doc_key: Deterministic key of the owning document.

    Returns:
        A Document shaped like one PgVectorRetriever returns.
    """
    return Document(
        page_content=f"content of {doc_key}",
        metadata={
            "doc_key": doc_key,
            "title": doc_key,
            "collection": "abilities",
            "source": doc_key,
            "chunk_index": 0,
            "distance": 0.1,
        },
    )


class Harness:
    """A fully mocked stand-in for retrieval, generation and every judge.

    Args:
        pipeline_answers: Answer the pipeline returns, keyed by question id.
        baseline_answers: Answer the baseline returns, keyed by question id.
        retrieved: Doc keys the retriever returns, keyed by question id.
        entailment: Verdict the entailment judge returns, keyed by question id.
        declined: Value the refusal judge returns, keyed by question id.
    """

    def __init__(
        self,
        pipeline_answers: dict[str, str],
        baseline_answers: dict[str, str],
        retrieved: dict[str, list[str]],
        entailment: dict[str, str],
        declined: dict[str, bool],
    ) -> None:
        self.pipeline_answers = pipeline_answers
        self.baseline_answers = baseline_answers
        self.retrieved = retrieved
        self.entailment = entailment
        self.declined = declined
        self.questions_seen: list[str] = []
        self.entailment_calls: list[str] = []
        self.refusal_calls: list[str] = []
        self.faithfulness_calls: list[str] = []
        self.baseline_calls: list[str] = []
        self.pipeline_models: list[str] = []
        self.baseline_models: list[str] = []
        self.pipeline_fallbacks: list[str] = []
        self.baseline_fallbacks: list[str] = []

    def _id_of(self, question: str) -> str:
        """Map a question's text back to its golden id.

        Args:
            question: The question text a mocked callable was handed.

        Returns:
            The golden id of that question.
        """
        for candidate in dataset().questions:
            if candidate.question == question:
                return candidate.id
        raise KeyError(question)

    def retrieve(self, question: str, session: object, settings: Settings) -> list[Document]:
        """Return the fixture chunks for a question."""
        self.questions_seen.append(self._id_of(question))
        return [chunk(key) for key in self.retrieved[self._id_of(question)]]

    def generate(self, question: str, documents: list[Document], settings: Settings) -> str:
        """Return the fixture pipeline answer for a question, recording both its models."""
        self.pipeline_models.append(settings.llm_model_name)
        self.pipeline_fallbacks.append(settings.llm_fallback_model_name)
        return self.pipeline_answers[self._id_of(question)]

    def generate_without_context(self, question: str, settings: Settings) -> str:
        """Return the fixture baseline answer for a question, recording both its models."""
        self.baseline_calls.append(self._id_of(question))
        self.baseline_models.append(settings.llm_model_name)
        self.baseline_fallbacks.append(settings.llm_fallback_model_name)
        return self.baseline_answers[self._id_of(question)]

    def judge_faithfulness(
        self, question: str, context: str, answer: str, settings: Settings
    ) -> FaithfulnessVerdict:
        """Return a fixed groundedness verdict."""
        self.faithfulness_calls.append(self._id_of(question))
        return FaithfulnessVerdict(score=4, reasoning="fixture")

    def judge_entailment(
        self, question: str, reference: str, answer: str, settings: Settings
    ) -> EntailmentVerdict:
        """Return the fixture entailment verdict for a question."""
        self.entailment_calls.append(self._id_of(question))
        return EntailmentVerdict(
            verdict=self.entailment[self._id_of(question)], reasoning="fixture"
        )

    def judge_refusal(self, question: str, answer: str, settings: Settings) -> RefusalVerdict:
        """Return the fixture refusal classification for a question."""
        self.refusal_calls.append(self._id_of(question))
        return RefusalVerdict(declined=self.declined[self._id_of(question)], reasoning="fixture")


@contextmanager
def _null_session() -> Iterator[None]:
    """Yield nothing in place of a database session.

    Returns:
        A context manager yielding None, since every database call is mocked.
    """
    yield None


@pytest.fixture
def harness(monkeypatch) -> Harness:
    """Patch every model and database call the runner makes.

    Returns:
        The Harness whose fixture answers the runner will see; a perfect
        pipeline and a baseline that knows the prose and invents the numbers.
    """
    stub = Harness(
        pipeline_answers={
            "q-num": RIGHT_NUMERIC_ANSWER,
            "q-prose": "It pulls the first enemy hit toward Blitzcrank and deals magic damage.",
            "q-refuse": "The context does not carry win rates.",
            "q-limit": "The context lists only one champion.",
        },
        baseline_answers={
            "q-num": WRONG_NUMERIC_ANSWER,
            "q-prose": "Rocket Grab pulls an enemy in, deals magic damage and stuns them.",
        },
        retrieved={
            "q-num": ["ability:darius:R", "ability:garen:R"],
            "q-prose": ["ability:thresh:Q", "ability:blitzcrank:Q"],
            "q-refuse": ["champion:zed", "ability:zed:R"],
            "q-limit": ["champion:darius", "faction:noxus"],
        },
        entailment={"q-prose": "entails"},
        declined={"q-refuse": True},
    )
    monkeypatch.setattr(runner, "get_session", lambda settings: _null_session())
    monkeypatch.setattr(pipeline, "retrieve", stub.retrieve)
    monkeypatch.setattr(pipeline, "generate", stub.generate)
    monkeypatch.setattr(runner, "generate_without_context", stub.generate_without_context)
    monkeypatch.setattr(runner, "judge_faithfulness", stub.judge_faithfulness)
    monkeypatch.setattr(runner, "judge_entailment", stub.judge_entailment)
    monkeypatch.setattr(runner, "judge_refusal", stub.judge_refusal)
    return stub


@pytest.fixture
def report(harness: Harness) -> EvalReport:
    """Run the mocked harness over the fixture dataset.

    Returns:
        The EvalReport of a run with no live model behind it.
    """
    return run_evaluation(get_settings(), dataset())


# ---------- systems ----------


def test_the_pipeline_answers_every_question(report: EvalReport):
    answered = {result.id for result in report.results if result.system == SYSTEM_PIPELINE}

    assert answered == {"q-num", "q-prose", "q-refuse", "q-limit"}


def test_the_baseline_runs_only_on_the_retrieval_block(report: EvalReport, harness: Harness):
    answered = {result.id for result in report.results if result.system == SYSTEM_BASELINE}

    assert answered == {"q-num", "q-prose"}
    assert harness.baseline_calls == ["q-num", "q-prose"]


def test_the_baseline_retrieves_nothing_and_is_not_scored_on_retrieval(report: EvalReport):
    baseline = [result for result in report.results if result.system == SYSTEM_BASELINE]

    assert report.baseline.retrieval is None
    assert all(result.retrieved_doc_keys == [] for result in baseline)
    assert all(result.hit_at_k is None for result in baseline)
    assert all(result.reciprocal_rank is None for result in baseline)


def test_the_baseline_gets_no_faithfulness(report: EvalReport):
    baseline = [result for result in report.results if result.system == SYSTEM_BASELINE]

    assert report.baseline.mean_faithfulness is None
    assert all(result.faithfulness_score is None for result in baseline)


# ---------- generation model ----------


def test_both_systems_generate_with_the_same_model(report: EvalReport, harness: Harness):
    assert harness.pipeline_models
    assert harness.baseline_models
    assert set(harness.pipeline_models) == set(harness.baseline_models)
    assert len(set(harness.pipeline_models)) == 1


def test_generation_uses_the_eval_model_and_not_the_product_model(harness: Harness):
    settings = get_settings().model_copy(
        update={"llm_model_name": "product-model", "eval_model_name": "eval-model"}
    )

    run_evaluation(settings, dataset())

    assert set(harness.pipeline_models) == {"eval-model"}
    assert set(harness.baseline_models) == {"eval-model"}


def test_generation_never_falls_back_to_the_product_model(harness: Harness):
    settings = get_settings().model_copy(
        update={
            "llm_model_name": "product-model",
            "llm_fallback_model_name": "product-fallback",
            "eval_model_name": "eval-model",
        }
    )

    run_evaluation(settings, dataset())

    assert set(harness.pipeline_fallbacks) == {"eval-model"}
    assert set(harness.baseline_fallbacks) == {"eval-model"}


def test_the_report_names_the_models_that_produced_it(harness: Harness):
    settings = get_settings().model_copy(
        update={"eval_model_name": "eval-model", "eval_judge_model_name": "judge-model"}
    )

    report = run_evaluation(settings, dataset())

    assert report.generation_model_name == "eval-model"
    assert report.judge_model_name == "judge-model"


# ---------- scoring paths ----------


def test_the_numeric_path_calls_no_judge(harness: Harness, report: EvalReport):
    assert "q-num" not in harness.entailment_calls
    assert harness.entailment_calls == ["q-prose", "q-prose"]


def test_the_numeric_path_separates_a_right_answer_from_a_confident_wrong_one(
    report: EvalReport,
):
    scored = {result.system: result for result in report.results if result.id == "q-num"}

    assert scored[SYSTEM_PIPELINE].answer_score == 1.0
    assert scored[SYSTEM_PIPELINE].numeric is not None
    assert scored[SYSTEM_BASELINE].answer_score == 0.0
    assert scored[SYSTEM_BASELINE].numeric is not None
    assert scored[SYSTEM_BASELINE].numeric.missing == ["120/100/80"]


def test_the_entailment_path_scores_by_verdict(report: EvalReport):
    prose = next(
        result
        for result in report.results
        if result.id == "q-prose" and result.system == SYSTEM_PIPELINE
    )

    assert prose.entailment is not None
    assert prose.entailment.verdict == "entails"
    assert prose.answer_score == 1.0


def test_the_refusal_path_scores_only_whether_the_answer_declined(
    report: EvalReport, harness: Harness
):
    refusal = next(result for result in report.results if result.id == "q-refuse")

    assert harness.refusal_calls == ["q-refuse"]
    assert refusal.declined is True
    assert refusal.answer_score == 1.0
    assert report.refusal.refusal_accuracy == 1.0
    assert report.refusal.asserted_ids == []


def test_the_limitation_path_is_answered_but_never_scored(report: EvalReport):
    limitation = next(result for result in report.results if result.id == "q-limit")

    assert limitation.answer_score is None
    assert limitation.numeric is None
    assert limitation.entailment is None
    assert limitation.declined is None
    assert report.limitation.num_questions == 1


# ---------- metrics by scoring mode ----------


def test_retrieval_metrics_cover_the_retrieval_block_alone(report: EvalReport):
    assert report.pipeline.retrieval is not None
    assert report.pipeline.retrieval.num_questions == 2


def test_a_refusal_question_never_counts_as_a_retrieval_miss(report: EvalReport):
    assert report.pipeline.retrieval is not None
    assert report.pipeline.retrieval.hit_rate_at_k == 1.0
    assert report.pipeline.retrieval.hit_rate_at_1 == 0.5
    assert report.pipeline.retrieval.mrr == pytest.approx(0.75)


def test_refusal_and_limitation_questions_carry_no_retrieval_flags(report: EvalReport):
    unscored = [result for result in report.results if result.id in {"q-refuse", "q-limit"}]

    assert all(result.hit_at_k is None for result in unscored)
    assert all(result.reciprocal_rank is None for result in unscored)


def test_faithfulness_is_judged_for_every_pipeline_question(report: EvalReport, harness: Harness):
    assert sorted(harness.faithfulness_calls) == ["q-limit", "q-num", "q-prose", "q-refuse"]
    assert report.pipeline.mean_faithfulness == 4.0


def test_answer_metrics_split_the_numeric_and_entailment_blocks(report: EvalReport):
    assert report.pipeline.answers.num_numeric == 1
    assert report.pipeline.answers.numeric_pass_rate == 1.0
    assert report.pipeline.answers.num_entailment == 1
    assert report.pipeline.answers.entails == 1
    assert report.baseline.answers.numeric_pass_rate == 0.0
    assert report.baseline.answers.mean_answer_score == 0.5


def test_report_counts_every_block(report: EvalReport):
    assert report.num_questions == 4
    assert report.num_retrieval == 2
    assert report.num_refusal == 1
    assert report.num_limitation == 1
    assert len(report.results) == 6


# ---------- stratification ----------


def test_every_category_gets_its_own_row(report: EvalReport):
    categories = [row.category for row in report.categories]

    assert categories == [
        "ability-numbers",
        "ability-mechanics",
        "refusal-meta",
        "list-aggregation",
    ]


def test_a_category_row_compares_the_two_systems(report: EvalReport):
    numbers = next(row for row in report.categories if row.category == "ability-numbers")

    assert numbers.pipeline_score == 1.0
    assert numbers.baseline_score == 0.0
    assert numbers.delta == 1.0
    assert numbers.hit_rate_at_k == 1.0


def test_a_refusal_category_row_has_no_baseline_and_no_retrieval(report: EvalReport):
    refusal = next(row for row in report.categories if row.category == "refusal-meta")

    assert refusal.pipeline_score == 1.0
    assert refusal.baseline_score is None
    assert refusal.delta is None
    assert refusal.hit_rate_at_k is None


def test_a_limitation_category_row_is_scored_nowhere(report: EvalReport):
    limitation = next(row for row in report.categories if row.category == "list-aggregation")

    assert limitation.pipeline_score is None
    assert limitation.baseline_score is None
    assert limitation.num_questions == 1


# ---------- reporting ----------


def test_write_report_emits_json_markdown_and_the_judgement_sample(report: EvalReport, tmp_path):
    json_path, markdown_path, sample_path = write_report(report, str(tmp_path / "reports"))

    assert json_path.exists()
    assert markdown_path.exists()
    assert sample_path.name == SAMPLE_FILENAME
    assert EvalReport.model_validate_json(json_path.read_text(encoding="utf-8"))


def test_the_written_sample_is_the_report_sample_left_unscored(report: EvalReport, tmp_path):
    _, _, sample_path = write_report(report, str(tmp_path / "reports"))

    written = load_entailment_sample(sample_path)

    assert written == report.entailment_sample
    assert [sample.id for sample in written] == ["q-prose", "q-prose"]
    assert all(sample.human_verdict is None for sample in written)


def test_write_report_never_touches_the_hand_scored_file(report: EvalReport, tmp_path):
    directory = tmp_path / "reports"
    directory.mkdir()
    scored = directory / SCORED_FILENAME
    scored.write_text("[]", encoding="utf-8")

    write_report(report, str(directory))

    assert scored.read_text(encoding="utf-8") == "[]"


def test_the_markdown_shows_both_systems_and_every_category(report: EvalReport, tmp_path):
    _, markdown_path, _ = write_report(report, str(tmp_path / "reports"))

    markdown = markdown_path.read_text(encoding="utf-8")

    assert "| pipeline |" in markdown
    assert "| baseline |" in markdown
    assert "## By category" in markdown
    for row in report.categories:
        assert f"| {row.category} |" in markdown


def test_undefined_figures_render_as_dashes_not_zeros(report: EvalReport, tmp_path):
    _, markdown_path, _ = write_report(report, str(tmp_path / "reports"))

    baseline_row = next(
        line
        for line in markdown_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("| baseline |")
    )

    cells = [cell.strip() for cell in baseline_row.strip("|").split("|")]

    assert cells.count("-") == 5
    assert "0.000" in cells


def test_the_markdown_header_names_both_models(report: EvalReport, tmp_path):
    _, markdown_path, _ = write_report(report, str(tmp_path / "reports"))

    markdown = markdown_path.read_text(encoding="utf-8")

    assert f"- Generation model, both systems: {report.generation_model_name}" in markdown
    assert f"- Judge model: {report.judge_model_name}" in markdown


def test_the_markdown_says_agreement_is_unmeasured_until_it_is(report: EvalReport, tmp_path):
    _, markdown_path, _ = write_report(report, str(tmp_path / "reports"))

    assert "Not measured" in markdown_path.read_text(encoding="utf-8")


# ---------- judge agreement ----------


def test_the_agreement_sample_holds_only_entailment_judgements(report: EvalReport):
    samples = build_agreement_sample(report.results, dataset(), sample_size=10)

    assert [sample.id for sample in samples] == ["q-prose", "q-prose"]
    assert {sample.system for sample in samples} == {SYSTEM_PIPELINE, SYSTEM_BASELINE}
    assert all(sample.human_verdict is None for sample in samples)
    assert all(sample.reference == PROSE_REFERENCE for sample in samples)


def test_a_hand_scored_sample_reaches_the_report(harness: Harness, tmp_path):
    scored = tmp_path / "entailment_scored.json"
    samples = [
        EntailmentSample(
            id="q-prose",
            system=SYSTEM_PIPELINE,
            question="What does Blitzcrank's hook do?",
            reference=PROSE_REFERENCE,
            answer="It pulls an enemy in.",
            judge_verdict="entails",
            human_verdict="partial",
        )
    ]
    scored.write_text(
        f"[{samples[0].model_dump_json()}]",
        encoding="utf-8",
    )

    report = run_evaluation(get_settings(), dataset(), human_scores_path=scored)

    assert report.judge_agreement is not None
    assert report.judge_agreement.num_hand_scored == 1
    assert report.judge_agreement.agreement_rate == 0.0
    assert compute_agreement(samples).disagreement_ids == ["q-prose"]
