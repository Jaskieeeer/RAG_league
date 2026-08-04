"""The checked-in golden dataset and the schema every question is validated against.

A question's `scoring` field decides which headline figure it contributes to and
`check` decides how its answer is graded. The two are related but not derivable
from one another: retrieval questions split between a deterministic numeric check
and an LLM entailment check, and only the dataset knows which of the two a given
question was authored for.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

_DEFAULT_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"

SCORING_RETRIEVAL = "retrieval"
SCORING_REFUSAL = "refusal"
SCORING_LIMITATION = "limitation"

CHECK_NUMERIC = "numeric"
CHECK_ENTAILMENT = "entailment"
CHECK_REFUSAL = "refusal"
CHECK_NONE = "none"

_CHECKS_BY_SCORING = {
    SCORING_RETRIEVAL: frozenset({CHECK_NUMERIC, CHECK_ENTAILMENT}),
    SCORING_REFUSAL: frozenset({CHECK_REFUSAL}),
    SCORING_LIMITATION: frozenset({CHECK_NONE}),
}


class GoldenQuestion(BaseModel):
    """A single golden question and the way its answer is to be scored.

    Args:
        id: Stable question identifier, e.g. "abil-num-001".
        question: Natural-language question posed to the system under test.
        category: Authoring category the question belongs to, e.g.
            "ability-numbers"; the unit the report stratifies by.
        scoring: Which headline figure the question contributes to, one of
            retrieval, refusal, limitation.
        check: How the answer is graded, one of numeric, entailment, refusal,
            none.
        expected_doc_keys: Corpus doc_keys that answer the question, e.g.
            "ability:aatrox:Q"; empty for questions with no correct document.
        expected_answer: Reference answer, the only authority the entailment
            judge and the numeric check are allowed to compare against.
        collection: Collection the answering documents live in, one of
            abilities, champion_stats, equipment, lore, or None when there is no
            answering document.
        failure_mode: Authoring note naming the failure this question is meant
            to expose.
    """

    id: str = Field(description="Stable question identifier.")
    question: str = Field(description="Natural-language question posed to the system.")
    category: str = Field(description="Authoring category the report stratifies by.")
    scoring: Literal["retrieval", "refusal", "limitation"] = Field(
        description="Which headline figure the question contributes to."
    )
    check: Literal["numeric", "entailment", "refusal", "none"] = Field(
        description="How the answer is graded."
    )
    expected_doc_keys: list[str] = Field(description="Corpus doc_keys that answer the question.")
    expected_answer: str = Field(description="Reference answer the check compares against.")
    collection: str | None = Field(description="Collection the answering documents live in.")
    failure_mode: str = Field(description="Failure this question is meant to expose.")

    @model_validator(mode="after")
    def _check_matches_scoring(self) -> "GoldenQuestion":
        """Reject a question whose check or doc keys contradict its scoring mode.

        Returns:
            The validated question.

        Raises:
            ValueError: If the check is not one the scoring mode allows, if a
                retrieval question names no expected doc key, or if a refusal or
                limitation question names one; either would silently distort a
                headline figure.
        """
        allowed = _CHECKS_BY_SCORING[self.scoring]
        if self.check not in allowed:
            raise ValueError(
                f"{self.id}: scoring {self.scoring!r} allows checks {sorted(allowed)}, "
                f"not {self.check!r}"
            )
        if self.scoring == SCORING_RETRIEVAL and not self.expected_doc_keys:
            raise ValueError(f"{self.id}: a retrieval question must name at least one doc key")
        if self.scoring != SCORING_RETRIEVAL and self.expected_doc_keys:
            raise ValueError(
                f"{self.id}: a {self.scoring} question has no correct document, so it must "
                "name no doc keys"
            )
        return self


class GoldenDataset(BaseModel):
    """The full checked-in golden evaluation dataset.

    Args:
        description: Human-readable description of the dataset and its intent.
        version: Integer dataset version, bumped when questions change.
        questions: Golden questions the harness evaluates against.
    """

    description: str = Field(description="Human-readable description of the dataset.")
    version: int = Field(description="Integer dataset version.")
    questions: list[GoldenQuestion] = Field(description="Golden questions to evaluate.")

    @model_validator(mode="after")
    def _ids_are_unique(self) -> "GoldenDataset":
        """Reject a dataset that reuses a question id.

        Returns:
            The validated dataset.

        Raises:
            ValueError: If two questions share an id, which would make a
                per-question result ambiguous to trace back.
        """
        counts = Counter(question.id for question in self.questions)
        duplicates = sorted(id_ for id_, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate question ids: {duplicates}")
        return self

    def by_scoring(self, scoring: str) -> list[GoldenQuestion]:
        """Select the questions belonging to one scoring mode.

        Args:
            scoring: Scoring mode to select, one of retrieval, refusal,
                limitation.

        Returns:
            The questions whose scoring equals the given mode, in dataset order.
        """
        return [question for question in self.questions if question.scoring == scoring]


def load_golden_dataset(path: Path | None = None) -> GoldenDataset:
    """Load and validate the golden evaluation dataset from disk.

    Args:
        path: Path to the dataset JSON file, or None to use the packaged
            golden_dataset.json next to this module.

    Returns:
        GoldenDataset parsed and validated from the JSON file.

    Raises:
        pydantic.ValidationError: If any question violates the schema or the
            scoring and check fields contradict one another.
    """
    dataset_path = path if path is not None else _DEFAULT_DATASET_PATH
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    return GoldenDataset.model_validate(raw)
