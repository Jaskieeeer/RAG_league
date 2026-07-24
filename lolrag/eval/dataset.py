import json
from pathlib import Path

from pydantic import BaseModel, Field

_DEFAULT_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


class GoldenQuestion(BaseModel):
    """A single golden question paired with the champion sources that answer it.

    Args:
        id: Stable question identifier, e.g. "q01".
        question: Natural-language question posed to the RAG pipeline.
        expected_champion_ids: Data Dragon champion ids whose corpus document
            should be retrieved to answer the question.
        category: Question category, either "factual" or "refusal".
        notes: Optional authoring note explaining the expected behaviour.
    """

    id: str = Field(description="Stable question identifier.")
    question: str = Field(description="Natural-language question posed to the pipeline.")
    expected_champion_ids: list[str] = Field(
        description="Champion ids whose document should be retrieved."
    )
    category: str = Field(description="Question category, factual or refusal.")
    notes: str | None = Field(default=None, description="Optional authoring note.")


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


def load_golden_dataset(path: Path | None = None) -> GoldenDataset:
    """Load and validate the golden evaluation dataset from disk.

    Args:
        path: Path to the dataset JSON file, or None to use the packaged
            golden_dataset.json next to this module.

    Returns:
        GoldenDataset parsed and validated from the JSON file.
    """
    dataset_path = path if path is not None else _DEFAULT_DATASET_PATH
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    return GoldenDataset.model_validate(raw)
