"""Judge-human agreement over a sample of entailment judgements.

The entailment judge cannot be taken on trust, and the only honest mitigation is
to measure it: dump a deterministic sample of its judgements, hand-score them,
and report how often the two agree. The round trip is a single JSON file whose
human_verdict fields start null and are filled in by hand.
"""

import json
import random
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter

SAMPLE_FILENAME = "entailment_sample.json"
SCORED_FILENAME = "entailment_scored.json"


class EntailmentSample(BaseModel):
    """One entailment judgement offered up for hand-scoring.

    Args:
        id: Golden question identifier.
        system: System whose answer was judged, pipeline or baseline.
        question: The question that was asked.
        reference: The dataset's reference answer the judge compared against.
        answer: The answer the system produced.
        judge_verdict: The judge's verdict, one of entails, partial, misses.
        human_verdict: The hand-assigned verdict, or None until a human fills it
            in.
    """

    id: str = Field(description="Golden question identifier.")
    system: str = Field(description="System whose answer was judged.")
    question: str = Field(description="The question that was asked.")
    reference: str = Field(description="Reference answer the judge compared against.")
    answer: str = Field(description="The answer the system produced.")
    judge_verdict: Literal["entails", "partial", "misses"] = Field(
        description="The judge's verdict."
    )
    human_verdict: Literal["entails", "partial", "misses"] | None = Field(
        default=None, description="The hand-assigned verdict, null until scored."
    )


class AgreementSummary(BaseModel):
    """How often the hand scores and the judge agreed.

    Args:
        num_hand_scored: Judgements a human actually scored.
        num_agreed: Judgements where the human verdict equalled the judge's.
        agreement_rate: num_agreed divided by num_hand_scored.
        disagreement_ids: Question ids where the two differed, so the prompt can
            be argued with rather than guessed at.
    """

    num_hand_scored: int = Field(description="Judgements a human actually scored.")
    num_agreed: int = Field(description="Judgements where human and judge matched.")
    agreement_rate: float = Field(description="Fraction of hand-scored judgements that matched.")
    disagreement_ids: list[str] = Field(description="Question ids where the two differed.")


_SAMPLE_ADAPTER: TypeAdapter[list[EntailmentSample]] = TypeAdapter(list[EntailmentSample])


# ---------- sampling ----------


def select_sample(
    candidates: list[EntailmentSample], sample_size: int, seed: int = 0
) -> list[EntailmentSample]:
    """Draw a deterministic sample of entailment judgements for hand-scoring.

    Args:
        candidates: Every entailment judgement made during a run.
        sample_size: Number of judgements to draw; the whole list is returned
            when it holds fewer.
        seed: Seed for the draw, fixed so two runs of the same size sample the
            same questions and the hand scores stay comparable.

    Returns:
        The drawn judgements, ordered by question id and then system.
    """
    if sample_size >= len(candidates):
        drawn = list(candidates)
    else:
        drawn = random.Random(seed).sample(candidates, sample_size)
    return sorted(drawn, key=lambda sample: (sample.id, sample.system))


# ---------- round trip ----------


def dump_entailment_sample(samples: list[EntailmentSample], path: Path) -> Path:
    """Write a sample of entailment judgements out for hand-scoring.

    Args:
        samples: The judgements to write.
        path: File to write them to; parent directories are created.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [sample.model_dump() for sample in samples]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_entailment_sample(path: Path) -> list[EntailmentSample]:
    """Read a hand-scored sample of entailment judgements back in.

    Args:
        path: File written by dump_entailment_sample, with human_verdict fields
            filled in.

    Returns:
        The judgements, including any whose human_verdict is still null.

    Raises:
        pydantic.ValidationError: If a hand-written verdict is not one of
            entails, partial or misses.
    """
    return _SAMPLE_ADAPTER.validate_json(path.read_text(encoding="utf-8"))


def compute_agreement(samples: list[EntailmentSample]) -> AgreementSummary:
    """Measure how often the hand scores agreed with the judge.

    Args:
        samples: Judgements, of which only those carrying a human_verdict count.

    Returns:
        AgreementSummary over the hand-scored judgements, with an agreement rate
        of 0.0 when none were scored.
    """
    scored = [sample for sample in samples if sample.human_verdict is not None]
    agreed = [sample for sample in scored if sample.human_verdict == sample.judge_verdict]
    disagreed = [sample.id for sample in scored if sample.human_verdict != sample.judge_verdict]
    return AgreementSummary(
        num_hand_scored=len(scored),
        num_agreed=len(agreed),
        agreement_rate=len(agreed) / len(scored) if scored else 0.0,
        disagreement_ids=sorted(disagreed),
    )
