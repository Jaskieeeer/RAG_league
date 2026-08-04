"""Deterministic numeric answer scoring, with no model anywhere in the loop.

The numeric block is where a retrieval-free baseline is expected to lose, so it
is graded by string arithmetic rather than by a judge: a general-knowledge model
asked to grade numbers it also knows would quietly grade its own recall.

The rule, stated once:

1. Only the first sentence of the reference answer supplies required numbers. A
   reference reads "2450 gold. The Arena version is a different item at 2500." -
   the later sentences are authoring commentary, and their numbers are either
   contrasts a correct answer must NOT repeat or unit conversions the corpus
   deliberately does not make. If the first sentence carries no number at all,
   the next sentence that does is used instead.
2. Inside that sentence a slash-joined run of numbers is one required token, a
   per-rank series, satisfied only when every member appears in the response.
   Every other number is a required scalar token.
3. Numbers that index a rank rather than state a value are dropped: those
   directly preceded by rank, ranks, level, levels, tier, tiers, and those
   directly followed by times, stacks, charges.
4. A scalar equal to a member of a required series is dropped as redundant.
5. A token is satisfied when its numbers appear anywhere among the numbers in
   the response, compared as decimals, so 0.90 matches 0.9 and 1,200 matches
   1200. Order does not matter and extra numbers in the response are ignored;
   the response is only ever asked to contain the reference's numbers, never to
   contain nothing else.
6. No unit conversion is applied. A reference of 0.15 is not satisfied by "15
   percent": the corpus stores several values as unflagged fractions and the
   dataset asserts that a grounded answer must not silently convert them.
"""

import re
from decimal import Decimal

from pydantic import BaseModel, Field

_THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_SERIES = re.compile(r"\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_TRAILING_WORD = re.compile(r"([A-Za-z]+)\W*$")
_LEADING_WORD = re.compile(r"^\W*([A-Za-z]+)")

RANK_QUALIFIERS = frozenset({"rank", "ranks", "level", "levels", "tier", "tiers"})
COUNT_QUALIFIERS = frozenset({"times", "stacks", "charges"})

SERIES_SEPARATOR = "/"


class NumericScore(BaseModel):
    """The outcome of grading one answer against the reference's numbers.

    Args:
        required: Required numeric tokens rendered as text, a per-rank series
            joined by slashes and a scalar on its own.
        matched: Required tokens every number of which appears in the response.
        missing: Required tokens at least one number of which is absent.
        coverage: Fraction of required tokens matched, 1.0 when the reference
            requires no number at all.
        passed: Whether every required token was matched.
    """

    required: list[str] = Field(description="Required numeric tokens rendered as text.")
    matched: list[str] = Field(description="Required tokens present in the response.")
    missing: list[str] = Field(description="Required tokens absent from the response.")
    coverage: float = Field(description="Fraction of required tokens matched.")
    passed: bool = Field(description="Whether every required token was matched.")


# ---------- extraction ----------


def _normalise(text: str) -> str:
    """Strip thousands separators so "1,200" reads as one number.

    Args:
        text: Raw reference or response text.

    Returns:
        The text with a comma removed wherever it sits between a digit and
        exactly three further digits, which leaves list commas such as "10, 25"
        untouched.
    """
    return _THOUSANDS_SEPARATOR.sub("", text)


def extract_numbers(text: str) -> list[Decimal]:
    """Extract every number in a piece of text as a decimal.

    Args:
        text: Text to scan, typically a generated answer.

    Returns:
        Every unsigned integer or decimal literal in reading order, thousands
        separators removed first. Decimals are used rather than floats so that
        0.1 compares equal to 0.10 and never suffers binary rounding.
    """
    return [Decimal(match.group()) for match in _NUMBER.finditer(_normalise(text))]


def _is_rank_index(sentence: str, start: int, end: int) -> bool:
    """Report whether a number at a given span indexes a rank instead of stating a value.

    Args:
        sentence: The sentence the number was found in, already normalised.
        start: Start offset of the number.
        end: End offset of the number.

    Returns:
        True if the word immediately before the number names a rank or level, or
        the word immediately after counts occurrences, otherwise False.
    """
    before = _TRAILING_WORD.search(sentence[:start])
    after = _LEADING_WORD.search(sentence[end:])
    if before is not None and before.group(1).lower() in RANK_QUALIFIERS:
        return True
    return after is not None and after.group(1).lower() in COUNT_QUALIFIERS


def _sentence_tokens(sentence: str) -> list[tuple[Decimal, ...]]:
    """Extract the required numeric tokens of a single sentence.

    Args:
        sentence: One sentence of a reference answer, already normalised.

    Returns:
        One tuple per required token, a per-rank series as its members in source
        order and a scalar as a one-element tuple. Rank indices are dropped,
        scalars already covered by a series are dropped, and duplicate tokens
        appear once.
    """
    series: list[tuple[Decimal, ...]] = []
    spans: list[tuple[int, int]] = []
    for match in _SERIES.finditer(sentence):
        series.append(tuple(Decimal(part) for part in _NUMBER.findall(match.group())))
        spans.append(match.span())

    covered = {value for token in series for value in token}
    scalars: list[tuple[Decimal, ...]] = []
    for match in _NUMBER.finditer(sentence):
        start, end = match.span()
        if any(span_start <= start < span_end for span_start, span_end in spans):
            continue
        if _is_rank_index(sentence, start, end):
            continue
        value = Decimal(match.group())
        if value in covered:
            continue
        scalars.append((value,))

    tokens: list[tuple[Decimal, ...]] = []
    for token in [*series, *scalars]:
        if token not in tokens:
            tokens.append(token)
    return tokens


def required_tokens(expected_answer: str) -> list[tuple[Decimal, ...]]:
    """Extract the numeric tokens a response must contain to satisfy a reference.

    Args:
        expected_answer: The dataset's reference answer.

    Returns:
        The tokens of the first sentence that carries any, in that sentence's
        order, or an empty list when the reference states no number anywhere.
    """
    for sentence in _SENTENCE_BOUNDARY.split(_normalise(expected_answer)):
        tokens = _sentence_tokens(sentence)
        if tokens:
            return tokens
    return []


# ---------- scoring ----------


def render_token(token: tuple[Decimal, ...]) -> str:
    """Render one numeric token for the report.

    Args:
        token: Required token, a series of two or more numbers or a lone scalar.

    Returns:
        The numbers joined by slashes, which is the notation the corpus itself
        uses for a per-rank series.
    """
    return SERIES_SEPARATOR.join(str(value) for value in token)


def score_numeric(expected_answer: str, response: str) -> NumericScore:
    """Grade a response against the numbers a reference answer requires.

    Args:
        expected_answer: The dataset's reference answer.
        response: The answer produced by the system under test.

    Returns:
        NumericScore listing which required tokens the response contains, which
        it omits, and whether it contains all of them. A reference that requires
        no number passes with a coverage of 1.0; the dataset is asserted to hold
        no such numeric question in the test suite.
    """
    tokens = required_tokens(expected_answer)
    present = set(extract_numbers(response))
    matched = [token for token in tokens if all(value in present for value in token)]
    missing = [token for token in tokens if token not in matched]
    return NumericScore(
        required=[render_token(token) for token in tokens],
        matched=[render_token(token) for token in matched],
        missing=[render_token(token) for token in missing],
        coverage=len(matched) / len(tokens) if tokens else 1.0,
        passed=not missing,
    )
