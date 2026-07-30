from dataclasses import dataclass
from math import floor, isfinite, log10

from lolrag.ingest.formulas import (
    BreakpointsTerm,
    ByLevelTerm,
    ConstantTerm,
    ProductTerm,
    StatTerm,
    SumTerm,
    Term,
)

MIN_CHARACTER_LEVEL = 1
MAX_CHARACTER_LEVEL = 18

SIGNIFICANT_DIGITS = 4
PERCENT_SCALE = 100.0
STAT_REFERENCE = 100

RANK_SEPARATOR = "/"
LEVEL_RANGE_SEPARATOR = " to "
LEVEL_ANNOTATION = "(based on level)"
UNNAMED_STAT = "of an unnamed stat"
PART_SEPARATOR = " "

STAT_NAMES = {
    "ad": "AD",
    "ap": "AP",
    "armor": "armor",
    "magic_resist": "magic resist",
    "attack_speed": "attack speed",
    "crit": "critical strike chance",
    "health": "health",
}


# ---------- flattened form ----------


@dataclass(frozen=True)
class _Range:
    """One number per rank, at the lowest and the highest character level.

    Args:
        low: Value at each rank when the champion is at level one.
        high: Value at each rank when the champion is at the maximum level.
            Equal to low entry by entry for anything that does not scale with
            character level.
    """

    low: tuple[float, ...]
    high: tuple[float, ...]


@dataclass(frozen=True)
class _Scaling:
    """One champion stat the formula scales with.

    Args:
        coefficient: The fraction of the stat the formula adds, per rank.
        stat: Scaling stat code, or None when the source enum is undecoded.
        stat_formula: "total" or "bonus", or None when the source enum is
            undecoded.
    """

    coefficient: _Range
    stat: str | None
    stat_formula: str | None


@dataclass(frozen=True)
class _Flat:
    """A term tree reduced to a base number plus its stat scalings.

    Args:
        base: Everything in the tree that is a plain number, already summed.
        scalings: The stat-scaling contributions, in source order.
    """

    base: _Range
    scalings: tuple[_Scaling, ...]


# ---------- arithmetic ----------


def _broadcast(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:
    """Stretch a one-entry tuple so two per-rank tuples line up.

    Args:
        left: Per-rank numbers.
        right: Per-rank numbers.

    Returns:
        The two tuples at equal length, or an empty tuple when neither is a
        single entry and the lengths differ, which no formula in this corpus
        produces but which must not silently truncate if one ever does.
    """
    if len(left) == len(right):
        return left, right
    if len(left) == 1:
        return left * len(right), right
    if len(right) == 1:
        return left, right * len(left)
    return ()


def _combine(left: _Range, right: _Range, multiply: bool) -> _Range | None:
    """Add or multiply two ranges entry by entry.

    Args:
        left: Left operand.
        right: Right operand.
        multiply: True to multiply, False to add.

    Returns:
        The combined range, or None when the two operands cannot be lined up.
    """
    lows = _broadcast(left.low, right.low)
    highs = _broadcast(left.high, right.high)
    if not lows or not highs:
        return None
    if multiply:
        return _Range(
            low=tuple(a * b for a, b in zip(*lows, strict=True)),
            high=tuple(a * b for a, b in zip(*highs, strict=True)),
        )
    return _Range(
        low=tuple(a + b for a, b in zip(*lows, strict=True)),
        high=tuple(a + b for a, b in zip(*highs, strict=True)),
    )


def _scaled(value: _Range, factor: float) -> _Range:
    """Multiply every number in a range by one constant.

    Args:
        value: The range.
        factor: The constant.

    Returns:
        The scaled range.
    """
    return _Range(
        low=tuple(entry * factor for entry in value.low),
        high=tuple(entry * factor for entry in value.high),
    )


def _multiply_flats(left: _Flat, right: _Flat) -> _Flat | None:
    """Multiply two flattened terms.

    Args:
        left: Left operand.
        right: Right operand.

    Returns:
        The product, or None when both operands carry stat scalings, because a
        stat times a stat is a quadratic this renderer cannot state honestly.
    """
    if left.scalings and right.scalings:
        return None
    if right.scalings:
        left, right = right, left
    base = _combine(left.base, right.base, multiply=True)
    if base is None:
        return None
    scalings: list[_Scaling] = []
    for scaling in left.scalings:
        coefficient = _combine(scaling.coefficient, right.base, multiply=True)
        if coefficient is None:
            return None
        scalings.append(
            _Scaling(
                coefficient=coefficient,
                stat=scaling.stat,
                stat_formula=scaling.stat_formula,
            )
        )
    return _Flat(base=base, scalings=tuple(scalings))


def _breakpoints_value(term: BreakpointsTerm, level: int) -> float:
    """Walk a breakpoints term up to one character level.

    Args:
        term: The breakpoints term.
        level: Character level to evaluate at.

    Returns:
        The value at that level. A step's bonus-per-level applies from its own
        level onward and replaces the rate that was in force before it, so a
        step publishing nothing but a level sets the rate to zero and stops the
        growth there. The corpus proves this: Camille's Q converts 40% of the
        attack at level one and grows 4% a level, and its only step is a bare
        level 17, which caps the value at exactly 100% instead of running on to
        an impossible 108%; Azir's W secondary-target modifier reaches exactly
        100% at level 18 the same way.
    """
    value = term.level1_value
    rate = term.initial_bonus_per_level
    steps = sorted(term.steps, key=lambda step: step.level)
    for current in range(MIN_CHARACTER_LEVEL + 1, level + 1):
        for step in steps:
            if step.level == current:
                value += step.additional_bonus
                rate = step.bonus_per_level_after
        value += rate
    return value


def _flatten(term: Term) -> _Flat | None:
    """Reduce a term tree to a base number plus its stat scalings.

    Args:
        term: Root of the term tree.

    Returns:
        The flattened form, or None when the tree has a shape this renderer
        cannot lay out as one line, which is a stat multiplied by a stat.
    """
    if isinstance(term, ConstantTerm):
        return _Flat(base=_Range(low=term.values, high=term.values), scalings=())

    if isinstance(term, ByLevelTerm):
        return _Flat(base=_Range(low=(term.start,), high=(term.end,)), scalings=())

    if isinstance(term, BreakpointsTerm):
        return _Flat(
            base=_Range(
                low=(_breakpoints_value(term, MIN_CHARACTER_LEVEL),),
                high=(_breakpoints_value(term, MAX_CHARACTER_LEVEL),),
            ),
            scalings=(),
        )

    if isinstance(term, StatTerm):
        coefficient = _flatten(term.coefficient)
        if coefficient is None or coefficient.scalings:
            return None
        return _Flat(
            base=_Range(low=(0.0,), high=(0.0,)),
            scalings=(
                _Scaling(
                    coefficient=coefficient.base,
                    stat=term.stat,
                    stat_formula=term.stat_formula,
                ),
            ),
        )

    if isinstance(term, SumTerm):
        base = _Range(low=(0.0,), high=(0.0,))
        scalings: list[_Scaling] = []
        for child in term.terms:
            flat = _flatten(child)
            if flat is None:
                return None
            combined = _combine(base, flat.base, multiply=False)
            if combined is None:
                return None
            base = combined
            scalings.extend(flat.scalings)
        return _Flat(base=base, scalings=tuple(scalings))

    if isinstance(term, ProductTerm):
        product = _Flat(base=_Range(low=(1.0,), high=(1.0,)), scalings=())
        for child in term.terms:
            flat = _flatten(child)
            if flat is None:
                return None
            multiplied = _multiply_flats(product, flat)
            if multiplied is None:
                return None
            product = multiplied
        return product

    return None


# ---------- number formatting ----------


def _round_significant(value: float, digits: int) -> float:
    """Drop the float32 noise the source carries.

    Args:
        value: The raw number.
        digits: Number of significant digits to keep.

    Returns:
        The number rounded to that many significant digits, so a stored
        0.800000011920929 reads as 0.8 and a stored 0.3333300054073334 reads as
        0.3333.
    """
    if value == 0.0 or not isfinite(value):
        return value
    return round(value, digits - 1 - floor(log10(abs(value))))


def _format_number(value: float, decimals: int | None) -> str:
    """Render one number without scientific notation or trailing zeros.

    Args:
        value: The number.
        decimals: Exact number of decimal places the source asked for, or None
            to keep four significant digits.

    Returns:
        The number as text. An explicit decimal count is honoured exactly, so a
        source asking for one decimal renders 1.0 rather than 1; otherwise
        trailing zeros are dropped. A value that rounds to nothing negative
        renders as "0" rather than "-0".
    """
    if decimals is not None:
        text = f"{value:.{decimals}f}"
        return text.removeprefix("-") if float(text) == 0.0 else text
    text = f"{_round_significant(value, SIGNIFICANT_DIGITS):.10f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("", "-", "-0") else text


def _format_range(value: _Range, decimals: int | None) -> tuple[str, bool] | None:
    """Render a per-rank range as one string.

    Args:
        value: The range.
        decimals: Exact number of decimal places, or None for four significant
            digits.

    Returns:
        The text and whether any entry spans two character levels, or None when
        the range's two ends cannot be lined up. Entries identical at every rank
        collapse to a single number, which is what 68 percent of this corpus
        needs; the rest join with a slash.
    """
    ends = _broadcast(value.low, value.high)
    if not ends:
        return None
    entries: list[str] = []
    level_scaled = False
    for low, high in zip(*ends, strict=True):
        low_text = _format_number(low, decimals)
        high_text = _format_number(high, decimals)
        if low_text == high_text:
            entries.append(low_text)
            continue
        entries.append(f"{low_text}{LEVEL_RANGE_SEPARATOR}{high_text}")
        level_scaled = True
    if len(set(entries)) == 1:
        entries = entries[:1]
    return RANK_SEPARATOR.join(entries), level_scaled


def _stat_label(scaling: _Scaling) -> str | None:
    """Name the stat a scaling applies to.

    Args:
        scaling: The scaling.

    Returns:
        The readable stat name preceded by "total" or "bonus", or None when the
        source enum leaves the stat undecoded. Nothing is guessed: a coefficient
        whose stat cannot be proven is shown without a stat name rather than
        attributed to the wrong one.
    """
    if scaling.stat is None:
        return None
    name = STAT_NAMES.get(scaling.stat, scaling.stat)
    if scaling.stat_formula is None:
        return name
    return f"{scaling.stat_formula} {name}"


# ---------- public api ----------


def format_term(
    term: Term, *, scale: float = 1.0, percent: bool = False, decimals: int | None = None
) -> str | None:
    """Render a term tree as the number a tooltip shows in its place.

    Args:
        term: Root of the term tree, as evaluate_calculation returns it.
        scale: Constant every number in the tree is multiplied by first, which
            is how a tooltip token's own arithmetic reaches the renderer.
        percent: Whether the result is a fraction to show as a percentage. This
            changes the units of a stat scaling as well as the base: a
            coefficient that adds damage per point of a stat renders as a
            percentage of that stat, whereas one that adds percentage points per
            point of a stat renders per hundred points of it.
        decimals: Exact number of decimal places the source asked for, or None
            to keep four significant digits.

    Returns:
        The rendered text, or None when the tree cannot be laid out on one line:
        a stat multiplied by another stat, a percentage scaling with a stat the
        source enum leaves undecoded, or per-rank arrays of different lengths.
        None is returned rather than a partial string, because half a formula
        reads as a whole one.
    """
    flat = _flatten(term)
    if flat is None:
        return None

    base = _scaled(flat.base, scale * (PERCENT_SCALE if percent else 1.0))
    formatted = _format_range(base, decimals)
    if formatted is None:
        return None
    base_text, base_level_scaled = formatted

    parts: list[str] = []
    if not (flat.scalings and all(entry == 0.0 for entry in base.low + base.high)):
        text = f"{base_text}%" if percent else base_text
        parts.append(f"{text} {LEVEL_ANNOTATION}" if base_level_scaled else text)

    for scaling in flat.scalings:
        label = _stat_label(scaling)
        if percent:
            if label is None:
                return None
            factor = scale * PERCENT_SCALE * STAT_REFERENCE
            suffix = f" per {STAT_REFERENCE} {label}"
        else:
            factor = scale * PERCENT_SCALE
            suffix = f" {label}" if label is not None else f" {UNNAMED_STAT}"
        coefficient = _format_range(_scaled(scaling.coefficient, factor), None)
        if coefficient is None:
            return None
        coefficient_text, coefficient_level_scaled = coefficient
        body = f"{coefficient_text}%{suffix}"
        if coefficient_level_scaled:
            body = f"{body} {LEVEL_ANNOTATION}"
        parts.append(body if not parts else f"(+{body})")

    return PART_SEPARATOR.join(parts)
