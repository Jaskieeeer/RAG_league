from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lolrag.ingest.values import (
    DATA_VALUE_OFFSET,
    INTERPOLATION_TYPE,
    SCALING_STAT_BY_CODE,
    STAT_BY_NAMED_DATA_VALUE_TYPE,
    STAT_BY_SUB_PART_TYPE,
    STAT_FORMULA_BY_CODE,
    slice_ranks,
)

GAME_CALCULATION_TYPE = "GameCalculation"
GAME_CALCULATION_MODIFIED_TYPE = "GameCalculationModified"
NAMED_DATA_VALUE_TYPE = "NamedDataValueCalculationPart"
NUMBER_TYPE = "NumberCalculationPart"
EFFECT_VALUE_TYPE = "EffectValueCalculationPart"
STAT_BY_COEFFICIENT_TYPE = "StatByCoefficientCalculationPart"
SUM_OF_SUB_PARTS_TYPE = "SumOfSubPartsCalculationPart"
PRODUCT_OF_SUB_PARTS_TYPE = "ProductOfSubPartsCalculationPart"
BREAKPOINTS_TYPE = "ByCharLevelBreakpointsCalculationPart"
BREAKPOINT_TYPE = "Breakpoint"

MISSING_CALCULATION_TYPE = "MissingGameCalculation"
NON_OBJECT_TYPE = "NonObjectCalculationPart"

MODIFIED_REFERENCE_KEY = "mModifiedGameCalculation"
DAMAGE_TYPE_FIELD = "{72c5c2a8}"

RECOGNISED_FIELDS = {
    GAME_CALCULATION_TYPE: frozenset(
        {
            "mFormulaParts",
            "mMultiplier",
            "mDisplayAsPercent",
            "mPrecision",
            "tooltipOnly",
            "mSimpleTooltipCalculationDisplay",
            "mExpandedTooltipCalculationDisplay",
            DAMAGE_TYPE_FIELD,
        }
    ),
    GAME_CALCULATION_MODIFIED_TYPE: frozenset(
        {
            MODIFIED_REFERENCE_KEY,
            "mMultiplier",
            "tooltipOnly",
            "mSimpleTooltipCalculationDisplay",
            "mExpandedTooltipCalculationDisplay",
            DAMAGE_TYPE_FIELD,
        }
    ),
    NAMED_DATA_VALUE_TYPE: frozenset({"mDataValue"}),
    NUMBER_TYPE: frozenset({"mNumber"}),
    EFFECT_VALUE_TYPE: frozenset({"mEffectIndex"}),
    STAT_BY_COEFFICIENT_TYPE: frozenset({"mCoefficient", "mStat", "mStatFormula"}),
    STAT_BY_NAMED_DATA_VALUE_TYPE: frozenset({"mDataValue", "mStat", "mStatFormula"}),
    STAT_BY_SUB_PART_TYPE: frozenset({"mSubpart", "mStat", "mStatFormula"}),
    SUM_OF_SUB_PARTS_TYPE: frozenset({"mSubparts"}),
    PRODUCT_OF_SUB_PARTS_TYPE: frozenset({"mPart1", "mPart2"}),
    INTERPOLATION_TYPE: frozenset({"mStartValue", "mEndValue", "mScalePastDefaultMaxLevel"}),
    BREAKPOINTS_TYPE: frozenset({"mLevel1Value", "mInitialBonusPerLevel", "mBreakpoints"}),
    BREAKPOINT_TYPE: frozenset(
        {"mLevel", "mAdditionalBonusAtThisLevel", "mBonusPerLevelAtAndAfter"}
    ),
}


# ---------- term tree ----------


@dataclass(frozen=True)
class ConstantTerm:
    """A number the formula states outright, one entry per learnable rank.

    Args:
        values: The number at each rank, lowest rank first. A rank-invariant
            number is repeated rather than collapsed, so every consumer of the
            tree handles one shape.
    """

    values: tuple[float, ...]


@dataclass(frozen=True)
class StatTerm:
    """A champion stat multiplied by a coefficient.

    Args:
        coefficient: The term the stat is multiplied by, itself any term because
            the source lets a subpart supply the coefficient.
        stat: Scaling stat name, or None when the source mStat enum has no
            proven meaning. Never guessed.
        stat_formula: Which part of the stat the coefficient applies to, one of
            "total", "bonus", or None when the source mStatFormula enum has no
            proven meaning.
    """

    coefficient: "Term"
    stat: str | None
    stat_formula: str | None


@dataclass(frozen=True)
class ByLevelTerm:
    """A number interpolated linearly across the champion's levels.

    Args:
        start: Value at character level one.
        end: Value at the default maximum character level.
    """

    start: float
    end: float


@dataclass(frozen=True)
class BreakpointStep:
    """One character level at which a by-level number changes how it grows.

    Args:
        level: Character level the step applies from.
        additional_bonus: One-off amount added on reaching the level.
        bonus_per_level_after: Amount added per level at and after this level.
            An omitted source field is zero like every other omitted field, so a
            step carrying nothing but a level stops the growth there.
    """

    level: int
    additional_bonus: float
    bonus_per_level_after: float


@dataclass(frozen=True)
class BreakpointsTerm:
    """A number that grows with character level in piecewise steps.

    Args:
        level1_value: Value at character level one.
        initial_bonus_per_level: Amount added per level before the first step.
        steps: The steps, in source order.
    """

    level1_value: float
    initial_bonus_per_level: float
    steps: tuple[BreakpointStep, ...]


@dataclass(frozen=True)
class SumTerm:
    """The sum of its child terms.

    Args:
        terms: Child terms, in source order.
    """

    terms: tuple["Term", ...]


@dataclass(frozen=True)
class ProductTerm:
    """The product of its child terms.

    Args:
        terms: Child terms, in source order.
    """

    terms: tuple["Term", ...]


type Term = ConstantTerm | StatTerm | ByLevelTerm | BreakpointsTerm | SumTerm | ProductTerm


@dataclass(frozen=True)
class Formula:
    """One fully evaluated GameCalculation.

    Args:
        term: Root of the term tree the calculation reduces to.
        display_as_percent: Whether the source asks for the result as a
            percentage. Read off the calculation the caller named and never
            inherited through a reference, because the modifying calculation is
            the one the tooltip renders.
        precision: Number of decimal places the source asks for.
    """

    term: Term
    display_as_percent: bool
    precision: int


@dataclass(frozen=True)
class Unevaluable:
    """The reason a calculation could not be reduced to a term tree.

    Args:
        node_type: Bin __type of the node evaluation stopped at, so the caller
            can count which parts of the source are still unsupported. Two
            pseudo types stand in where no node exists: MissingGameCalculation
            for a name the spell does not publish, and NonObjectCalculationPart
            for a formula slot holding something other than an object.
        reason: Short machine-stable phrase naming what blocked the node, such
            as the missing data value or the unrecognised field.
    """

    node_type: str
    reason: str


# ---------- evaluation ----------


@dataclass(frozen=True)
class _Context:
    """Everything the evaluator needs beyond the node it is looking at.

    Args:
        calculations: The owning spell's mSpellCalculations object, which is
            also the whole set of calculations a reference can name.
        data_values: The owning spell's data values, mapping name to the raw
            source array, or to None for an array the bin publishes as null.
        max_rank: Number of learnable ranks, or None for a passive.
        ranks: Number of entries every constant term carries.
        effect_values: The owning spell's effect amounts by one-based effect
            index, each the raw source array or None for a null one.
    """

    calculations: Mapping[str, Any]
    data_values: Mapping[str, Sequence[float] | None]
    max_rank: int | None
    ranks: int
    effect_values: Mapping[int, Sequence[float] | None]


def evaluate_calculation(
    name: str,
    calculations: Mapping[str, Any],
    data_values: Mapping[str, Sequence[float] | None],
    max_rank: int | None,
    effect_values: Mapping[int, Sequence[float] | None] | None = None,
) -> Formula | Unevaluable:
    """Reduce one named GameCalculation to a term tree.

    Args:
        name: Key of the calculation inside calculations, as a tooltip token
            names it.
        calculations: The owning spell's mSpellCalculations object. Every
            reference this corpus publishes names a sibling in the same object.
        data_values: The owning spell's data values, mapping name to the raw
            seven-wide one-indexed source array, or to None for an array the bin
            publishes as null, which means the value is zero at every rank.
        max_rank: Number of learnable ranks, or None for a passive, whose
            values cannot be sliced by rank.
        effect_values: The owning spell's mEffectAmount arrays by one-based
            effect index, in the same raw seven-wide one-indexed shape as the
            data values. Omit it when the caller has not read them; every
            calculation reaching an EffectValueCalculationPart is then
            unevaluable rather than wrong.

    Returns:
        A Formula when every node in the calculation and in everything it
        references is understood, otherwise an Unevaluable naming the first node
        that blocked it. A partial tree is never returned, because a formula
        missing one term would render a wrong number.
    """
    context = _Context(
        calculations=calculations,
        data_values=data_values,
        max_rank=max_rank,
        ranks=1 if max_rank is None else max_rank,
        effect_values=effect_values or {},
    )
    node = calculations.get(name)
    if not isinstance(node, dict):
        return Unevaluable(node_type=MISSING_CALCULATION_TYPE, reason=f"no calculation {name}")
    term = _calculation_term(name, context, frozenset())
    if isinstance(term, Unevaluable):
        return term
    return Formula(
        term=term,
        display_as_percent=bool(node.get("mDisplayAsPercent", False)),
        precision=int(node.get("mPrecision", 0)),
    )


def _calculation_term(name: str, context: _Context, seen: frozenset[str]) -> Term | Unevaluable:
    """Reduce the calculation a reference names, refusing to follow a cycle.

    Args:
        name: Calculation name to reduce.
        context: Evaluation context.
        seen: Calculation names already being reduced further up the chain.

    Returns:
        The term tree, or an Unevaluable when the name is unknown, the chain
        loops back on itself, or any node beneath it is unsupported.
    """
    if name in seen:
        return Unevaluable(
            node_type=GAME_CALCULATION_MODIFIED_TYPE, reason=f"reference cycle at {name}"
        )
    node = context.calculations.get(name)
    if not isinstance(node, dict):
        return Unevaluable(node_type=MISSING_CALCULATION_TYPE, reason=f"no calculation {name}")

    node_type = node.get("__type", "")
    if node_type not in (GAME_CALCULATION_TYPE, GAME_CALCULATION_MODIFIED_TYPE):
        return Unevaluable(node_type=node_type, reason="unsupported calculation type")
    blocked = _unrecognised_field(node, node_type)
    if blocked is not None:
        return blocked
    chain = seen | {name}

    if node_type == GAME_CALCULATION_TYPE:
        parts: list[Term] = []
        for part in node.get("mFormulaParts") or []:
            term = _part_term(part, context, chain)
            if isinstance(term, Unevaluable):
                return term
            parts.append(term)
        return _multiplied(SumTerm(terms=tuple(parts)), node, context, chain)

    reference = node.get(MODIFIED_REFERENCE_KEY)
    if not isinstance(reference, str):
        return Unevaluable(
            node_type=GAME_CALCULATION_MODIFIED_TYPE, reason="no modified calculation"
        )
    modified = _calculation_term(reference, context, chain)
    if isinstance(modified, Unevaluable):
        return modified
    return _multiplied(modified, node, context, chain)


def _multiplied(
    term: Term, node: Mapping[str, Any], context: _Context, seen: frozenset[str]
) -> Term | Unevaluable:
    """Apply a calculation wrapper's optional multiplier to the term it wraps.

    Args:
        term: Term the wrapper reduced to before the multiplier.
        node: The GameCalculation or GameCalculationModified object.
        context: Evaluation context.
        seen: Calculation names already being reduced further up the chain.

    Returns:
        The term unchanged when the wrapper publishes no mMultiplier, otherwise
        the product of the two. The multiplier is a formula part in its own
        right, not a plain number.
    """
    multiplier = node.get("mMultiplier")
    if multiplier is None:
        return term
    factor = _part_term(multiplier, context, seen)
    if isinstance(factor, Unevaluable):
        return factor
    return ProductTerm(terms=(term, factor))


def _part_term(node: Any, context: _Context, seen: frozenset[str]) -> Term | Unevaluable:
    """Reduce one formula part to a term.

    Args:
        node: A calculation part object taken from mFormulaParts, mSubparts,
            mSubpart, mPart1, mPart2 or mMultiplier.
        context: Evaluation context.
        seen: Calculation names already being reduced further up the chain.

    Returns:
        The term the part reduces to, or an Unevaluable naming the node type
        that blocked it.
    """
    if not isinstance(node, dict):
        return Unevaluable(node_type=NON_OBJECT_TYPE, reason=f"part is {type(node).__name__}")
    node_type = node.get("__type", "")
    if node_type not in RECOGNISED_FIELDS:
        return Unevaluable(node_type=node_type, reason="unsupported part type")
    blocked = _unrecognised_field(node, node_type)
    if blocked is not None:
        return blocked

    if node_type == NUMBER_TYPE:
        return _constant(float(node.get("mNumber", 0.0)), context)

    if node_type == NAMED_DATA_VALUE_TYPE:
        return _data_value_term(str(node.get("mDataValue", "")), context)

    if node_type == EFFECT_VALUE_TYPE:
        return _effect_value_term(int(node.get("mEffectIndex", 0)), context)

    if node_type == STAT_BY_COEFFICIENT_TYPE:
        return _stat_term(node, _constant(float(node.get("mCoefficient", 0.0)), context))

    if node_type == STAT_BY_NAMED_DATA_VALUE_TYPE:
        coefficient = _data_value_term(str(node.get("mDataValue", "")), context)
        if isinstance(coefficient, Unevaluable):
            return coefficient
        return _stat_term(node, coefficient)

    if node_type == STAT_BY_SUB_PART_TYPE:
        coefficient = _part_term(node.get("mSubpart"), context, seen)
        if isinstance(coefficient, Unevaluable):
            return coefficient
        return _stat_term(node, coefficient)

    if node_type == SUM_OF_SUB_PARTS_TYPE:
        terms: list[Term] = []
        for subpart in node.get("mSubparts") or []:
            term = _part_term(subpart, context, seen)
            if isinstance(term, Unevaluable):
                return term
            terms.append(term)
        return SumTerm(terms=tuple(terms))

    if node_type == PRODUCT_OF_SUB_PARTS_TYPE:
        factors: list[Term] = []
        for key in ("mPart1", "mPart2"):
            term = _part_term(node.get(key), context, seen)
            if isinstance(term, Unevaluable):
                return term
            factors.append(term)
        return ProductTerm(terms=tuple(factors))

    if node_type == INTERPOLATION_TYPE:
        return ByLevelTerm(
            start=float(node.get("mStartValue", 0.0)), end=float(node.get("mEndValue", 0.0))
        )

    if node_type == BREAKPOINTS_TYPE:
        return _breakpoints_term(node)

    return Unevaluable(node_type=node_type, reason="unsupported part type")


def _breakpoints_term(node: Mapping[str, Any]) -> Term | Unevaluable:
    """Reduce a by-character-level breakpoints part to its structure.

    Args:
        node: A ByCharLevelBreakpointsCalculationPart object.

    Returns:
        A BreakpointsTerm carrying the level-one value and every step, or an
        Unevaluable when a step is not a recognised Breakpoint. The per-level
        arithmetic is left to the caller: this module resolves structure, not
        one champion level.
    """
    steps: list[BreakpointStep] = []
    for entry in node.get("mBreakpoints") or []:
        if not isinstance(entry, dict) or entry.get("__type") != BREAKPOINT_TYPE:
            return Unevaluable(node_type=BREAKPOINTS_TYPE, reason="breakpoint is not a Breakpoint")
        blocked = _unrecognised_field(entry, BREAKPOINT_TYPE)
        if blocked is not None:
            return blocked
        steps.append(
            BreakpointStep(
                level=int(entry.get("mLevel", 0)),
                additional_bonus=float(entry.get("mAdditionalBonusAtThisLevel", 0.0)),
                bonus_per_level_after=float(entry.get("mBonusPerLevelAtAndAfter", 0.0)),
            )
        )
    return BreakpointsTerm(
        level1_value=float(node.get("mLevel1Value", 0.0)),
        initial_bonus_per_level=float(node.get("mInitialBonusPerLevel", 0.0)),
        steps=tuple(steps),
    )


def _stat_term(node: Mapping[str, Any], coefficient: Term) -> StatTerm:
    """Wrap a coefficient in the champion stat it scales with.

    Args:
        node: A stat-scaling calculation part carrying mStat and mStatFormula.
        coefficient: Term the stat multiplies.

    Returns:
        A StatTerm whose stat and stat_formula are None when the source enum has
        no proven meaning. An absent mStat is zero, which is Ability Power. An
        absent mStatFormula is zero, which the data value names date as the total
        stat: no value named "Total...Ratio" ever sets the field, and 56 of the
        60 values named for a bonus stat set it to two. Enum value one appears
        three times with no naming evidence either way and stays undecided.
    """
    return StatTerm(
        coefficient=coefficient,
        stat=SCALING_STAT_BY_CODE.get(node.get("mStat", 0)),
        stat_formula=STAT_FORMULA_BY_CODE.get(node.get("mStatFormula", 0)),
    )


def _data_value_term(name: str, context: _Context) -> Term | Unevaluable:
    """Resolve a named data value to a per-rank constant.

    Args:
        name: Data value name the part refers to.
        context: Evaluation context.

    Returns:
        A ConstantTerm holding the value at each rank, or an Unevaluable when
        the spell publishes no value of that name. A name with no entry is a
        stale reference, not a defaulted field, and reading it as zero would put
        a wrong number in the corpus.
    """
    if name not in context.data_values:
        return Unevaluable(node_type=NAMED_DATA_VALUE_TYPE, reason=f"no data value {name}")
    return _array_term(
        context.data_values[name], context, NAMED_DATA_VALUE_TYPE, f"data value {name}"
    )


def _effect_value_term(index: int, context: _Context) -> Term | Unevaluable:
    """Resolve a spell effect index to a per-rank constant.

    Args:
        index: One-based effect index the part refers to.
        context: Evaluation context.

    Returns:
        A ConstantTerm holding the effect value at each rank, or an Unevaluable
        when the caller supplied no array at that index.
    """
    if index not in context.effect_values:
        return Unevaluable(node_type=EFFECT_VALUE_TYPE, reason=f"no effect value {index}")
    return _array_term(
        context.effect_values[index], context, EFFECT_VALUE_TYPE, f"effect value {index}"
    )


def _array_term(
    values: Sequence[float] | None, context: _Context, node_type: str, label: str
) -> Term | Unevaluable:
    """Cut one padded source array down to a per-rank constant.

    Args:
        values: The raw seven-wide one-indexed source array, or None for an
            array the bin publishes as null.
        context: Evaluation context.
        node_type: Bin __type reported when the array is unusable.
        label: Phrase naming the array in the unevaluable reason.

    Returns:
        A ConstantTerm holding the value at each rank. A null array is a
        defaulted field and reads as zero everywhere. An array too short to
        cover every rank is refused rather than padded.
    """
    if values is None:
        return _constant(0.0, context)
    if len(values) < DATA_VALUE_OFFSET + context.ranks:
        return Unevaluable(node_type=node_type, reason=f"short {label}")
    return ConstantTerm(values=tuple(slice_ranks(values, context.max_rank, DATA_VALUE_OFFSET)))


def _constant(value: float, context: _Context) -> ConstantTerm:
    """Spread one rank-invariant number across every rank.

    Args:
        value: The number.
        context: Evaluation context supplying the rank count.

    Returns:
        A ConstantTerm repeating the number once per rank.
    """
    return ConstantTerm(values=(value,) * context.ranks)


def _unrecognised_field(node: Mapping[str, Any], node_type: str) -> Unevaluable | None:
    """Refuse a node carrying a field this module has never decoded.

    Args:
        node: The calculation or calculation part object.
        node_type: The node's bin __type.

    Returns:
        An Unevaluable naming the first unrecognised field, or None when every
        field is one this module either reads or has decided is display-only. A
        field nobody has decoded can change the number, so a node carrying one
        is refused rather than evaluated on the fields that are understood.
    """
    recognised = RECOGNISED_FIELDS.get(node_type, frozenset())
    for key in node:
        if key != "__type" and key not in recognised:
            return Unevaluable(node_type=node_type, reason=f"unrecognised field {key}")
    return None
