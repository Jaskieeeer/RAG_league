import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lolrag.ingest.formatting import format_term
from lolrag.ingest.formulas import ConstantTerm, Formula, Term, evaluate_calculation
from lolrag.ingest.values import DATA_VALUE_OFFSET, slice_ranks

APPEND_TOKEN = "SpellModifierDescriptionAppend"

MODIFIED_REFERENCE_KEY = "mModifiedGameCalculation"
DISPLAY_AS_PERCENT_KEY = "mDisplayAsPercent"
PRECISION_KEY = "mPrecision"

TOKEN_PATTERN = re.compile(r"@([^@]+)@")
EXPRESSION_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\.(?P<decimals>\d+))?"
    r"(?:\*(?P<multiplier>-?\d+(?:\.\d+)?))?$"
)
EFFECT_PATTERN = re.compile(r"^Effect(\d+)Amount$")


# ---------- token grammar ----------


@dataclass(frozen=True)
class TokenExpression:
    """One tooltip token taken apart.

    Args:
        name: Base name the token resolves against.
        decimals: Exact number of decimal places the token asks for, or None
            when it asks for none. The corpus writes this as a dot suffix such
            as "@f15.2@s", which the engine also puts on values that have no
            per-rank array at all, so it is a display directive and never an
            index.
        multiplier: Constant the resolved value is multiplied by, 1.0 when the
            token states none. The corpus uses 100 and -100 to turn a stored
            fraction into a percentage magnitude, and also -1, 4 and 0.5, so it
            is plain arithmetic rather than a percentage marker.
    """

    name: str
    decimals: int | None
    multiplier: float


def parse_token(token: str) -> TokenExpression | None:
    """Take one tooltip token apart into a name, a decimal count and a factor.

    Args:
        token: Token text without its surrounding "@", e.g.
            "RCooldownReduction.0*100".

    Returns:
        The parsed expression, or None when the token is not in the grammar this
        corpus uses. The cross-spell form "@spell.GnarQ:SlowAmount@" and the
        engine's computed "@f1@" placeholders are deliberately outside it: the
        first names a value belonging to another spell and the second names a
        number the client works out at runtime, and neither can be answered from
        the spell in hand.
    """
    match = EXPRESSION_PATTERN.match(token)
    if match is None:
        return None
    decimals = match.group("decimals")
    multiplier = match.group("multiplier")
    return TokenExpression(
        name=match.group("name"),
        decimals=None if decimals is None else int(decimals),
        multiplier=1.0 if multiplier is None else float(multiplier),
    )


# ---------- spell context ----------


@dataclass(frozen=True)
class SpellContext:
    """Everything one spell publishes that a tooltip token can name.

    Args:
        data_values: Data value name to the raw seven-wide one-indexed source
            array, or None for an array the bin publishes as null.
        calculations: The spell's mSpellCalculations object.
        effect_values: mEffectAmount arrays by one-based effect index, in the
            same raw shape as the data values.
        max_rank: Number of learnable ranks, or None for a passive.
    """

    data_values: Mapping[str, Sequence[float] | None]
    calculations: Mapping[str, Any]
    effect_values: Mapping[int, Sequence[float] | None]
    max_rank: int | None


def spell_context(spell: Mapping[str, Any], max_rank: int | None) -> SpellContext:
    """Read the token-resolvable content out of one bin SpellObject.

    Args:
        spell: The spell's parsed SpellObject, whose "mSpell" key holds the
            payload.
        max_rank: Number of learnable ranks, or None for a passive.

    Returns:
        A SpellContext. A data value name published twice keeps its first
        occurrence, matching the value loader, so the tooltip shows the number
        the corpus stores.
    """
    payload = spell.get("mSpell") or {}
    data_values: dict[str, Sequence[float] | None] = {}
    for entry in payload.get("DataValues") or []:
        data_values.setdefault(entry["name"], entry.get("values"))
    effect_values = {
        index: entry.get("value")
        for index, entry in enumerate(payload.get("mEffectAmount") or [], start=1)
    }
    return SpellContext(
        data_values=data_values,
        calculations=payload.get("mSpellCalculations") or {},
        effect_values=effect_values,
        max_rank=max_rank,
    )


# ---------- resolution ----------


@dataclass(frozen=True)
class TokenBlocked:
    """The reason a token, and therefore its whole tooltip, could not be filled.

    Args:
        token: Token text without its surrounding "@".
        reason: Short machine-stable phrase naming what blocked it.
    """

    token: str
    reason: str


@dataclass(frozen=True)
class _DisplayIntent:
    """How the source asks for a calculation's result to be shown.

    Args:
        percent: Whether the result is a fraction to show as a percentage.
        decimals: Exact number of decimal places, or None when the source states
            none.
    """

    percent: bool
    decimals: int | None


def _display_intent(name: str, calculations: Mapping[str, Any]) -> _DisplayIntent:
    """Find the display flags a named calculation is shown under.

    Args:
        name: Calculation name the tooltip token uses.
        calculations: The spell's mSpellCalculations object.

    Returns:
        The flags, taken from the first calculation in the reference chain that
        publishes them. A GameCalculationModified never publishes either field
        anywhere in this corpus, the type having no slot for them, so its
        absent flags carry no intent and the calculation it defers to is the
        only place the intent exists. Seven tooltip tokens depend on this:
        Shyvana's dragon-form slow would read "0.3" instead of "30%" without it.
    """
    seen: set[str] = set()
    current = name
    while current not in seen:
        seen.add(current)
        node = calculations.get(current)
        if not isinstance(node, dict):
            break
        if DISPLAY_AS_PERCENT_KEY in node or PRECISION_KEY in node:
            precision = node.get(PRECISION_KEY, 0)
            return _DisplayIntent(
                percent=bool(node.get(DISPLAY_AS_PERCENT_KEY, False)),
                decimals=int(precision) if precision >= 1 else None,
            )
        reference = node.get(MODIFIED_REFERENCE_KEY)
        if not isinstance(reference, str):
            break
        current = reference
    return _DisplayIntent(percent=False, decimals=None)


def _array_term(values: Sequence[float] | None, context: SpellContext) -> Term | None:
    """Cut one padded source array down to a per-rank constant term.

    Args:
        values: The raw seven-wide one-indexed source array, or None for an
            array the bin publishes as null.
        context: The owning spell's context.

    Returns:
        A ConstantTerm holding the value at each rank, or None when the array is
        too short to cover every rank. A null array is a defaulted field and
        reads as zero everywhere, these bins omitting anything at its default.
    """
    ranks = 1 if context.max_rank is None else context.max_rank
    if values is None:
        return ConstantTerm(values=(0.0,) * ranks)
    if len(values) < DATA_VALUE_OFFSET + ranks:
        return None
    return ConstantTerm(values=tuple(slice_ranks(values, context.max_rank, DATA_VALUE_OFFSET)))


def render_token(token: str, context: SpellContext) -> str | TokenBlocked:
    """Resolve one tooltip token to the text that replaces it.

    Args:
        token: Token text without its surrounding "@".
        context: The owning spell's context.

    Returns:
        The replacement text, or a TokenBlocked naming what stopped it. The name
        is looked for as a stored data value first, then as a sibling
        GameCalculation, then as an mEffectAmount entry for an "EffectNAmount"
        name; the corpus has no name in two of those places at once, so the
        order settles nothing it does not have to.
    """
    if token == APPEND_TOKEN:
        return ""

    expression = parse_token(token)
    if expression is None:
        return TokenBlocked(token=token, reason="token outside the grammar")

    percent = False
    decimals = expression.decimals

    if expression.name in context.data_values:
        term = _array_term(context.data_values[expression.name], context)
        if term is None:
            return TokenBlocked(token=token, reason=f"short data value {expression.name}")
    elif expression.name in context.calculations:
        result = evaluate_calculation(
            expression.name,
            context.calculations,
            context.data_values,
            context.max_rank,
            context.effect_values,
        )
        if not isinstance(result, Formula):
            return TokenBlocked(token=token, reason=f"{result.node_type}: {result.reason}")
        term = result.term
        intent = _display_intent(expression.name, context.calculations)
        percent = intent.percent
        if decimals is None:
            decimals = intent.decimals
    else:
        effect = EFFECT_PATTERN.match(expression.name)
        if effect is None or int(effect.group(1)) not in context.effect_values:
            return TokenBlocked(token=token, reason=f"unknown name {expression.name}")
        term = _array_term(context.effect_values[int(effect.group(1))], context)
        if term is None:
            return TokenBlocked(token=token, reason=f"short effect value {expression.name}")

    text = format_term(term, scale=expression.multiplier, percent=percent, decimals=decimals)
    if text is None:
        return TokenBlocked(token=token, reason="formula has no single-line form")
    return text


def substitute_tooltip(text: str, context: SpellContext) -> str | TokenBlocked:
    """Replace every token in one tooltip with the number it names.

    Args:
        text: Community Dragon dynamicDescription for the spell.
        context: The owning spell's context.

    Returns:
        The tooltip with every token replaced, or the first TokenBlocked that
        stopped one of them. The whole tooltip falls back rather than any part
        of it: a tooltip carrying three real numbers and one surviving token
        reads as finished text and would put a placeholder into the corpus.
    """
    replacements: list[str] = []
    for match in TOKEN_PATTERN.finditer(text):
        rendered = render_token(match.group(1), context)
        if isinstance(rendered, TokenBlocked):
            return rendered
        replacements.append(rendered)
    replaced = iter(replacements)
    return TOKEN_PATTERN.sub(lambda _: next(replaced), text)
