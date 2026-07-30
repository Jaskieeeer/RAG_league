import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from lolrag.config import get_settings
from lolrag.fetch import cdragon, cdragon_bin, ddragon
from lolrag.fetch.client import FetchClient
from lolrag.ingest.identifiers import PASSIVE_SLOT, SPELL_SLOTS
from lolrag.ingest.tooltips import (
    APPEND_TOKEN,
    TOKEN_PATTERN,
    TokenBlocked,
    TokenExpression,
    parse_token,
    render_token,
    spell_context,
    substitute_tooltip,
)
from lolrag.ingest.values import build_ability_values, group_spells, spell_key

MAX_RANK = 5
Q_BASE_DAMAGE = [-5.0, 10.0, 25.0, 40.0, 55.0, 70.0, 85.0]
SLOW_PERCENT = [0.800000011920929] * 7


# ---------- fixtures ----------


def spell(
    *,
    data_values: dict[str, list[float] | None] | None = None,
    calculations: dict[str, Any] | None = None,
    effect_values: list[list[float] | None] | None = None,
) -> dict[str, Any]:
    """Build one bin SpellObject publishing exactly what is given.

    Args:
        data_values: Data value name to its raw seven-wide source array.
        calculations: The spell's mSpellCalculations object.
        effect_values: Raw mEffectAmount arrays in one-based effect order.

    Returns:
        A SpellObject. Anything left out is a field the bin omits because it
        sits at its default.
    """
    payload: dict[str, Any] = {}
    if data_values is not None:
        payload["DataValues"] = [
            {"name": name, "values": values, "__type": "SpellDataValue"}
            for name, values in data_values.items()
        ]
    if calculations is not None:
        payload["mSpellCalculations"] = calculations
    if effect_values is not None:
        payload["mEffectAmount"] = [
            {"value": values, "__type": "SpellEffectAmount"} if values is not None else {}
            for values in effect_values
        ]
    return {"mSpell": payload, "__type": "SpellObject"}


def calculation(*parts: dict[str, Any], **fields: Any) -> dict[str, Any]:
    """Build a GameCalculation summing the given parts.

    Args:
        parts: The parts of mFormulaParts, in source order.
        fields: Further fields such as mDisplayAsPercent or mPrecision.

    Returns:
        A GameCalculation object.
    """
    return {"mFormulaParts": list(parts), **fields, "__type": "GameCalculation"}


def data_value_part(name: str) -> dict[str, Any]:
    """Build a formula part naming one data value.

    Args:
        name: Data value name.

    Returns:
        A NamedDataValueCalculationPart object.
    """
    return {"mDataValue": name, "__type": "NamedDataValueCalculationPart"}


def modified(reference: str, multiplier: float) -> dict[str, Any]:
    """Build a GameCalculationModified scaling another calculation by a number.

    Args:
        reference: Name of the calculation this one defers to.
        multiplier: Constant factor.

    Returns:
        A GameCalculationModified object.
    """
    return {
        "mModifiedGameCalculation": reference,
        "mMultiplier": {"mNumber": multiplier, "__type": "NumberCalculationPart"},
        "__type": "GameCalculationModified",
    }


def context(spell_object: dict[str, Any], max_rank: int | None = MAX_RANK):
    """Build the spell context a tooltip resolves against.

    Args:
        spell_object: The bin SpellObject.
        max_rank: Number of learnable ranks, or None for a passive.

    Returns:
        The SpellContext.
    """
    return spell_context(spell_object, max_rank)


# ---------- token grammar ----------


def test_a_bare_name_is_the_whole_expression() -> None:
    """The commonest token names a value and asks for nothing else."""
    assert parse_token("BaseDamage") == TokenExpression(
        name="BaseDamage", decimals=None, multiplier=1.0
    )


def test_a_token_can_multiply_by_a_hundred() -> None:
    """The tooltip turns a stored fraction into a percentage magnitude itself."""
    assert parse_token("SlowPercent*100") == TokenExpression(
        name="SlowPercent", decimals=None, multiplier=100.0
    )


def test_a_token_can_multiply_by_a_negative_number() -> None:
    """A slow stored as a negative modifier is flipped for display."""
    assert parse_token("WSlowPercentage*-100") == TokenExpression(
        name="WSlowPercentage", decimals=None, multiplier=-100.0
    )


def test_a_token_can_multiply_by_a_fraction() -> None:
    """Yone's W splits one base damage into two halves, so the factor is 0.5."""
    assert parse_token("BaseDamage*0.5") == TokenExpression(
        name="BaseDamage", decimals=None, multiplier=0.5
    )


def test_a_dot_suffix_asks_for_a_number_of_decimal_places() -> None:
    """The corpus puts this suffix on engine scalars with no rank array at all,
    which no index reading survives: it is a display directive."""
    assert parse_token("MaxDuration.1") == TokenExpression(
        name="MaxDuration", decimals=1, multiplier=1.0
    )


def test_a_dot_suffix_and_a_multiplier_can_appear_together() -> None:
    """K'Sante's Q asks for a third, times a hundred, at zero decimals."""
    assert parse_token("RCooldownReduction.0*100") == TokenExpression(
        name="RCooldownReduction", decimals=0, multiplier=100.0
    )


def test_a_cross_spell_reference_is_outside_the_grammar() -> None:
    """It names a value belonging to another spell, which this spell cannot answer."""
    assert parse_token("spell.GnarQ:SlowAmount") is None
    assert parse_token("Spell.GlacialStorm:SlowAmount*100") is None


def test_a_token_that_is_not_an_expression_is_outside_the_grammar() -> None:
    """Anything the corpus never writes is refused rather than half parsed."""
    assert parse_token("") is None
    assert parse_token("1Damage") is None
    assert parse_token("BaseDamage+10") is None


# ---------- resolution ----------


def test_the_append_token_resolves_to_nothing() -> None:
    """Every spell carries it and none of them means anything by it."""
    assert render_token(APPEND_TOKEN, context(spell())) == ""


def test_a_data_value_resolves_to_its_per_rank_numbers() -> None:
    """The padded source array is cut down to the ranks the ability has."""
    resolved = render_token(
        "QBaseDamage", context(spell(data_values={"QBaseDamage": Q_BASE_DAMAGE}))
    )

    assert resolved == "10/25/40/55/70"


def test_a_data_value_with_a_null_array_resolves_to_zero() -> None:
    """These bins omit a field at its default, so a null array is zero everywhere."""
    assert render_token("Unused", context(spell(data_values={"Unused": None}))) == "0"


def test_a_data_value_is_multiplied_by_the_token_factor() -> None:
    """K'Sante's slow is stored as 0.8 and the tooltip asks for it as 80."""
    resolved = render_token(
        "SlowPercent*100", context(spell(data_values={"SlowPercent": SLOW_PERCENT}))
    )

    assert resolved == "80"


def test_a_calculation_resolves_through_the_formula_evaluator() -> None:
    """A name the spell publishes as a calculation is evaluated, not looked up."""
    spell_object = spell(
        data_values={"QBaseDamage": Q_BASE_DAMAGE},
        calculations={
            "QDamage": calculation(
                data_value_part("QBaseDamage"),
                {"mCoefficient": 0.6, "mStat": 2, "__type": "StatByCoefficientCalculationPart"},
            )
        },
    )

    assert render_token("QDamage", context(spell_object)) == "10/25/40/55/70 (+60% total AD)"


def test_a_data_value_is_looked_for_before_a_calculation_of_the_same_name() -> None:
    """The stored number wins, which is what the corpus itself stores for that name."""
    spell_object = spell(
        data_values={"Damage": Q_BASE_DAMAGE},
        calculations={"Damage": calculation({"mNumber": 999.0, "__type": "NumberCalculationPart"})},
    )

    assert render_token("Damage", context(spell_object)) == "10/25/40/55/70"


def test_an_effect_amount_resolves_when_the_name_matches_the_effect_pattern() -> None:
    """Older tooltips name the spell's raw effect slots rather than a data value."""
    spell_object = spell(effect_values=[None, Q_BASE_DAMAGE])

    assert render_token("Effect2Amount", context(spell_object)) == "10/25/40/55/70"


def test_an_effect_amount_the_spell_does_not_publish_blocks() -> None:
    """An index past the end of mEffectAmount names nothing and is not guessed."""
    blocked = render_token("Effect3Amount", context(spell(effect_values=[Q_BASE_DAMAGE])))

    assert blocked == TokenBlocked(token="Effect3Amount", reason="unknown name Effect3Amount")


def test_a_name_the_spell_publishes_nowhere_blocks() -> None:
    """The engine computes "@f1@" at runtime and the bin cannot answer it."""
    assert render_token("f1", context(spell())) == TokenBlocked(
        token="f1", reason="unknown name f1"
    )


def test_an_unevaluable_calculation_blocks_and_reports_the_evaluator_reason() -> None:
    """The blocking node is named so the unsupported tail stays countable."""
    spell_object = spell(
        calculations={"Damage": calculation({"__type": "CooldownMultiplierCalculationPart"})}
    )

    assert render_token("Damage", context(spell_object)) == TokenBlocked(
        token="Damage", reason="CooldownMultiplierCalculationPart: unsupported part type"
    )


def test_a_formula_with_no_single_line_form_blocks() -> None:
    """A stat times a stat is a curve, and half of it would read as all of it."""
    spell_object = spell(
        calculations={
            "Damage": calculation(
                {
                    "mPart1": {
                        "mCoefficient": 1.0,
                        "mStat": 8,
                        "__type": "StatByCoefficientCalculationPart",
                    },
                    "mPart2": {
                        "mCoefficient": 1.0,
                        "mStat": 2,
                        "__type": "StatByCoefficientCalculationPart",
                    },
                    "__type": "ProductOfSubPartsCalculationPart",
                }
            )
        }
    )

    assert render_token("Damage", context(spell_object)) == TokenBlocked(
        token="Damage", reason="formula has no single-line form"
    )


# ---------- display intent ----------


def test_a_calculation_flagged_as_a_percentage_is_shown_as_one() -> None:
    """The source states the fraction and asks for the percentage."""
    spell_object = spell(
        data_values={"SlowAmount": [0.0] + [0.3] * 6},
        calculations={"Slow": calculation(data_value_part("SlowAmount"), mDisplayAsPercent=True)},
    )

    assert render_token("Slow", context(spell_object)) == "30%"


def test_a_modified_calculation_inherits_the_percentage_it_defers_to() -> None:
    """A GameCalculationModified has no slot for the flag anywhere in this corpus,
    so its absence carries no intent and the base calculation holds the only one."""
    spell_object = spell(
        data_values={"SlowAmount": [0.0] + [0.3] * 6},
        calculations={
            "Slow": calculation(data_value_part("SlowAmount"), mDisplayAsPercent=True),
            "DragonSlow": modified("Slow", 1.0),
        },
    )

    assert render_token("DragonSlow", context(spell_object)) == "30%"


def test_a_published_precision_sets_the_decimal_places() -> None:
    """Precision is published only where a fraction matters, and then it is honoured."""
    spell_object = spell(
        data_values={"HealthDamage": [0.0] + [0.06250000093132257] * 6},
        calculations={
            "Damage": calculation(
                data_value_part("HealthDamage"), mDisplayAsPercent=True, mPrecision=1
            )
        },
    )

    assert render_token("Damage", context(spell_object)) == "6.3%"


def test_the_token_decimal_count_beats_the_calculations_precision() -> None:
    """The tooltip states its own intent at the point of use."""
    spell_object = spell(
        data_values={"HealthDamage": [0.0] + [0.06250000093132257] * 6},
        calculations={
            "Damage": calculation(
                data_value_part("HealthDamage"), mDisplayAsPercent=True, mPrecision=1
            )
        },
    )

    assert render_token("Damage.0", context(spell_object)) == "6%"


# ---------- substitution ----------


def test_every_token_in_a_tooltip_is_replaced() -> None:
    """The K'Sante shape: a data value, a scaled fraction and the append token."""
    spell_object = spell(
        data_values={
            "BaseDamage": Q_BASE_DAMAGE,
            "SlowPercent": SLOW_PERCENT,
            "SlowDuration": [0.0] + [0.5] * 6,
        }
    )
    text = (
        "K'Sante slams, dealing @BaseDamage@ damage and Slowing by "
        "@SlowPercent*100@% for @SlowDuration@s.@SpellModifierDescriptionAppend@"
    )

    assert substitute_tooltip(text, context(spell_object)) == (
        "K'Sante slams, dealing 10/25/40/55/70 damage and Slowing by 80% for 0.5s."
    )


def test_one_unresolvable_token_falls_the_whole_tooltip_back() -> None:
    """Three real numbers and one surviving token would read as finished text."""
    spell_object = spell(data_values={"BaseDamage": Q_BASE_DAMAGE})
    text = "Deals @BaseDamage@ damage over @f1@ seconds.@SpellModifierDescriptionAppend@"

    assert substitute_tooltip(text, context(spell_object)) == TokenBlocked(
        token="f1", reason="unknown name f1"
    )


def test_a_tooltip_with_no_tokens_comes_back_unchanged() -> None:
    """Substitution never rewrites the words around the numbers."""
    assert substitute_tooltip("Aatrox dashes.", context(spell())) == "Aatrox dashes."


def test_a_repeated_token_is_replaced_at_every_occurrence() -> None:
    """Yone's W names the same halved base damage twice."""
    spell_object = spell(data_values={"BaseDamage": Q_BASE_DAMAGE})

    assert substitute_tooltip(
        "@BaseDamage*0.5@ physical and @BaseDamage*0.5@ magic", context(spell_object)
    ) == ("5/12.5/20/27.5/35 physical and 5/12.5/20/27.5/35 magic")


def test_a_passive_takes_the_rank_one_entry_of_every_array() -> None:
    """A passive publishes no rank count, so each value carries one number."""
    spell_object = spell(data_values={"BaseDamage": Q_BASE_DAMAGE})

    assert substitute_tooltip("@BaseDamage@", context(spell_object, max_rank=None)) == "10"


# ---------- the stored-value control ----------

CACHE_DIR = Path(get_settings().cache_dir)

CONTROL_SPELLS = 62
CONTROL_TOKENS = 156
NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
RELATIVE_TOLERANCE = 2e-3


def _expected_numbers(values: Sequence[float], multiplier: float) -> list[float]:
    """Scale a stored value array the way a token asks for it, collapsing a flat one.

    Args:
        values: The per-rank numbers the value loader stores.
        multiplier: The token's own factor.

    Returns:
        The numbers the substituted text must contain, in order.
    """
    scaled = [value * multiplier for value in values]
    if len({round(value, 6) for value in scaled}) == 1:
        return scaled[:1]
    return scaled


@pytest.mark.corpus
@pytest.mark.skipif(not CACHE_DIR.is_dir(), reason=f"no warm corpus cache at {CACHE_DIR}")
async def test_every_tooltip_built_only_from_stored_data_values_shows_those_values() -> None:
    """Where a whole tooltip is stored numbers, substitution is checkable without
    any domain knowledge: every number it prints must be a number the loader stores."""
    settings = get_settings()
    async with FetchClient(settings) as client:
        champion_list = await ddragon.fetch_champion_list(client, settings)
        ddragon_ids = list(champion_list["data"])
        champion_keys = [int(entry["key"]) for entry in champion_list["data"].values()]
        details = await ddragon.fetch_all_champion_details(client, settings, ddragon_ids)
        bins = await cdragon_bin.fetch_all_champion_bins(client, settings, ddragon_ids)
        records = await cdragon.fetch_all_champions(client, settings, champion_keys)

    slots = {slot: index for index, slot in enumerate((PASSIVE_SLOT, *SPELL_SLOTS))}
    checked_spells = 0
    checked_tokens = 0
    mismatches: list[str] = []

    for ddragon_id, champion_key in zip(ddragon_ids, champion_keys, strict=True):
        bin_payload = bins[ddragon_id]
        record = records[champion_key]
        rows = build_ability_values(ddragon_id, details[ddragon_id], bin_payload, record, slots)
        groups = {
            group.slot: group
            for group in group_spells(ddragon_id, details[ddragon_id], bin_payload)
        }

        for slot, source_spell in zip(SPELL_SLOTS, record["spells"], strict=True):
            group = groups[slot]
            root_key = spell_key(group.root_key)
            stored = {
                row.name: row
                for row in rows
                if row.spell_key == root_key and row.kind != "by_level"
            }
            text = source_spell["dynamicDescription"]
            tokens = [token for token in TOKEN_PATTERN.findall(text) if token != APPEND_TOKEN]
            expressions = [parse_token(token) for token in tokens]
            if any(
                expression is None or expression.name not in stored for expression in expressions
            ):
                continue

            checked_spells += 1
            spell_ctx = spell_context(bin_payload[group.root_key], group.max_rank)
            if isinstance(substitute_tooltip(text, spell_ctx), TokenBlocked):
                mismatches.append(f"{ddragon_id} {slot}: tooltip fell back")
                continue
            for token, expression in zip(tokens, expressions, strict=True):
                checked_tokens += 1
                rendered = render_token(token, spell_ctx)
                shown = [float(match) for match in NUMBER_PATTERN.findall(str(rendered))]
                expected = _expected_numbers(stored[expression.name].values, expression.multiplier)
                if len(shown) != len(expected) or any(
                    abs(left - right) > max(1e-9, abs(right) * RELATIVE_TOLERANCE)
                    for left, right in zip(shown, expected, strict=True)
                ):
                    mismatches.append(
                        f"{ddragon_id} {slot} @{token}@ showed {shown} for stored {expected}"
                    )

    assert mismatches == []
    assert (checked_spells, checked_tokens) == (CONTROL_SPELLS, CONTROL_TOKENS)
