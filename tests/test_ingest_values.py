from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from lolrag.config import get_settings
from lolrag.db.models import Ability, AbilityValue, Champion, Faction, Item, ItemValue
from lolrag.ingest.values import (
    ValueLoadStats,
    build_ability_values,
    build_item_values,
    group_spells,
    load_values,
    parse_damage_types,
    passive_spell_key,
    resolve_value_attributes,
    slice_ranks,
    spell_key,
)
from tests.test_fetch_client import build_client, build_settings

ABILITY_IDS = {"P": 1, "Q": 2, "W": 3, "E": 4, "R": 5}
SPELL_SLOT_IDS = ("Q", "W", "E", "R")


# ---------- bin fixture builders ----------


def spell_object(
    *,
    data_values: Mapping[str, list[float] | None] | None = None,
    cooldown: list[float] | None = None,
    mana: list[float] | None = None,
    calculations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one bin SpellObject carrying only the fields the loader reads.

    Args:
        data_values: Value name to its seven-wide one-indexed source array, or
            None for a value the bin publishes with a null array.
        cooldown: Seven-wide one-indexed cooldownTime array, or None.
        mana: Six-wide zero-indexed mana array, or None.
        calculations: The spell's mSpellCalculations object, or None.

    Returns:
        A SpellObject whose mSpell holds exactly the supplied fields, matching
        the bin's habit of omitting anything at its default.
    """
    spell: dict[str, Any] = {"__type": "SpellDataResource"}
    if data_values is not None:
        spell["DataValues"] = [
            {"name": name, "values": values, "__type": "SpellDataValue"}
            for name, values in data_values.items()
        ]
    if cooldown is not None:
        spell["cooldownTime"] = cooldown
        spell["Cooldown"] = {"values": cooldown, "__type": "{0a0eddc9}"}
    if mana is not None:
        spell["mana"] = mana
    if calculations is not None:
        spell["mSpellCalculations"] = dict(calculations)
    return {"__type": "SpellObject", "mSpell": spell}


def champion_bin(
    champion: str,
    ability_spells: Mapping[str, Mapping[str, dict[str, Any]]],
    *,
    passive_key: str,
    loose_spells: Mapping[str, dict[str, Any]] | None = None,
    extra_records: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a champion bin file from ability folders and their spells.

    Args:
        champion: Bin champion folder name, e.g. "Aatrox".
        ability_spells: Ability folder name to an ordered mapping of spell name
            to SpellObject; the first spell of each folder is its root.
        passive_key: Full bin key the root CharacterRecord points its
            mCharacterPassiveSpell at.
        loose_spells: Spell name to SpellObject for spells that hang directly
            off the champion's spell folder with no AbilityObject parent.
        extra_records: Suffixes of additional CharacterRecords, e.g. "URF",
            reproducing champions that publish more than one record.

    Returns:
        A bin payload shaped like a real Community Dragon champion bin file.
    """
    passive_record = {"__type": "CharacterRecord", "mCharacterPassiveSpell": passive_key}
    payload: dict[str, Any] = {
        f"Characters/{champion}/CharacterRecords/{suffix}": dict(passive_record)
        for suffix in extra_records
    }
    payload[f"Characters/{champion}/CharacterRecords/Root"] = passive_record
    for folder, spells in ability_spells.items():
        base = f"Characters/{champion}/Spells/{folder}"
        names = list(spells)
        payload[base] = {
            "__type": "AbilityObject",
            "mName": folder,
            "mRootSpell": f"{base}/{names[0]}",
            "mChildSpells": [f"{base}/{name}" for name in names],
        }
        for name, spell in spells.items():
            payload[f"{base}/{name}"] = spell
    for name, spell in (loose_spells or {}).items():
        payload[f"Characters/{champion}/Spells/{name}"] = spell
    return payload


def ddragon_detail(champion: str, spells: Sequence[tuple[str, int]]) -> dict[str, Any]:
    """Build a Data Dragon champion detail body naming four spells in slot order.

    Args:
        champion: Data Dragon champion id the body is keyed on.
        spells: The four (spell id, maxrank) pairs, in Q, W, E, R slot order.

    Returns:
        Payload whose "data" key maps champion to a record with "passive" and a
        four-entry "spells" list.
    """
    return {
        "data": {
            champion: {
                "id": champion,
                "passive": {"name": f"{champion} Passive", "description": "Passive."},
                "spells": [
                    {
                        "id": spell_id,
                        "name": spell_id,
                        "description": "Spell.",
                        "tooltip": "Spell tooltip.",
                        "maxrank": max_rank,
                    }
                    for spell_id, max_rank in spells
                ],
            }
        }
    }


def cdragon_record(descriptions: Sequence[str | None] = (None, None, None, None)) -> dict[str, Any]:
    """Build a Community Dragon champion record carrying four dynamic descriptions.

    Args:
        descriptions: The four dynamicDescription strings, in Q, W, E, R slot
            order; None for a spell whose tooltip names no damage type.

    Returns:
        Payload whose "spells" list holds one entry per slot. The passive has no
        dynamicDescription in the source and none is published here.
    """
    return {
        "passive": {"name": "Passive", "description": "Passive."},
        "spells": [
            {"spellKey": key, "name": key.upper(), "dynamicDescription": description}
            for key, description in zip("qwer", descriptions, strict=True)
        ],
    }


# ---------- champion fixtures ----------

AATROX_Q_COOLDOWN = [14.0, 14.0, 12.0, 10.0, 8.0, 6.0, 4.0]
AATROX_Q2_COOLDOWN = [16.0, 16.0, 15.0, 14.0, 13.0, 12.0, 12.0]
AATROX_Q_BASE_DAMAGE = [-5.0, 10.0, 25.0, 40.0, 55.0, 70.0, 85.0]
AATROX_Q_AD_RATIO = [0.525, 0.6, 0.675, 0.75, 0.825, 0.9, 0.975]
AATROX_Q_DESCRIPTION = (
    "Aatrox slams his greatsword, dealing "
    "<physicalDamage>@QDamage@ physical damage</physicalDamage>."
)

ANNIE_Q_BASE_DAMAGE = [35.0, 80.0, 125.0, 170.0, 215.0, 260.0, 305.0]
ANNIE_Q_AP_RATIO = [0.8] * 7
ANNIE_Q_COOLDOWN = [4.0] * 7
ANNIE_Q_MANA = [60.0, 65.0, 70.0, 75.0, 80.0, 85.0]
ANNIE_Q_DESCRIPTION = (
    "Annie hurls a fireball, dealing <magicDamage>@TotalDamage@ magic damage</magicDamage>."
)


def aatrox_bin() -> dict[str, Any]:
    """Build Aatrox's bin file with a three-spell Q and an interpolated passive.

    Returns:
        A bin payload where AatroxQ2 and AatroxQ3 publish the recast cooldown
        that must never be mistaken for the ability's own.
    """
    return champion_bin(
        "Aatrox",
        {
            "AatroxPassiveAbility": {
                "AatroxPassive": spell_object(
                    data_values={"PBonusAARange": [50.0] * 7},
                    calculations={
                        "PDamage": {
                            "mFormulaParts": [
                                {
                                    "mStartValue": 0.04,
                                    "mEndValue": 0.10,
                                    "__type": "ByCharLevelInterpolationCalculationPart",
                                }
                            ],
                            "mDisplayAsPercent": True,
                            "__type": "GameCalculation",
                        }
                    },
                )
            },
            "AatroxQAbility": {
                "AatroxQ": spell_object(
                    data_values={
                        "QBaseDamage": AATROX_Q_BASE_DAMAGE,
                        "QTotalADRatio": AATROX_Q_AD_RATIO,
                    },
                    cooldown=AATROX_Q_COOLDOWN,
                    calculations={
                        "QDamage": {
                            "mFormulaParts": [
                                {
                                    "mDataValue": "QBaseDamage",
                                    "__type": "NamedDataValueCalculationPart",
                                },
                                {
                                    "mStat": 2,
                                    "mDataValue": "QTotalADRatio",
                                    "__type": "StatByNamedDataValueCalculationPart",
                                },
                            ],
                            "__type": "GameCalculation",
                        }
                    },
                ),
                "AatroxQ2": spell_object(cooldown=AATROX_Q2_COOLDOWN),
                "AatroxQ3": spell_object(cooldown=AATROX_Q2_COOLDOWN),
            },
            "AatroxWAbility": {"AatroxW": spell_object()},
            "AatroxEAbility": {"AatroxE": spell_object()},
            "AatroxRAbility": {"AatroxR": spell_object()},
        },
        passive_key="Characters/Aatrox/Spells/AatroxPassiveAbility/AatroxPassive",
    )


def aatrox_detail() -> dict[str, Any]:
    """Build Aatrox's Data Dragon detail body.

    Returns:
        Payload whose four spells are AatroxQ, AatroxW and AatroxE at five ranks
        and AatroxR at three.
    """
    return ddragon_detail(
        "Aatrox", [("AatroxQ", 5), ("AatroxW", 5), ("AatroxE", 5), ("AatroxR", 3)]
    )


def annie_bin() -> dict[str, Any]:
    """Build Annie's bin file, whose Q publishes both a cooldown and a mana cost.

    Returns:
        A bin payload whose APRatio scales off an mStat that the source omits,
        which means Ability Power.
    """
    return champion_bin(
        "Annie",
        {
            "AnniePassiveAbility": {"AnniePassive": spell_object()},
            "AnnieQAbility": {
                "AnnieQ": spell_object(
                    data_values={
                        "BaseDamage": ANNIE_Q_BASE_DAMAGE,
                        "APRatio": ANNIE_Q_AP_RATIO,
                    },
                    cooldown=ANNIE_Q_COOLDOWN,
                    mana=ANNIE_Q_MANA,
                    calculations={
                        "TotalDamage": {
                            "mFormulaParts": [
                                {
                                    "mDataValue": "BaseDamage",
                                    "__type": "NamedDataValueCalculationPart",
                                },
                                {
                                    "mDataValue": "APRatio",
                                    "__type": "StatByNamedDataValueCalculationPart",
                                },
                            ],
                            "__type": "GameCalculation",
                        }
                    },
                )
            },
            "AnnieWAbility": {"AnnieW": spell_object()},
            "AnnieEAbility": {"AnnieE": spell_object()},
            "AnnieRAbility": {"AnnieR": spell_object()},
        },
        passive_key="Characters/Annie/Spells/AnniePassiveAbility/AnniePassive",
    )


def annie_detail() -> dict[str, Any]:
    """Build Annie's Data Dragon detail body.

    Returns:
        Payload whose four spells are AnnieQ, AnnieW, AnnieE and AnnieR.
    """
    return ddragon_detail("Annie", [("AnnieQ", 5), ("AnnieW", 5), ("AnnieE", 5), ("AnnieR", 3)])


def naafiri_bin() -> dict[str, Any]:
    """Build Naafiri's bin file, whose spell ids no longer match their slots.

    Returns:
        A bin payload holding NaafiriQ, NaafiriW, NaafiriE and NaafiriR, each
        publishing a data value naming the spell it came from.
    """
    return champion_bin(
        "Naafiri",
        {
            "NaafiriPassiveAbility": {"NaafiriPassive": spell_object()},
            "NaafiriQAbility": {"NaafiriQ": spell_object(data_values={"QMark": [1.0] * 7})},
            "NaafiriWAbility": {"NaafiriW": spell_object(data_values={"WMark": [2.0] * 7})},
            "NaafiriEAbility": {"NaafiriE": spell_object(data_values={"EMark": [3.0] * 7})},
            "NaafiriRAbility": {"NaafiriR": spell_object(data_values={"RMark": [4.0] * 7})},
        },
        passive_key="Characters/Naafiri/Spells/NaafiriPassiveAbility/NaafiriPassive",
    )


def naafiri_detail() -> dict[str, Any]:
    """Build Naafiri's Data Dragon detail body in its published spell order.

    Returns:
        Payload whose spells list is NaafiriQ, NaafiriR, NaafiriE, NaafiriW,
        the rework having swapped W and R while the ids kept the old letters.
    """
    return ddragon_detail(
        "Naafiri", [("NaafiriQ", 5), ("NaafiriR", 5), ("NaafiriE", 5), ("NaafiriW", 3)]
    )


def anivia_bin() -> dict[str, Any]:
    """Build Anivia's bin file, whose spells are named after the ability.

    Returns:
        A bin payload with no champion name in any spell key, so a slot letter
        cannot be pattern-matched out of it.
    """
    return champion_bin(
        "Anivia",
        {
            "AniviaPassiveAbility": {"AniviaPassive": spell_object()},
            "FlashFrostAbility": {"FlashFrost": spell_object()},
            "CrystallizeAbility": {"Crystallize": spell_object()},
            "FrostbiteAbility": {"Frostbite": spell_object()},
            "GlacialStormAbility": {
                "GlacialStorm": spell_object(data_values={"SlowAmount": [0.2] * 7})
            },
        },
        passive_key="Characters/Anivia/Spells/AniviaPassiveAbility/AniviaPassive",
    )


def anivia_detail() -> dict[str, Any]:
    """Build Anivia's Data Dragon detail body.

    Returns:
        Payload whose spells are FlashFrost, Crystallize, Frostbite and
        GlacialStorm.
    """
    return ddragon_detail(
        "Anivia",
        [("FlashFrost", 5), ("Crystallize", 5), ("Frostbite", 5), ("GlacialStorm", 3)],
    )


def udyr_bin() -> dict[str, Any]:
    """Build Udyr's bin file, whose passive hangs off no AbilityObject.

    Returns:
        A bin payload where the passive sits directly in the champion's spell
        folder, so a prefix sweep would swallow every other spell.
    """
    return champion_bin(
        "Udyr",
        {
            "UdyrQAbility": {"UdyrQ": spell_object(data_values={"QDamage": [10.0] * 7})},
            "UdyrWAbility": {"UdyrW": spell_object()},
            "UdyrEAbility": {"UdyrE": spell_object()},
            "UdyrRAbility": {"UdyrR": spell_object()},
        },
        passive_key="Characters/Udyr/Spells/UdyrPassive",
        loose_spells={"UdyrPassive": spell_object(data_values={"PBonusDamage": [5.0] * 7})},
    )


def udyr_detail() -> dict[str, Any]:
    """Build Udyr's Data Dragon detail body.

    Returns:
        Payload whose four spells each rank up six times.
    """
    return ddragon_detail("Udyr", [("UdyrQ", 6), ("UdyrW", 6), ("UdyrE", 6), ("UdyrR", 6)])


def aurelion_sol_bin() -> dict[str, Any]:
    """Build Aurelion Sol's bin file, whose R and R2 both publish BaseDamage.

    Returns:
        A bin payload reproducing the two same-named arrays that only spell_key
        tells apart.
    """
    return champion_bin(
        "AurelionSol",
        {
            "AurelionSolPassiveAbility": {"AurelionSolPassive": spell_object()},
            "AurelionSolQAbility": {"AurelionSolQ": spell_object()},
            "AurelionSolWAbility": {"AurelionSolW": spell_object()},
            "AurelionSolEAbility": {"AurelionSolE": spell_object()},
            "AurelionSolRAbility": {
                "AurelionSolR": spell_object(
                    data_values={"BaseDamage": [50.0, 150.0, 250.0, 350.0, 450.0, 550.0, 650.0]}
                ),
                "AurelionSolR2": spell_object(
                    data_values={"BaseDamage": [125.0, 200.0, 275.0, 350.0, 425.0, 500.0, 575.0]}
                ),
            },
        },
        passive_key="Characters/AurelionSol/Spells/AurelionSolPassiveAbility/AurelionSolPassive",
    )


def aurelion_sol_detail() -> dict[str, Any]:
    """Build Aurelion Sol's Data Dragon detail body.

    Returns:
        Payload whose ultimate ranks up three times.
    """
    return ddragon_detail(
        "AurelionSol",
        [("AurelionSolQ", 5), ("AurelionSolW", 5), ("AurelionSolE", 5), ("AurelionSolR", 3)],
    )


# ---------- helper tests ----------


def test_spell_key_returns_the_last_path_segment() -> None:
    """The spell key is the tail of the bin key, and a hashed key has no tail."""
    assert spell_key("Characters/Aatrox/Spells/AatroxQAbility/AatroxQ") == "AatroxQ"
    assert spell_key("{3e27cfac}") == "{3e27cfac}"


def test_passive_spell_key_reads_the_root_character_record() -> None:
    """The passive comes from the root record even when mode records repeat it."""
    payload = champion_bin(
        "Braum",
        {"BraumPassiveAbility": {"BraumPassive": spell_object()}},
        passive_key="Characters/Braum/Spells/BraumPassiveAbility/BraumPassive",
        extra_records=("URF", "SLIME"),
    )

    assert passive_spell_key(payload) == "Characters/Braum/Spells/BraumPassiveAbility/BraumPassive"


def test_passive_spell_key_rejects_a_bin_without_a_root_record() -> None:
    """A bin publishing no root CharacterRecord fails loudly."""
    with pytest.raises(ValueError):
        passive_spell_key({"Characters/X/Spells/XQ": spell_object()})


def test_slice_ranks_cuts_the_padding_off_both_array_shapes() -> None:
    """A seven-wide array starts at index one and a six-wide mana array at zero."""
    assert slice_ranks(AATROX_Q_COOLDOWN, 5, 1) == [14.0, 12.0, 10.0, 8.0, 6.0]
    assert slice_ranks(ANNIE_Q_MANA, 5, 0) == [60.0, 65.0, 70.0, 75.0, 80.0]


def test_slice_ranks_takes_one_entry_for_a_rankless_passive() -> None:
    """A passive has no max_rank, so only the rank-one slot is kept."""
    assert slice_ranks(AATROX_Q_BASE_DAMAGE, None, 1) == [10.0]
    assert slice_ranks(ANNIE_Q_MANA, None, 0) == [60.0]


# ---------- grouping tests ----------


def test_group_spells_assigns_slots_by_position_not_by_the_letter_in_the_id() -> None:
    """Naafiri's W and R come from array position, so the stale letters do not mislabel them."""
    groups = {
        group.slot: group for group in group_spells("Naafiri", naafiri_detail(), naafiri_bin())
    }

    assert spell_key(groups["W"].root_key) == "NaafiriR"
    assert spell_key(groups["R"].root_key) == "NaafiriW"
    assert groups["R"].max_rank == 3


def test_group_spells_joins_a_champion_whose_spells_are_named_after_the_ability() -> None:
    """Anivia's spell ids carry no champion name yet still resolve against the bin."""
    groups = {group.slot: group for group in group_spells("Anivia", anivia_detail(), anivia_bin())}

    assert spell_key(groups["R"].root_key) == "GlacialStorm"
    assert spell_key(groups["Q"].root_key) == "FlashFrost"


def test_group_spells_collects_the_child_spells_of_an_ability() -> None:
    """Aatrox's Q spans its two recast spells as well as the root."""
    groups = {group.slot: group for group in group_spells("Aatrox", aatrox_detail(), aatrox_bin())}

    assert [spell_key(key) for key in groups["Q"].member_keys] == [
        "AatroxQ",
        "AatroxQ2",
        "AatroxQ3",
    ]
    assert spell_key(groups["Q"].root_key) == "AatroxQ"


def test_group_spells_keeps_a_loose_passive_from_swallowing_its_neighbours() -> None:
    """Udyr's passive sits beside the other spells, so its group holds only itself."""
    groups = {group.slot: group for group in group_spells("Udyr", udyr_detail(), udyr_bin())}

    assert groups["P"].member_keys == ("Characters/Udyr/Spells/UdyrPassive",)
    assert groups["P"].max_rank is None


def test_group_spells_rejects_a_spell_id_absent_from_the_bin() -> None:
    """A Data Dragon spell with no bin counterpart fails loudly instead of vanishing."""
    detail = ddragon_detail(
        "Anivia", [("Missing", 5), ("Crystallize", 5), ("Frostbite", 5), ("GlacialStorm", 3)]
    )

    with pytest.raises(ValueError):
        group_spells("Anivia", detail, anivia_bin())


# ---------- calculation attribute tests ----------


def test_parse_damage_types_reads_the_tag_wrapping_each_reference() -> None:
    """A calculation reference takes the damage type of the tag around it."""
    description = (
        "<magicDamage>@Burn@ magic damage</magicDamage> and "
        "<physicalDamage>@Slash@</physicalDamage> and <trueDamage>@Execute*100@</trueDamage>"
    )

    assert parse_damage_types(description) == {
        "Burn": "magic",
        "Slash": "physical",
        "Execute": "true",
    }


def test_parse_damage_types_drops_a_reference_wrapped_in_two_tags() -> None:
    """A name claimed by two damage types is dropped rather than guessed."""
    description = "<magicDamage>@Hit@</magicDamage> then <physicalDamage>@Hit@</physicalDamage>"

    assert parse_damage_types(description) == {}


def test_parse_damage_types_handles_a_missing_description() -> None:
    """A passive publishes no dynamic description and yields no damage types."""
    assert parse_damage_types(None) == {}


def test_resolve_value_attributes_nulls_a_stat_two_calculations_disagree_on() -> None:
    """One value scaled as AD by one calculation and as AP by another scales with neither."""
    calculations = {
        "First": {
            "mFormulaParts": [
                {
                    "mStat": 2,
                    "mDataValue": "Ratio",
                    "__type": "StatByNamedDataValueCalculationPart",
                }
            ],
            "__type": "GameCalculation",
        },
        "Second": {
            "mFormulaParts": [
                {"mDataValue": "Ratio", "__type": "StatByNamedDataValueCalculationPart"}
            ],
            "__type": "GameCalculation",
        },
    }

    attributes = resolve_value_attributes(calculations, {})

    assert attributes["Ratio"].scaling_stat is None


def test_resolve_value_attributes_reads_a_stat_through_a_subpart() -> None:
    """A stat declared on a subpart wrapper reaches the value inside it."""
    calculations = {
        "Heal": {
            "mFormulaParts": [
                {
                    "mStat": 12,
                    "mSubpart": {
                        "mDataValue": "HealthRatio",
                        "__type": "NamedDataValueCalculationPart",
                    },
                    "__type": "StatBySubPartCalculationPart",
                }
            ],
            "__type": "GameCalculation",
        }
    }

    attributes = resolve_value_attributes(calculations, {})

    assert attributes["HealthRatio"].scaling_stat == "health"


def stat_scaling_calculation(part: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap one stat-scaling formula part in the calculation that owns it.

    Args:
        part: Fields of a StatByNamedDataValueCalculationPart, without its
            __type, so a test states only the enums it is about.

    Returns:
        A GameCalculation whose single formula part is that stat-scaling part.
    """
    return {
        "mFormulaParts": [dict(part) | {"__type": "StatByNamedDataValueCalculationPart"}],
        "__type": "GameCalculation",
    }


def test_resolve_value_attributes_reads_an_absent_mstatformula_as_the_total_stat() -> None:
    """These bins omit a field at its default, and the mStatFormula default is the total stat."""
    calculations = {"Damage": stat_scaling_calculation({"mStat": 2, "mDataValue": "Ratio"})}

    attributes = resolve_value_attributes(calculations, {})

    assert attributes["Ratio"].scaling_stat == "ad"
    assert attributes["Ratio"].stat_formula == "total"


def test_resolve_value_attributes_reads_mstatformula_two_as_the_bonus_stat() -> None:
    """A coefficient declaring mStatFormula 2 applies to the bonus amount of its stat."""
    calculations = {
        "Damage": stat_scaling_calculation(
            {"mStat": 2, "mStatFormula": 2, "mDataValue": "BonusRatio"}
        )
    }

    attributes = resolve_value_attributes(calculations, {})

    assert attributes["BonusRatio"].scaling_stat == "ad"
    assert attributes["BonusRatio"].stat_formula == "bonus"


def test_resolve_value_attributes_leaves_an_undecoded_mstatformula_null() -> None:
    """Enum value 1 has no proven meaning, so it is left NULL rather than guessed."""
    calculations = {
        "Damage": stat_scaling_calculation({"mStat": 2, "mStatFormula": 1, "mDataValue": "Ratio"})
    }

    attributes = resolve_value_attributes(calculations, {})

    assert attributes["Ratio"].scaling_stat == "ad"
    assert attributes["Ratio"].stat_formula is None


def test_resolve_value_attributes_nulls_a_formula_two_calculations_disagree_on() -> None:
    """One value read as total AD by one calculation and as bonus AD by another gets neither."""
    calculations = {
        "First": stat_scaling_calculation({"mStat": 2, "mDataValue": "Ratio"}),
        "Second": stat_scaling_calculation({"mStat": 2, "mStatFormula": 2, "mDataValue": "Ratio"}),
    }

    attributes = resolve_value_attributes(calculations, {})

    assert attributes["Ratio"].scaling_stat == "ad"
    assert attributes["Ratio"].stat_formula is None


def test_resolve_value_attributes_gives_an_unscaled_value_no_formula() -> None:
    """A value no stat-scaling part encloses scales with nothing, so no formula applies to it."""
    calculations = {
        "Damage": {
            "mFormulaParts": [
                {"mDataValue": "BaseDamage", "__type": "NamedDataValueCalculationPart"}
            ],
            "__type": "GameCalculation",
        }
    }

    attributes = resolve_value_attributes(calculations, {"Damage": "magic"})

    assert attributes["BaseDamage"].scaling_stat is None
    assert attributes["BaseDamage"].stat_formula is None


# ---------- ability value tests ----------


def ability_values_by_key(rows: Sequence[AbilityValue]) -> dict[tuple[str, str], AbilityValue]:
    """Index built rows by the spell and value name that identify them.

    Args:
        rows: Rows returned by build_ability_values.

    Returns:
        Mapping of (spell_key, name) to the row publishing it.
    """
    return {(row.spell_key, row.name): row for row in rows}


def test_build_ability_values_slices_the_aatrox_q_ground_truth() -> None:
    """Aatrox's Q cooldown and base damage match the values the game shows."""
    rows = ability_values_by_key(
        build_ability_values(
            "Aatrox",
            aatrox_detail(),
            aatrox_bin(),
            cdragon_record((AATROX_Q_DESCRIPTION, None, None, None)),
            ABILITY_IDS,
        )
    )

    assert rows[("AatroxQ", "CooldownTime")].values == [14.0, 12.0, 10.0, 8.0, 6.0]
    assert rows[("AatroxQ", "QBaseDamage")].values == [10.0, 25.0, 40.0, 55.0, 70.0]
    assert rows[("AatroxQ", "QBaseDamage")].kind == "per_rank"


def test_build_ability_values_takes_the_cooldown_from_the_root_spell_only() -> None:
    """The recast spells publish their own cooldown and it never reaches the corpus."""
    rows = build_ability_values(
        "Aatrox",
        aatrox_detail(),
        aatrox_bin(),
        cdragon_record((AATROX_Q_DESCRIPTION, None, None, None)),
        ABILITY_IDS,
    )
    cooldowns = [row for row in rows if row.name == "CooldownTime" and row.ability_id == 2]

    assert len(cooldowns) == 1
    assert cooldowns[0].spell_key == "AatroxQ"
    assert cooldowns[0].values == [14.0, 12.0, 10.0, 8.0, 6.0]


def test_build_ability_values_reads_mana_and_data_values_at_different_offsets() -> None:
    """Annie's Q proves mana starts at index zero while data values start at index one."""
    rows = ability_values_by_key(
        build_ability_values(
            "Annie",
            annie_detail(),
            annie_bin(),
            cdragon_record((ANNIE_Q_DESCRIPTION, None, None, None)),
            ABILITY_IDS,
        )
    )

    assert rows[("AnnieQ", "Mana")].values == [60.0, 65.0, 70.0, 75.0, 80.0]
    assert rows[("AnnieQ", "BaseDamage")].values == [80.0, 125.0, 170.0, 215.0, 260.0]


def test_build_ability_values_writes_no_mana_row_for_a_manaless_champion() -> None:
    """Aatrox pays no mana, so no zero row is invented for him."""
    rows = build_ability_values(
        "Aatrox",
        aatrox_detail(),
        aatrox_bin(),
        cdragon_record((AATROX_Q_DESCRIPTION, None, None, None)),
        ABILITY_IDS,
    )

    assert [row for row in rows if row.name == "Mana"] == []


def test_build_ability_values_keeps_both_same_named_ultimate_arrays() -> None:
    """Aurelion Sol's two BaseDamage arrays survive, told apart by spell_key."""
    rows = ability_values_by_key(
        build_ability_values(
            "AurelionSol",
            aurelion_sol_detail(),
            aurelion_sol_bin(),
            cdragon_record(),
            ABILITY_IDS,
        )
    )

    assert rows[("AurelionSolR", "BaseDamage")].values == [150.0, 250.0, 350.0]
    assert rows[("AurelionSolR2", "BaseDamage")].values == [200.0, 275.0, 350.0]


def test_build_ability_values_resolves_a_passive_for_a_champion_with_a_loose_one() -> None:
    """Udyr's passive resolves through the character record and keeps its own values only."""
    rows = build_ability_values("Udyr", udyr_detail(), udyr_bin(), cdragon_record(), ABILITY_IDS)
    passive = [row for row in rows if row.ability_id == ABILITY_IDS["P"]]

    assert [(row.spell_key, row.name, row.kind) for row in passive] == [
        ("UdyrPassive", "PBonusDamage", "scalar")
    ]
    assert passive[0].values == [5.0]


def test_build_ability_values_emits_a_by_level_row_with_exactly_two_values() -> None:
    """A number that exists only as a level interpolation lands as its two endpoints."""
    rows = ability_values_by_key(
        build_ability_values(
            "Aatrox",
            aatrox_detail(),
            aatrox_bin(),
            cdragon_record((AATROX_Q_DESCRIPTION, None, None, None)),
            ABILITY_IDS,
        )
    )
    row = rows[("AatroxPassive", "PDamage")]

    assert row.kind == "by_level"
    assert row.values == [0.04, 0.10]
    assert row.display_as_percent is True


def test_build_ability_values_reads_an_absent_mstat_as_ability_power() -> None:
    """Annie's AP ratio omits mStat, which these bins do for any field at its default."""
    rows = ability_values_by_key(
        build_ability_values(
            "Annie",
            annie_detail(),
            annie_bin(),
            cdragon_record((ANNIE_Q_DESCRIPTION, None, None, None)),
            ABILITY_IDS,
        )
    )

    assert rows[("AnnieQ", "APRatio")].scaling_stat == "ap"


def test_build_ability_values_reads_mstat_two_as_attack_damage() -> None:
    """Aatrox's total AD ratio declares mStat 2, which is attack damage."""
    rows = ability_values_by_key(
        build_ability_values(
            "Aatrox",
            aatrox_detail(),
            aatrox_bin(),
            cdragon_record((AATROX_Q_DESCRIPTION, None, None, None)),
            ABILITY_IDS,
        )
    )

    assert rows[("AatroxQ", "QTotalADRatio")].scaling_stat == "ad"


def test_build_ability_values_leaves_an_undecoded_mstat_null() -> None:
    """An mStat enum with no known meaning is left NULL rather than guessed."""
    payload = champion_bin(
        "Yasuo",
        {
            "YasuoPassiveAbility": {"YasuoPassive": spell_object()},
            "YasuoQAbility": {
                "YasuoQ": spell_object(
                    data_values={"ADRatio": [1.0] * 7},
                    calculations={
                        "Damage": {
                            "mFormulaParts": [
                                {
                                    "mStat": 9,
                                    "mDataValue": "ADRatio",
                                    "__type": "StatByNamedDataValueCalculationPart",
                                }
                            ],
                            "__type": "GameCalculation",
                        }
                    },
                )
            },
            "YasuoWAbility": {"YasuoW": spell_object()},
            "YasuoEAbility": {"YasuoE": spell_object()},
            "YasuoRAbility": {"YasuoR": spell_object()},
        },
        passive_key="Characters/Yasuo/Spells/YasuoPassiveAbility/YasuoPassive",
    )
    detail = ddragon_detail("Yasuo", [("YasuoQ", 5), ("YasuoW", 5), ("YasuoE", 5), ("YasuoR", 3)])

    rows = ability_values_by_key(
        build_ability_values("Yasuo", detail, payload, cdragon_record(), ABILITY_IDS)
    )

    assert rows[("YasuoQ", "ADRatio")].scaling_stat is None


def test_build_ability_values_propagates_the_tooltip_damage_type() -> None:
    """A calculation wrapped in a physical damage tag marks every value it sums."""
    rows = ability_values_by_key(
        build_ability_values(
            "Aatrox",
            aatrox_detail(),
            aatrox_bin(),
            cdragon_record((AATROX_Q_DESCRIPTION, None, None, None)),
            ABILITY_IDS,
        )
    )

    assert rows[("AatroxQ", "QBaseDamage")].damage_type == "physical"
    assert rows[("AatroxQ", "QTotalADRatio")].damage_type == "physical"
    assert rows[("AatroxQ", "CooldownTime")].damage_type is None


def test_build_ability_values_skips_a_value_the_bin_publishes_as_null() -> None:
    """A null array means the value sits at its default and carries no fact."""
    payload = anivia_bin()
    payload["Characters/Anivia/Spells/GlacialStormAbility/GlacialStorm"] = spell_object(
        data_values={"SlowAmount": [0.2] * 7, "Unused": None}
    )

    rows = ability_values_by_key(
        build_ability_values("Anivia", anivia_detail(), payload, cdragon_record(), ABILITY_IDS)
    )

    assert ("GlacialStorm", "SlowAmount") in rows
    assert ("GlacialStorm", "Unused") not in rows


def test_build_ability_values_keeps_the_first_of_two_values_sharing_a_name() -> None:
    """A spell publishing one name twice keeps the first, satisfying the unique key."""
    payload = anivia_bin()
    spell = spell_object(data_values={"SlowAmount": [0.2] * 7})
    spell["mSpell"]["DataValues"].append(
        {"name": "SlowAmount", "values": [0.9] * 7, "__type": "SpellDataValue"}
    )
    payload["Characters/Anivia/Spells/GlacialStormAbility/GlacialStorm"] = spell

    rows = [
        row
        for row in build_ability_values(
            "Anivia", anivia_detail(), payload, cdragon_record(), ABILITY_IDS
        )
        if row.name == "SlowAmount"
    ]

    assert len(rows) == 1
    assert rows[0].values == [0.2, 0.2, 0.2]


def test_build_ability_values_gives_naafiri_her_reworked_slots() -> None:
    """The rows for W and R hang off the abilities their array positions name."""
    rows = build_ability_values(
        "Naafiri", naafiri_detail(), naafiri_bin(), cdragon_record(), ABILITY_IDS
    )
    owners = {row.spell_key: row.ability_id for row in rows}

    assert owners["NaafiriR"] == ABILITY_IDS["W"]
    assert owners["NaafiriW"] == ABILITY_IDS["R"]
    assert owners["NaafiriQ"] == ABILITY_IDS["Q"]
    assert owners["NaafiriE"] == ABILITY_IDS["E"]


# ---------- item value tests ----------


def ddragon_items() -> dict[str, Any]:
    """Build a Data Dragon item.json body covering the stat and duplicate cases.

    Returns:
        Payload holding Blade of the Ruined King, Youmuu's Ghostblade and
        Bloodthirster, the last two being the items that publish one bin value
        name twice.
    """
    return {
        "data": {
            "3153": {
                "name": "Blade of the Ruined King",
                "description": "<mainText>Bork</mainText>",
                "gold": {"base": 725, "total": 3200},
                "stats": {"FlatPhysicalDamageMod": 40, "PercentAttackSpeedMod": 0.25},
            },
            "3142": {
                "name": "Youmuu's Ghostblade",
                "description": "<mainText>Ghostblade</mainText>",
                "gold": {"base": 700, "total": 2800},
                "stats": {"FlatPhysicalDamageMod": 55},
            },
            "223072": {
                "name": "Bloodthirster",
                "description": "<mainText>Bloodthirster</mainText>",
                "gold": {"base": 450, "total": 3300},
                "stats": {},
            },
        }
    }


def item_bin() -> dict[str, Any]:
    """Build an items.cdtb bin body matching the Data Dragon item fixture.

    Returns:
        Payload whose Youmuu's and Bloodthirster entries publish a repeated
        mName, and whose Bloodthirster entry omits one mValue entirely.
    """

    def data_value(name: str, value: float | None) -> dict[str, Any]:
        entry: dict[str, Any] = {"mName": name, "__type": "ItemDataValue"}
        if value is not None:
            entry["mValue"] = value
        return entry

    return {
        "Items/3153": {
            "__type": "ItemData",
            "itemID": 3153,
            "mFlatPhysicalDamageMod": 40.0,
            "mPercentAttackSpeedMod": 0.25,
            "mDataValues": [data_value("MeleeValue", 0.09), data_value("RangedValue", 0.06)],
        },
        "Items/3142": {
            "__type": "ItemData",
            "itemID": 3142,
            "mFlatPhysicalDamageMod": 55.0,
            "mDataValues": [data_value("Cooldown", 45.0), data_value("Cooldown", 45.0)],
        },
        "Items/223072": {
            "__type": "ItemData",
            "itemID": 223072,
            "mDataValues": [
                data_value("DecayTime", 25.0),
                data_value("DecayTime", 25.0),
                data_value("Overheal", None),
            ],
        },
        "Items/ItemGroups/Default": {"__type": "ItemGroup"},
    }


def item_values_by_key(rows: Sequence[ItemValue]) -> dict[tuple[str, str], ItemValue]:
    """Index built item rows by the item and value name that identify them.

    Args:
        rows: Rows returned by build_item_values.

    Returns:
        Mapping of (item_id, name) to the row publishing it.
    """
    return {(row.item_id, row.name): row for row in rows}


def test_build_item_values_takes_stats_from_data_dragon_and_values_from_the_bin() -> None:
    """Blade of the Ruined King carries its stat from one source and its ratio from the other."""
    rows = item_values_by_key(build_item_values(ddragon_items(), item_bin()))

    assert rows[("3153", "FlatPhysicalDamageMod")].values == [40.0]
    assert rows[("3153", "FlatPhysicalDamageMod")].source == "ddragon"
    assert rows[("3153", "MeleeValue")].values == [0.09]
    assert rows[("3153", "MeleeValue")].source == "cdragon"
    assert rows[("3153", "MeleeValue")].kind == "scalar"


def test_build_item_values_collapses_a_name_an_item_publishes_twice() -> None:
    """Youmuu's doubled Cooldown and Bloodthirster's doubled DecayTime yield one row each."""
    rows = build_item_values(ddragon_items(), item_bin())
    youmuus = [row for row in rows if row.item_id == "3142" and row.name == "Cooldown"]
    bloodthirster = [row for row in rows if row.item_id == "223072" and row.name == "DecayTime"]

    assert [row.values for row in youmuus] == [[45.0]]
    assert [row.values for row in bloodthirster] == [[25.0]]


def test_build_item_values_reads_a_missing_mvalue_as_zero() -> None:
    """An entry with no mValue sits at its default, which is zero."""
    rows = item_values_by_key(build_item_values(ddragon_items(), item_bin()))

    assert rows[("223072", "Overheal")].values == [0.0]


def test_build_item_values_matches_every_data_dragon_stat_against_the_bin() -> None:
    """Every Data Dragon stat equals the m-prefixed field the bin publishes for it."""
    rows = item_values_by_key(build_item_values(ddragon_items(), item_bin()))
    entries = {
        str(entry["itemID"]): entry
        for entry in item_bin().values()
        if entry.get("__type") == "ItemData"
    }

    for (item_id, name), row in rows.items():
        if row.source != "ddragon":
            continue
        assert row.values == [entries[item_id][f"m{name}"]]


# ---------- orchestrator harness ----------

DDRAGON_DATA_PATH = "/cdn/16.14.1/data/en_US"
CDRAGON_CHAMPION_PATH = "/latest/plugins/rcp-be-lol-game-data/global/default/v1/champions"

CHAMPIONS = {"Aatrox": 266, "Annie": 1}

EXPECTED_STATS = ValueLoadStats(ability_values=9, item_values=8)


def build_routes() -> dict[str, Any]:
    """Build the URL-path to response-body map covering every endpoint load_values touches.

    Returns:
        Mapping of URL path to the JSON body served for it, spanning two
        champions and the three-item catalogue.
    """
    routes: dict[str, Any] = {
        f"{DDRAGON_DATA_PATH}/champion.json": {
            "data": {
                name: {"id": name, "key": str(key), "name": name, "tags": ["Fighter"]}
                for name, key in CHAMPIONS.items()
            }
        },
        f"{DDRAGON_DATA_PATH}/item.json": ddragon_items(),
        "/latest/game/items.cdtb.bin.json": item_bin(),
        f"{DDRAGON_DATA_PATH}/champion/Aatrox.json": aatrox_detail(),
        f"{DDRAGON_DATA_PATH}/champion/Annie.json": annie_detail(),
        "/latest/game/data/characters/aatrox/aatrox.bin.json": aatrox_bin(),
        "/latest/game/data/characters/annie/annie.bin.json": annie_bin(),
        f"{CDRAGON_CHAMPION_PATH}/266.json": cdragon_record(
            (AATROX_Q_DESCRIPTION, None, None, None)
        ),
        f"{CDRAGON_CHAMPION_PATH}/1.json": cdragon_record((ANNIE_Q_DESCRIPTION, None, None, None)),
    }
    return routes


ROUTES = build_routes()


def corpus_handler() -> Callable[[httpx.Request], httpx.Response]:
    """Build a handler serving the fixture corpus.

    Returns:
        Handler suitable for httpx.MockTransport. Any path outside the fixture
        corpus gets a 404 so a wrong URL fails loudly instead of passing.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        body = ROUTES.get(request.url.path)
        if body is None:
            return httpx.Response(404, json={"unrouted": request.url.path})
        return httpx.Response(200, json=body)

    return handler


@pytest.fixture
def db_session() -> Iterator[Session]:
    """Yield a Session bound to a transaction that is rolled back after the test.

    Returns:
        A Session against the configured database; every change made through it
        is discarded when the test finishes, so the database stays empty.
    """
    engine = create_engine(get_settings().database_url)
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    try:
        yield session
    finally:
        session.close()
        if transaction.is_active:
            transaction.rollback()
        connection.close()
        engine.dispose()


def seed_entities(session: Session) -> None:
    """Insert the champion, ability and item rows the value loader hangs values off.

    Args:
        session: Open Session the rows are added to.

    Returns:
        None. The rows are flushed so the value loader can read back the
        surrogate ability ids.
    """
    session.add(Faction(slug="unaffiliated", name="Unaffiliated"))
    session.flush()
    for ddragon_id in CHAMPIONS:
        slug = ddragon_id.lower()
        session.add(
            Champion(
                slug=slug,
                ddragon_key=ddragon_id,
                name=ddragon_id,
                title=f"the {ddragon_id}",
                faction_slug="unaffiliated",
                bio_full="Bio.",
                bio_full_text="Bio.",
                playable=True,
            )
        )
        session.flush()
        for slot in ("P", *SPELL_SLOT_IDS):
            session.add(
                Ability(
                    champion_slug=slug,
                    slot=slot,
                    name=f"{ddragon_id} {slot}",
                    description="Ability.",
                    max_rank=None if slot == "P" else 5,
                )
            )
    for item_id, record in ddragon_items()["data"].items():
        session.add(
            Item(
                ddragon_id=item_id,
                name=record["name"],
                description=record["description"],
                description_text=record["name"],
                gold_total=record["gold"]["total"],
                gold_base=record["gold"]["base"],
                purchasable=True,
                in_store=True,
            )
        )
    session.flush()


# ---------- orchestrator tests ----------


async def test_load_values_persists_both_value_tables(db_session: Session, tmp_path: Path) -> None:
    """One run lands every fixture value and reports matching counts."""
    seed_entities(db_session)
    settings = build_settings(tmp_path)
    async with build_client(settings, corpus_handler()) as client:
        stats = await load_values(db_session, client, settings)

    assert stats == EXPECTED_STATS
    assert db_session.execute(select(func.count()).select_from(AbilityValue)).scalar_one() == 9
    assert db_session.execute(select(func.count()).select_from(ItemValue)).scalar_one() == 8


async def test_load_values_stores_the_aatrox_q_ground_truth(
    db_session: Session, tmp_path: Path
) -> None:
    """The stored Aatrox Q rows carry the sliced arrays and the resolved scaling stat."""
    seed_entities(db_session)
    settings = build_settings(tmp_path)
    async with build_client(settings, corpus_handler()) as client:
        await load_values(db_session, client, settings)

    rows = dict(
        db_session.execute(
            select(AbilityValue.name, AbilityValue.values)
            .join(Ability, Ability.id == AbilityValue.ability_id)
            .where(Ability.champion_slug == "aatrox", Ability.slot == "Q")
        ).all()
    )
    assert rows["CooldownTime"] == [14.0, 12.0, 10.0, 8.0, 6.0]
    assert rows["QBaseDamage"] == [10.0, 25.0, 40.0, 55.0, 70.0]


async def test_load_values_stores_the_duplicated_item_names_once(
    db_session: Session, tmp_path: Path
) -> None:
    """Bloodthirster and Youmuu's each land one row for their doubled value name."""
    seed_entities(db_session)
    settings = build_settings(tmp_path)
    async with build_client(settings, corpus_handler()) as client:
        await load_values(db_session, client, settings)

    rows = db_session.execute(
        select(ItemValue.item_id, ItemValue.name, ItemValue.values).where(
            ItemValue.name.in_(("Cooldown", "DecayTime"))
        )
    ).all()
    assert sorted(rows) == [("223072", "DecayTime", [25.0]), ("3142", "Cooldown", [45.0])]


async def test_load_values_run_twice_is_idempotent(db_session: Session, tmp_path: Path) -> None:
    """A second run over the same corpus updates rows in place instead of duplicating."""
    seed_entities(db_session)
    settings = build_settings(tmp_path)
    async with build_client(settings, corpus_handler()) as client:
        first = await load_values(db_session, client, settings)
        ability_ids = sorted(db_session.execute(select(AbilityValue.id)).scalars())
        item_ids = sorted(db_session.execute(select(ItemValue.id)).scalars())

        second = await load_values(db_session, client, settings)

    assert second == first
    assert sorted(db_session.execute(select(AbilityValue.id)).scalars()) == ability_ids
    assert sorted(db_session.execute(select(ItemValue.id)).scalars()) == item_ids
