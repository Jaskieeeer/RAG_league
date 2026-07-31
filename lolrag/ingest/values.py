import asyncio
import logging
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from lolrag.config import Settings
from lolrag.db.models import Ability, AbilityValue, Base, Champion, ItemValue
from lolrag.fetch import cdragon, cdragon_bin, ddragon
from lolrag.fetch.client import FetchClient
from lolrag.ingest.identifiers import PASSIVE_SLOT, SPELL_SLOTS

logger = logging.getLogger(__name__)

ABILITY_OBJECT_TYPE = "AbilityObject"
SPELL_OBJECT_TYPE = "SpellObject"
CHARACTER_RECORD_TYPE = "CharacterRecord"
ROOT_RECORD_SUFFIX = "/Root"
ITEM_DATA_TYPE = "ItemData"

DATA_VALUE_OFFSET = 1
MANA_OFFSET = 0

COOLDOWN_VALUE_NAME = "CooldownTime"
MANA_VALUE_NAME = "Mana"

KIND_PER_RANK = "per_rank"
KIND_BY_LEVEL = "by_level"
KIND_SCALAR = "scalar"

SOURCE_DDRAGON = "ddragon"
SOURCE_CDRAGON = "cdragon"

SCALING_STAT_BY_CODE = {
    0: "ap",
    1: "armor",
    2: "ad",
    4: "attack_speed",
    6: "magic_resist",
    8: "crit",
    12: "health",
}

STAT_FORMULA_BY_CODE = {0: "total", 2: "bonus"}

DAMAGE_TYPE_BY_TAG = {
    "magicDamage": "magic",
    "physicalDamage": "physical",
    "trueDamage": "true",
}

STAT_BY_NAMED_DATA_VALUE_TYPE = "StatByNamedDataValueCalculationPart"
STAT_BY_SUB_PART_TYPE = "StatBySubPartCalculationPart"
NAMED_DATA_VALUE_TYPES = frozenset(
    {"NamedDataValueCalculationPart", "BuffCounterByNamedDataValueCalculationPart"}
)
INTERPOLATION_TYPE = "ByCharLevelInterpolationCalculationPart"
CALCULATION_REFERENCE_KEYS = frozenset(
    {
        "mModifiedGameCalculation",
        "mConditionalGameCalculation",
        "mDefaultGameCalculation",
        "mSpellCalculationKey",
    }
)

_UNSCALED = "__unscaled__"
_UNDECODED_STAT = "__undecoded__"
_UNDECODED_FORMULA = "__undecoded_formula__"

_DAMAGE_TAG_PATTERN = re.compile(r"<(magicDamage|physicalDamage|trueDamage)>(.*?)</\1>", re.DOTALL)
_TOKEN_PATTERN = re.compile(r"@([^@]+)@")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


# ---------- spell grouping ----------


@dataclass(frozen=True)
class SpellGroup:
    """One ability's slot, rank count and the bin spells that publish its values.

    Args:
        slot: Ability slot, one of P, Q, W, E, R.
        max_rank: Number of learnable ranks, or None for the passive, whose
            values cannot be sliced by rank.
        root_key: Full bin key of the ability's root spell, the only spell whose
            cooldown and mana cost describe the ability itself.
        member_keys: Full bin keys of every spell that belongs to the ability,
            including root_key. Child spells publish their own data values.
    """

    slot: str
    max_rank: int | None
    root_key: str
    member_keys: tuple[str, ...]


def spell_key(bin_key: str) -> str:
    """Return the short spell name a full bin key ends with.

    Args:
        bin_key: Full bin key, e.g.
            "Characters/Aatrox/Spells/AatroxQAbility/AatroxQ".

    Returns:
        The last path segment, e.g. "AatroxQ". Hashed keys such as "{3e27cfac}"
        carry no path and are returned unchanged.
    """
    return bin_key.rsplit("/", 1)[-1]


def passive_spell_key(bin_payload: Mapping[str, Any]) -> str:
    """Return the bin key of the champion's passive spell.

    Args:
        bin_payload: Parsed Community Dragon champion bin file.

    Returns:
        The value of mCharacterPassiveSpell on the root CharacterRecord. The
        AbilityObject mType flag is deliberately not used: it marks only 165 of
        the 173 champions and misses Udyr entirely.

    Raises:
        ValueError: If the bin publishes no root CharacterRecord.
    """
    for key, value in bin_payload.items():
        if not isinstance(value, dict) or value.get("__type") != CHARACTER_RECORD_TYPE:
            continue
        if key.endswith(ROOT_RECORD_SUFFIX):
            return value["mCharacterPassiveSpell"]
    raise ValueError("bin payload publishes no root CharacterRecord")


def group_spells(
    ddragon_id: str, detail: Mapping[str, Any], bin_payload: Mapping[str, Any]
) -> list[SpellGroup]:
    """Join the Data Dragon ability slots to the bin spells that back them.

    Args:
        ddragon_id: Data Dragon champion id the detail body is keyed on.
        detail: Parsed Data Dragon champion detail body, whose "data" key holds
            the record with "spells".
        bin_payload: Parsed Community Dragon champion bin file.

    Returns:
        Five groups, the passive first and then the four spells in "spells"
        order as slots Q, W, E and R. Slot comes from array position, never from
        the letter inside the spell id: Naafiri publishes NaafiriR in the W
        position and NaafiriW in the R position. A spell is matched to the bin
        by comparing its Data Dragon id to the last segment of every bin spell
        key, which resolves champions that name spells after the ability, such
        as Anivia's GlacialStorm.

    Raises:
        ValueError: If a champion publishes a number of spells other than the
            four Q/W/E/R slots, or if a spell id matches no bin spell.
    """
    record = detail["data"][ddragon_id]
    spell_keys = {
        key
        for key, value in bin_payload.items()
        if isinstance(value, dict) and value.get("__type") == SPELL_OBJECT_TYPE
    }
    ability_keys = {
        key
        for key, value in bin_payload.items()
        if isinstance(value, dict) and value.get("__type") == ABILITY_OBJECT_TYPE
    }
    key_by_name = {spell_key(key): key for key in spell_keys}

    groups = [
        _build_group(PASSIVE_SLOT, None, passive_spell_key(bin_payload), spell_keys, ability_keys)
    ]
    for slot, spell in zip(SPELL_SLOTS, record["spells"], strict=True):
        root_key = key_by_name.get(spell["id"])
        if root_key is None:
            raise ValueError(f"{ddragon_id} spell {spell['id']} matches no bin spell")
        groups.append(_build_group(slot, spell["maxrank"], root_key, spell_keys, ability_keys))
    return groups


def _build_group(
    slot: str,
    max_rank: int | None,
    root_key: str,
    spell_keys: set[str],
    ability_keys: set[str],
) -> SpellGroup:
    """Collect the bin spells that share the root spell's AbilityObject parent.

    Args:
        slot: Ability slot the group describes.
        max_rank: Number of learnable ranks, None for the passive.
        root_key: Full bin key of the root spell.
        spell_keys: Every SpellObject key in the bin.
        ability_keys: Every AbilityObject key in the bin.

    Returns:
        A SpellGroup whose member_keys are the root spell's siblings when the
        root sits under an AbilityObject, and the root spell alone otherwise.
        Aphelios' Q and E and thirteen passives hang directly off the champion's
        spell folder, where a prefix sweep would swallow unrelated spells.
    """
    parent = root_key.rsplit("/", 1)[0]
    if parent in ability_keys:
        members = tuple(sorted(key for key in spell_keys if key.startswith(f"{parent}/")))
    else:
        members = (root_key,)
    return SpellGroup(slot=slot, max_rank=max_rank, root_key=root_key, member_keys=members)


# ---------- calculation attributes ----------


@dataclass(frozen=True)
class ValueAttributes:
    """Attributes a value inherits from the calculations that reference it.

    Args:
        scaling_stat: Champion stat the value scales with, or None when it does
            not scale, the source enum is undecoded, or two calculations
            disagree.
        stat_formula: Which amount of that stat the value applies to, one of
            total, bonus, or None when the value does not scale, the source enum
            is undecoded, or two calculations disagree.
        damage_type: Damage type the value contributes, or None when no
            calculation declares one or two calculations disagree.
        display_as_percent: Whether the referencing calculation displays its
            result as a percentage; False when calculations disagree.
    """

    scaling_stat: str | None = None
    stat_formula: str | None = None
    damage_type: str | None = None
    display_as_percent: bool = False


@dataclass(frozen=True)
class _CalculationLeaf:
    """One data value a calculation reaches, with what its enclosing part declares.

    Args:
        name: Source name of the data value the leaf refers to.
        stat: Scaling stat inherited from an enclosing stat-scaling part, the
            undecoded sentinel when that part's enum has no known meaning, or
            the unscaled sentinel when no such part encloses the leaf.
        stat_formula: Which amount of that stat the enclosing part applies the
            value to, the undecoded sentinel when its enum has no known meaning,
            or None when no stat encloses the leaf.
    """

    name: str
    stat: str
    stat_formula: str | None


def parse_damage_types(dynamic_description: str | None) -> dict[str, str]:
    """Map calculation names to the damage type their tooltip wraps them in.

    Args:
        dynamic_description: Community Dragon spell dynamicDescription, whose
            calculation references appear as "@Name@" inside <magicDamage>,
            <physicalDamage> or <trueDamage> tags. None for the passive, which
            publishes no dynamic description.

    Returns:
        Mapping of calculation name to damage type. A name wrapped in two
        different tags is dropped rather than guessed.
    """
    found: dict[str, set[str]] = defaultdict(set)
    for match in _DAMAGE_TAG_PATTERN.finditer(dynamic_description or ""):
        damage_type = DAMAGE_TYPE_BY_TAG[match.group(1)]
        for token in _TOKEN_PATTERN.findall(match.group(2)):
            identifier = _IDENTIFIER_PATTERN.match(token)
            if identifier is not None:
                found[identifier.group(0)].add(damage_type)
    return {name: next(iter(types)) for name, types in found.items() if len(types) == 1}


def resolve_value_attributes(
    spell_calculations: Mapping[str, Any], damage_types: Mapping[str, str]
) -> dict[str, ValueAttributes]:
    """Resolve the attributes every data value inherits from its calculations.

    Args:
        spell_calculations: The spell's mSpellCalculations object, mapping
            calculation name to a formula graph.
        damage_types: Calculation name to damage type, as parsed from the
            tooltip by parse_damage_types.

    Returns:
        Mapping of data value name to its ValueAttributes. The formula graph is
        walked once here and then discarded; formula structure is not persisted.
        A value referenced by two calculations that disagree on an attribute
        gets None for it rather than an arbitrary winner.
    """
    stats: dict[str, set[str]] = defaultdict(set)
    formulas: dict[str, set[str]] = defaultdict(set)
    damage: dict[str, set[str]] = defaultdict(set)
    percent: dict[str, set[bool]] = defaultdict(set)

    for name, calculation in spell_calculations.items():
        declared_damage = damage_types.get(name)
        declared_percent = (
            calculation.get("mDisplayAsPercent") if isinstance(calculation, dict) else None
        )
        for leaf in _calculation_leaves(name, spell_calculations):
            if leaf.stat != _UNSCALED:
                stats[leaf.name].add(leaf.stat)
            if leaf.stat_formula is not None:
                formulas[leaf.name].add(leaf.stat_formula)
            if declared_damage is not None:
                damage[leaf.name].add(declared_damage)
            if declared_percent is not None:
                percent[leaf.name].add(bool(declared_percent))

    names = set(stats) | set(formulas) | set(damage) | set(percent)
    return {
        name: ValueAttributes(
            scaling_stat=_single_stat(stats.get(name)),
            stat_formula=_single_formula(formulas.get(name)),
            damage_type=_single(damage.get(name)),
            display_as_percent=bool(_single(percent.get(name))),
        )
        for name in names
    }


def _single(values: set[Any] | None) -> Any:
    """Return the sole member of a set, or None when it is empty or divided.

    Args:
        values: Declared values collected from every referencing calculation, or
            None when nothing was declared.

    Returns:
        The single declared value, or None when the source declared nothing or
        declared two different things.
    """
    if values is None or len(values) != 1:
        return None
    return next(iter(values))


def _single_stat(values: set[str] | None) -> str | None:
    """Return the sole declared scaling stat, discarding undecoded source enums.

    Args:
        values: Scaling stats collected from every referencing calculation.

    Returns:
        The single declared stat, or None when the source declared nothing,
        declared two different stats, or used an enum value with no known
        meaning.
    """
    stat = _single(values)
    if stat == _UNDECODED_STAT:
        return None
    return stat


def _single_formula(values: set[str] | None) -> str | None:
    """Return the sole declared stat formula, discarding undecoded source enums.

    Args:
        values: Stat formulas collected from every referencing calculation.

    Returns:
        The single declared formula, or None when the source declared nothing,
        declared two different formulas, or used an enum value with no known
        meaning.
    """
    formula = _single(values)
    if formula == _UNDECODED_FORMULA:
        return None
    return formula


def _stat_of(part: Mapping[str, Any]) -> str:
    """Decode the mStat enum of a stat-scaling calculation part.

    Args:
        part: A StatByNamedDataValueCalculationPart or
            StatBySubPartCalculationPart object.

    Returns:
        The scaling stat name. These bins omit any field holding its default, so
        an absent mStat means 0, which is Ability Power. Enum values with no
        known meaning resolve to a sentinel that later becomes NULL.
    """
    return SCALING_STAT_BY_CODE.get(part.get("mStat", 0), _UNDECODED_STAT)


def _formula_of(part: Mapping[str, Any]) -> str:
    """Decode the mStatFormula enum of a stat-scaling calculation part.

    Args:
        part: A StatByNamedDataValueCalculationPart or
            StatBySubPartCalculationPart object.

    Returns:
        Which amount of the part's stat the value applies to. These bins omit any
        field holding its default, so an absent mStatFormula means 0, which is
        the total stat. Enum value 1 carries no proven meaning and, like any
        other unknown code, resolves to a sentinel that later becomes NULL.
    """
    return STAT_FORMULA_BY_CODE.get(part.get("mStatFormula", 0), _UNDECODED_FORMULA)


def _walk_part(
    node: Any,
    stat: str,
    stat_formula: str | None,
    leaves: list[_CalculationLeaf],
    references: list[str],
) -> None:
    """Collect the data values and calculation references under one formula node.

    Args:
        node: Any node of a calculation formula graph.
        stat: Scaling stat inherited from an enclosing stat-scaling part, or the
            unscaled sentinel.
        stat_formula: Stat formula inherited from that same part, or None when
            no stat encloses the node.
        leaves: Accumulator receiving one _CalculationLeaf per data value.
        references: Accumulator receiving the names of other calculations this
            node defers to.

    Returns:
        None. Both accumulators are mutated in place.
    """
    if isinstance(node, dict):
        node_type = node.get("__type")
        if node_type == STAT_BY_NAMED_DATA_VALUE_TYPE:
            leaves.append(
                _CalculationLeaf(
                    name=node["mDataValue"], stat=_stat_of(node), stat_formula=_formula_of(node)
                )
            )
            return
        if node_type == STAT_BY_SUB_PART_TYPE:
            _walk_part(node.get("mSubpart"), _stat_of(node), _formula_of(node), leaves, references)
            return
        if node_type in NAMED_DATA_VALUE_TYPES:
            leaves.append(
                _CalculationLeaf(name=node["mDataValue"], stat=stat, stat_formula=stat_formula)
            )
            return
        for key, value in node.items():
            if key in CALCULATION_REFERENCE_KEYS and isinstance(value, str):
                references.append(value)
                continue
            _walk_part(value, stat, stat_formula, leaves, references)
    elif isinstance(node, list):
        for value in node:
            _walk_part(value, stat, stat_formula, leaves, references)


def _calculation_leaves(name: str, spell_calculations: Mapping[str, Any]) -> list[_CalculationLeaf]:
    """Collect every data value one calculation reaches, following its references.

    Args:
        name: Calculation name to start from.
        spell_calculations: The spell's mSpellCalculations object.

    Returns:
        One _CalculationLeaf per data value the calculation reaches and per data
        value every calculation it defers to reaches, so a tooltip that names a
        modified calculation still reaches the values the base calculation sums.
    """
    leaves: list[_CalculationLeaf] = []
    pending = [name]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current in visited or current not in spell_calculations:
            continue
        visited.add(current)
        references: list[str] = []
        _walk_part(spell_calculations[current], _UNSCALED, None, leaves, references)
        pending.extend(references)
    return leaves


def _interpolation_part(node: Any) -> dict[str, Any] | None:
    """Find the first by-character-level interpolation part in a calculation.

    Args:
        node: Any node of a calculation formula graph.

    Returns:
        The ByCharLevelInterpolationCalculationPart object, or None when the
        calculation holds none.
    """
    if isinstance(node, dict):
        if node.get("__type") == INTERPOLATION_TYPE:
            return node
        for value in node.values():
            found = _interpolation_part(value)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _interpolation_part(value)
            if found is not None:
                return found
    return None


# ---------- ability value builders ----------


def slice_ranks(values: Sequence[float], max_rank: int | None, offset: int) -> list[float]:
    """Cut the engine's padded rank array down to the ranks that exist.

    Args:
        values: Source array. DataValues and cooldownTime arrays are seven wide
            and one-indexed; mana arrays are six wide and zero-indexed. The
            surplus slots are generated filler and are never stored.
        max_rank: Number of learnable ranks, or None for a passive.
        offset: Index of rank one within values.

    Returns:
        One entry per learnable rank, or the single rank-one entry for a
        passive. Passive arrays cannot be sliced by rank because abilities
        publish no max_rank for them, and 634 of 635 passive data values are
        constant across every slot anyway.
    """
    if max_rank is None:
        return [float(values[offset])]
    return [float(value) for value in values[offset : offset + max_rank]]


def build_ability_values(
    ddragon_id: str,
    detail: Mapping[str, Any],
    bin_payload: Mapping[str, Any],
    cdragon_champion: Mapping[str, Any],
    ability_ids: Mapping[str, int],
) -> list[AbilityValue]:
    """Build every numeric value one champion's abilities publish.

    Args:
        ddragon_id: Data Dragon champion id the detail body is keyed on.
        detail: Parsed Data Dragon champion detail body, supplying slot order
            and each spell's maxrank.
        bin_payload: Parsed Community Dragon champion bin file, supplying the
            data values, cooldowns, mana costs and formula graphs.
        cdragon_champion: Parsed Community Dragon champion record, supplying the
            dynamicDescription that names each calculation's damage type.
        ability_ids: Mapping of ability slot to the stored abilities.id.

    Returns:
        AbilityValue rows for every member spell of every ability. Cooldown and
        mana come from the root spell alone, because child spells publish their
        own recast windows and missile sub-spells publish zeros, which would put
        contradictory answers for one ability into the corpus. Data values whose
        array is null are skipped: these bins omit a field at its default, so a
        null array means the value is zero everywhere and carries no fact. A
        name published twice by one spell keeps its first occurrence.

    Raises:
        KeyError: If ability_ids has no entry for a slot.
        ValueError: If the Data Dragon spells cannot be joined to the bin.
    """
    damage_types_by_slot = {PASSIVE_SLOT: {}} | {
        slot: parse_damage_types(spell.get("dynamicDescription"))
        for slot, spell in zip(SPELL_SLOTS, cdragon_champion["spells"], strict=True)
    }

    rows: list[AbilityValue] = []
    for group in group_spells(ddragon_id, detail, bin_payload):
        ability_id = ability_ids[group.slot]
        damage_types = damage_types_by_slot[group.slot]
        for member_key in group.member_keys:
            rows.extend(
                _build_spell_values(
                    ability_id=ability_id,
                    group=group,
                    member_key=member_key,
                    spell=bin_payload[member_key],
                    damage_types=damage_types,
                )
            )
    return rows


def _build_spell_values(
    *,
    ability_id: int,
    group: SpellGroup,
    member_key: str,
    spell: Mapping[str, Any],
    damage_types: Mapping[str, str],
) -> list[AbilityValue]:
    """Build the value rows one member spell of an ability publishes.

    Args:
        ability_id: Stored abilities.id the rows hang off.
        group: The ability group the spell belongs to, carrying slot, max_rank
            and the root spell key.
        member_key: Full bin key of this spell.
        spell: The spell's parsed SpellObject.
        damage_types: Calculation name to damage type for the ability's slot.

    Returns:
        Cooldown and mana rows when this spell is the ability's root, then one
        row per data value, then one by-level row per calculation whose value
        exists only as a level-1 to level-18 interpolation.
    """
    payload = spell.get("mSpell") or {}
    calculations = payload.get("mSpellCalculations") or {}
    attributes = resolve_value_attributes(calculations, damage_types)
    key = spell_key(member_key)
    kind = KIND_SCALAR if group.max_rank is None else KIND_PER_RANK

    rows: list[AbilityValue] = []
    taken: set[str] = set()

    def add(name: str, row_kind: str, values: list[float], attrs: ValueAttributes) -> None:
        if name in taken:
            logger.debug("skipping duplicate value %s on spell %s", name, key)
            return
        taken.add(name)
        rows.append(
            AbilityValue(
                ability_id=ability_id,
                spell_key=key,
                name=name,
                kind=row_kind,
                values=values,
                scaling_stat=attrs.scaling_stat,
                stat_formula=attrs.stat_formula,
                damage_type=attrs.damage_type,
                display_as_percent=attrs.display_as_percent,
                source=SOURCE_CDRAGON,
            )
        )

    if member_key == group.root_key:
        cooldown = payload.get("cooldownTime")
        if cooldown is not None:
            add(
                COOLDOWN_VALUE_NAME,
                kind,
                slice_ranks(cooldown, group.max_rank, DATA_VALUE_OFFSET),
                ValueAttributes(),
            )
        mana = payload.get("mana")
        if mana is not None:
            add(
                MANA_VALUE_NAME,
                kind,
                slice_ranks(mana, group.max_rank, MANA_OFFSET),
                ValueAttributes(),
            )

    for entry in payload.get("DataValues") or []:
        values = entry.get("values")
        if values is None:
            logger.debug("skipping defaulted value %s on spell %s", entry["name"], key)
            continue
        name = entry["name"]
        add(
            name,
            kind,
            slice_ranks(values, group.max_rank, DATA_VALUE_OFFSET),
            attributes.get(name, ValueAttributes()),
        )

    for name, calculation in calculations.items():
        part = _interpolation_part(calculation)
        if part is None:
            continue
        add(
            name,
            KIND_BY_LEVEL,
            [float(part.get("mStartValue", 0.0)), float(part.get("mEndValue", 0.0))],
            ValueAttributes(
                damage_type=damage_types.get(name),
                display_as_percent=bool(calculation.get("mDisplayAsPercent")),
            ),
        )
    return rows


# ---------- item value builders ----------


def build_item_values(
    item_payload: Mapping[str, Any], item_bin: Mapping[str, Any]
) -> list[ItemValue]:
    """Build every numeric value the items publish, from both sources.

    Args:
        item_payload: Parsed Data Dragon item.json body, whose "data" key maps
            item id to a record carrying a "stats" object.
        item_bin: Parsed Community Dragon items.cdtb bin file, whose ItemData
            entries carry mDataValues.

    Returns:
        One scalar row per Data Dragon stat, then one per bin data value. The
        two sources share no value name. Within one source a repeated name keeps
        its first occurrence, which is what Bloodthirster's doubled DecayTime and
        Youmuu's doubled Cooldown need to satisfy the unique key; both publish
        the same number twice, so nothing is lost. An entry with no mValue means
        zero, these bins omitting any field at its default.
    """
    entries = {
        str(value["itemID"]): value
        for value in item_bin.values()
        if isinstance(value, dict) and value.get("__type") == ITEM_DATA_TYPE
    }

    rows: list[ItemValue] = []
    for item_id, record in item_payload["data"].items():
        taken: set[str] = set()
        for name, value in (record.get("stats") or {}).items():
            taken.add(name)
            rows.append(_item_value(item_id, name, float(value), SOURCE_DDRAGON))
        entry = entries.get(item_id)
        if entry is None:
            logger.warning("item %s has no bin entry", item_id)
            continue
        for data_value in entry.get("mDataValues") or []:
            name = data_value["mName"]
            if name in taken:
                logger.debug("skipping duplicate value %s on item %s", name, item_id)
                continue
            taken.add(name)
            rows.append(
                _item_value(item_id, name, float(data_value.get("mValue", 0.0)), SOURCE_CDRAGON)
            )
    return rows


def _item_value(item_id: str, name: str, value: float, source: str) -> ItemValue:
    """Build one scalar item value row.

    Args:
        item_id: Data Dragon item id the row hangs off.
        name: Source value name.
        value: The single numeric value.
        source: Origin API, one of ddragon, cdragon.

    Returns:
        An ItemValue of kind scalar. Neither source publishes a percentage hint
        for items, so display_as_percent stays False.
    """
    return ItemValue(
        item_id=item_id,
        name=name,
        kind=KIND_SCALAR,
        values=[value],
        display_as_percent=False,
        source=source,
    )


# ---------- orchestration ----------


@dataclass(frozen=True)
class ValueLoadStats:
    """Counts describing one numeric value load run.

    Args:
        ability_values: Number of ability value rows persisted.
        item_values: Number of item value rows persisted.
    """

    ability_values: int
    item_values: int


def _merge_all(session: Session, rows: Iterable[Base]) -> None:
    """Upsert every row through Session.merge.

    Args:
        session: Open Session the rows are merged into.
        rows: ORM instances whose primary keys are already set, so merge updates
            the existing row instead of inserting a duplicate.

    Returns:
        None. Nothing is flushed; the caller controls the transaction.
    """
    for row in rows:
        session.merge(row)


def _ability_ids_by_champion(session: Session) -> dict[str, dict[str, int]]:
    """Read the stored ability ids, keyed the way the Data Dragon corpus is.

    Args:
        session: Open Session queried for the stored abilities.

    Returns:
        Mapping of Data Dragon champion id to a mapping of slot to abilities.id.
        Lore-only characters carry no Data Dragon key and no abilities, so they
        never appear.
    """
    ids: dict[str, dict[str, int]] = defaultdict(dict)
    rows = session.execute(
        select(Champion.ddragon_key, Ability.slot, Ability.id)
        .join(Ability, Ability.champion_slug == Champion.slug)
        .where(Champion.ddragon_key.is_not(None))
    )
    for ddragon_key, slot, ability_id in rows:
        ids[ddragon_key][slot] = ability_id
    return dict(ids)


def _assign_existing_ability_value_ids(session: Session, rows: Sequence[AbilityValue]) -> None:
    """Give each built ability value the surrogate id its natural key already holds.

    Args:
        session: Open Session queried for the stored natural key to id mapping.
        rows: Built rows whose id is filled in when a row for the same ability,
            spell and name is already stored.

    Returns:
        None. Rows with no stored match keep id None and are inserted, which is
        what makes a repeated load idempotent despite the surrogate key.
    """
    stored = {
        (ability_id, key, name): value_id
        for value_id, ability_id, key, name in session.execute(
            select(
                AbilityValue.id,
                AbilityValue.ability_id,
                AbilityValue.spell_key,
                AbilityValue.name,
            )
        )
    }
    for row in rows:
        row.id = stored.get((row.ability_id, row.spell_key, row.name))


def _assign_existing_item_value_ids(session: Session, rows: Sequence[ItemValue]) -> None:
    """Give each built item value the surrogate id its natural key already holds.

    Args:
        session: Open Session queried for the stored natural key to id mapping.
        rows: Built rows whose id is filled in when a row for the same item and
            name is already stored.

    Returns:
        None. Rows with no stored match keep id None and are inserted, which is
        what makes a repeated load idempotent despite the surrogate key.
    """
    stored = {
        (item_id, name): value_id
        for value_id, item_id, name in session.execute(
            select(ItemValue.id, ItemValue.item_id, ItemValue.name)
        )
    }
    for row in rows:
        row.id = stored.get((row.item_id, row.name))


async def load_values(session: Session, client: FetchClient, settings: Settings) -> ValueLoadStats:
    """Populate ability_values and item_values from the corpus, upserting on natural keys.

    Args:
        session: Open Session the rows are merged into. Changes are flushed but
            never committed, so the caller decides whether the run is kept. The
            entity loader must have run first, because the ability and item rows
            these values hang off are looked up by their natural keys.
        client: Open FetchClient serving the corpus, from the on-disk cache when
            it is warm.
        settings: Application settings providing every ddragon_* and cdragon_*
            value plus cache_dir.

    Returns:
        ValueLoadStats with one count per table.

    Raises:
        httpx.HTTPStatusError: If any request fails after its retries.
        sqlalchemy.exc.SQLAlchemyError: If any row violates the schema.
        ValueError: If a Data Dragon spell cannot be joined to the bin.
    """
    champion_list, item_payload, item_bin = await asyncio.gather(
        ddragon.fetch_champion_list(client, settings),
        ddragon.fetch_items(client, settings),
        cdragon_bin.fetch_item_bin(client, settings),
    )
    champion_ids = list(champion_list["data"])
    champion_keys = [int(entry["key"]) for entry in champion_list["data"].values()]
    details, bins, records = await asyncio.gather(
        ddragon.fetch_all_champion_details(client, settings, champion_ids),
        cdragon_bin.fetch_all_champion_bins(client, settings, champion_ids),
        cdragon.fetch_all_champions(client, settings, champion_keys),
    )
    logger.info("loaded %d champion bins", len(bins))

    ability_ids = _ability_ids_by_champion(session)
    ability_values: list[AbilityValue] = []
    for ddragon_id, champion_key in zip(champion_ids, champion_keys, strict=True):
        slots = ability_ids.get(ddragon_id)
        if slots is None:
            logger.warning("champion %s has no stored abilities", ddragon_id)
            continue
        ability_values.extend(
            build_ability_values(
                ddragon_id,
                details[ddragon_id],
                bins[ddragon_id],
                records[champion_key],
                slots,
            )
        )
    _assign_existing_ability_value_ids(session, ability_values)
    _merge_all(session, ability_values)
    session.flush()

    item_values = build_item_values(item_payload, item_bin)
    _assign_existing_item_value_ids(session, item_values)
    _merge_all(session, item_values)
    session.flush()

    stats = ValueLoadStats(ability_values=len(ability_values), item_values=len(item_values))
    logger.info("loaded values: %s", stats)
    return stats
