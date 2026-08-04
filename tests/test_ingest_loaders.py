import logging
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from lolrag.config import get_settings
from lolrag.db.models import (
    Ability,
    Champion,
    ChampionStats,
    Faction,
    Item,
    Role,
    Rune,
    RunePath,
    Story,
    SummonerSpell,
    champion_related,
    champion_role,
    item_components,
    item_tag,
    story_champion,
)
from lolrag.ingest.associations import AssociationStats
from lolrag.ingest.identifiers import universe_slug
from lolrag.ingest.loaders import (
    CDRAGON_VARIANT_OF_FIELD,
    UNAFFILIATED_SLUG,
    LoadStats,
    build_abilities,
    build_champion_stats,
    build_champions,
    build_factions,
    build_items,
    build_roles,
    build_rune_paths,
    build_stories,
    build_summoner_spells,
    item_display_name_ids,
    item_variant_ids,
    load_all,
    parse_release_date,
)
from lolrag.ingest.markup import clean_markup
from tests.test_fetch_client import build_client, build_settings

# ---------- raw markup fixtures ----------

NOXUS_OVERVIEW = "<p>Noxus is a powerful empire.</p><p>Strength &amp; ambition rule.</p>"
AATROX_BIO_FULL = (
    "<p>Whether mistaken for a demon or god.</p><p>Few know his &quot;real&quot; name.</p>"
)
AATROX_BIO_SHORT = "<p>Once honored defenders of Shurima.</p>"
AATROX_Q_TOOLTIP = (
    "Aatrox slams his greatsword, dealing <physicalDamage>{{ qdamage }}</physicalDamage>."
)
STORY_BLOCK_ONE = "<p>The blade was quiet.</p><p>Nobody spoke.</p>"
STORY_BLOCK_TWO = "<p>Then it sang &amp; screamed.</p>"
STORY_BLOCK_THREE = "<p>Nothing answered.</p>"

Q_DYNAMIC_DESCRIPTION = "Deals <physicalDamage>@QDamage@ physical damage</physicalDamage>."
W_DYNAMIC_DESCRIPTION = "Slows by @spell.AatroxQ:SlowAmount@."
R_DYNAMIC_DESCRIPTION = "Aatrox grows in size and revives once."


# ---------- payload fixtures ----------


def ddragon_champion_stats(hp: float) -> dict[str, float]:
    """Build the twenty-field Data Dragon stats block for one champion.

    Args:
        hp: Base health, varied per champion so the rows can be told apart.

    Returns:
        A stats object carrying every field the source publishes, with an
        integer base and a fractional per-level figure so both reach the row.
    """
    return {
        "hp": hp,
        "hpperlevel": 114,
        "mp": 0,
        "mpperlevel": 0,
        "movespeed": 345,
        "armor": 38,
        "armorperlevel": 4.8,
        "spellblock": 32,
        "spellblockperlevel": 2.05,
        "attackrange": 175,
        "hpregen": 3,
        "hpregenperlevel": 0.5,
        "mpregen": 0,
        "mpregenperlevel": 0,
        "crit": 0,
        "critperlevel": 0,
        "attackdamage": 60,
        "attackdamageperlevel": 5,
        "attackspeedperlevel": 2.5,
        "attackspeed": 0.651,
    }


def ddragon_champion_list() -> dict[str, Any]:
    """Build a Data Dragon champion.json body covering two playable champions.

    Returns:
        Payload whose "data" key maps champion id to a summary carrying the
        numeric key, the role tags and the base stats block.
    """
    return {
        "data": {
            "Aatrox": {
                "id": "Aatrox",
                "key": "266",
                "name": "Aatrox",
                "tags": ["Fighter"],
                "stats": ddragon_champion_stats(650),
            },
            "Renata": {
                "id": "Renata",
                "key": "888",
                "name": "Renata Glasc",
                "tags": ["Support", "Mage"],
                "stats": ddragon_champion_stats(544),
            },
        }
    }


ROSTER: dict[str, Any] = {"aatrox": {}, "renataglasc": {}, "norra": {}}


def ddragon_champion_detail(champion_id: str) -> dict[str, Any]:
    """Build a Data Dragon champion detail body with a passive and four spells.

    Args:
        champion_id: Data Dragon champion id the body is keyed on.

    Returns:
        Payload whose "data" key maps champion_id to a record with "passive"
        and a four-entry "spells" list carrying distinct maxrank values and the
        spell ids the Community Dragon bin is joined on.
    """
    return {
        "data": {
            champion_id: {
                "id": champion_id,
                "passive": {
                    "name": f"{champion_id} Passive",
                    "description": "A passive that never ranks up.",
                },
                "spells": [
                    {
                        "id": f"{champion_id}Q",
                        "name": f"{champion_id} Q",
                        "description": "Q description.",
                        "tooltip": AATROX_Q_TOOLTIP,
                        "maxrank": 5,
                    },
                    {
                        "id": f"{champion_id}W",
                        "name": f"{champion_id} W",
                        "description": "W description.",
                        "tooltip": "W tooltip.",
                        "maxrank": 5,
                    },
                    {
                        "id": f"{champion_id}E",
                        "name": f"{champion_id} E",
                        "description": "E description.",
                        "tooltip": "E tooltip.",
                        "maxrank": 6,
                    },
                    {
                        "id": f"{champion_id}R",
                        "name": f"{champion_id} R",
                        "description": "R description.",
                        "tooltip": "R tooltip.",
                        "maxrank": 3,
                    },
                ],
            }
        }
    }


def cdragon_champion_record(champion_id: str) -> dict[str, Any]:
    """Build a Community Dragon champion record covering all four tooltip outcomes.

    Args:
        champion_id: Data Dragon champion id the record belongs to.

    Returns:
        Payload whose "spells" list holds, in Q/W/E/R order, a tooltip that
        resolves from a data value, one blocked by a cross-spell token, one that
        publishes no dynamicDescription at all, and one that carries no token.
    """
    return {
        "id": champion_id,
        "spells": [
            {"spellKey": "q", "dynamicDescription": Q_DYNAMIC_DESCRIPTION},
            {"spellKey": "w", "dynamicDescription": W_DYNAMIC_DESCRIPTION},
            {"spellKey": "e"},
            {"spellKey": "r", "dynamicDescription": R_DYNAMIC_DESCRIPTION},
        ],
    }


def cdragon_champion_bin(champion_id: str) -> dict[str, Any]:
    """Build a Community Dragon champion bin holding one champion's spell objects.

    Args:
        champion_id: Data Dragon champion id the bin belongs to.

    Returns:
        Payload with the root CharacterRecord naming the passive spell, an
        AbilityObject per slot and a SpellObject under each, the Q spell
        publishing the seven-wide one-indexed QDamage array its tooltip names.
    """
    spells = f"Characters/{champion_id}/Spells"
    passive_key = f"{spells}/{champion_id}PAbility/{champion_id}P"
    payload: dict[str, Any] = {
        f"Characters/{champion_id}/CharacterRecords/Root": {
            "__type": "CharacterRecord",
            "mCharacterPassiveSpell": passive_key,
        },
        f"{spells}/{champion_id}PAbility": {"__type": "AbilityObject"},
        passive_key: {"__type": "SpellObject", "mSpell": {}},
    }
    for slot in ("Q", "W", "E", "R"):
        payload[f"{spells}/{champion_id}{slot}Ability"] = {"__type": "AbilityObject"}
        payload[f"{spells}/{champion_id}{slot}Ability/{champion_id}{slot}"] = {
            "__type": "SpellObject",
            "mSpell": {"DataValues": [{"name": "QDamage", "values": [0, 10, 25, 40, 55, 70, 70]}]},
        }
    return payload


def universe_champion(
    slug: str, *, faction_slug: str, related: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Build a Universe champion payload with a biography and one story module.

    Args:
        slug: Universe champion slug the payload describes.
        faction_slug: Value stored under "associated-faction-slug"; an empty
            string reproduces a champion with no published faction.
        related: Slugs published under the top-level "related-champions" array.

    Returns:
        Payload shaped like a real Universe champion response, whose modules
        list references the champion's own story.
    """
    return {
        "champion": {
            "slug": slug,
            "name": slug.capitalize(),
            "title": f"the {slug.capitalize()}",
            "associated-faction-slug": faction_slug,
            "release-date": "2013-06-13T19:43:26.000Z",
            "biography": {"full": AATROX_BIO_FULL, "short": AATROX_BIO_SHORT},
        },
        "related-champions": [{"slug": entry} for entry in related],
        "modules": [{"type": "story-preview", "story-slug": f"{slug}-story"}],
    }


def universe_faction(slug: str) -> dict[str, Any]:
    """Build a Universe faction payload carrying an overview.

    Args:
        slug: Universe faction slug the payload describes.

    Returns:
        Payload whose "faction" object holds slug, name and overview.short.
    """
    return {
        "faction": {
            "slug": slug,
            "name": slug.capitalize(),
            "overview": {"short": NOXUS_OVERVIEW},
        }
    }


def universe_story(slug: str) -> dict[str, Any]:
    """Build a Universe story payload with two sections and three subsections.

    Args:
        slug: Universe story slug the payload describes.

    Returns:
        Payload whose second section carries a subsection with no content, so
        the two-level walk has an empty block to skip.
    """
    return {
        "id": slug,
        "release-date": "2018-04-02T15:00:00.000Z",
        "word-count": 9,
        "story": {
            "title": f"Story {slug}",
            "subtitle": "By Someone Uncredited",
            "story-sections": [
                {
                    "title": None,
                    "story-subsections": [
                        {"content": STORY_BLOCK_ONE},
                        {"content": STORY_BLOCK_TWO},
                    ],
                },
                {
                    "title": None,
                    "story-subsections": [
                        {"content": None},
                        {"content": STORY_BLOCK_THREE},
                    ],
                },
            ],
        },
    }


def ddragon_items() -> dict[str, Any]:
    """Build a Data Dragon item.json body with a recipe that repeats a component.

    Returns:
        Payload whose "data" key maps item id to a record; item 3035 lists
        component 1036 twice and item 1036 has neither components nor depth.
    """
    return {
        "data": {
            "1036": {
                "name": "Long Sword",
                "description": "<mainText><stats><attention>10</attention> AD</stats></mainText>",
                "plaintext": "Slightly increases Attack Damage",
                "gold": {"base": 350, "total": 350},
                "tags": ["Damage"],
            },
            "3035": {
                "name": "Last Whisper",
                "description": "<mainText>Armor Penetration &amp; more</mainText>",
                "plaintext": "",
                "from": ["1036", "1036"],
                "depth": 2,
                "gold": {"base": 500, "total": 1200},
                "tags": ["Damage", "ArmorPenetration"],
            },
        }
    }


def cdragon_item_bin() -> dict[str, Any]:
    """Build an items.cdtb bin body carrying both Community Dragon variant assertions.

    Returns:
        Payload keyed "Items/{id}"; 323035 declares itself a variant of 3035 and
        is also published under 3035's name key, 663035 is published under
        another id's name key without declaring a variant, and 1036 and 3035
        declare neither. The non-item key proves the reader ignores it.
    """
    return {
        "Items/1036": {"itemID": 1036, "mDisplayName": "Item_1036_Name", "__type": "ItemData"},
        "Items/3035": {"itemID": 3035, "mDisplayName": "Item_3035_Name", "__type": "ItemData"},
        "Items/323035": {
            "itemID": 323035,
            "mDisplayName": "Item_3035_Name",
            "{4f958685}": 3035,
            "__type": "ItemData",
        },
        "Items/663035": {"itemID": 663035, "mDisplayName": "Item_3035_Name", "__type": "ItemData"},
        "Maps/Map11": {"__type": "MapData"},
    }


def ddragon_runes() -> list[dict[str, Any]]:
    """Build a Data Dragon runesReforged.json body with one path and two rows.

    Returns:
        A single path record whose slots hold two runes and one rune, so both
        row_index and position_index are exercised.
    """
    return [
        {
            "id": 8100,
            "key": "Domination",
            "name": "Domination",
            "slots": [
                {
                    "runes": [
                        {
                            "id": "8112",
                            "key": "Electrocute",
                            "name": "Electrocute",
                            "shortDesc": "Hitting a champion with <b>3</b> attacks.",
                            "longDesc": "Hitting a champion &amp; dealing damage.",
                        },
                        {
                            "id": "8124",
                            "key": "Predator",
                            "name": "Predator",
                            "shortDesc": "Enchant your boots.",
                            "longDesc": "Enchant your boots &amp; sprint.",
                        },
                    ]
                },
                {
                    "runes": [
                        {
                            "id": "8126",
                            "key": "CheapShot",
                            "name": "Cheap Shot",
                            "shortDesc": "Deal bonus true damage.",
                            "longDesc": "Deal bonus true damage to impaired targets.",
                        }
                    ]
                },
            ],
        }
    ]


def ddragon_summoner_spells() -> dict[str, Any]:
    """Build a Data Dragon summoner.json body covering the id and cooldown edge cases.

    Returns:
        Payload whose "data" key maps spell id to a record: Flash with a whole
        cooldown, an Arena spell whose id exceeds sixteen characters and whose
        cooldown is fractional, and the two engine placeholder spells.
    """
    return {
        "data": {
            "SummonerFlash": {
                "id": "SummonerFlash",
                "key": "4",
                "name": "Flash",
                "description": "Teleports your champion a short distance <b>instantly</b>.",
                "cooldown": [300],
                "summonerLevel": 7,
                "modes": ["NEXUSBLITZ", "CLASSIC", "ARAM", "CLASSIC"],
            },
            "SummonerCherryFlash": {
                "id": "SummonerCherryFlash",
                "key": "2202",
                "name": "Flash",
                "description": "Teleports your champion a short distance.",
                "cooldown": [0.25],
                "summonerLevel": 1,
                "modes": ["CHERRY"],
            },
            "Summoner_UltBookPlaceholder": {
                "id": "Summoner_UltBookPlaceholder",
                "key": "54",
                "name": "Placeholder",
                "description": "Placeholder ultimate spellbook spell.",
                "cooldown": [0],
                "summonerLevel": 1,
            },
            "Summoner_UltBookSmitePlaceholder": {
                "id": "Summoner_UltBookSmitePlaceholder",
                "key": "55",
                "name": "Placeholder and Attack-Smite",
                "description": "Placeholder ultimate spellbook smite.",
                "cooldown": [0],
                "summonerLevel": 1,
            },
        }
    }


# ---------- builder tests ----------


def test_universe_slug_lowercases_and_overrides_renata() -> None:
    """The Universe slug is the lowercased id except for the Renata override."""
    assert universe_slug("Aatrox") == "aatrox"
    assert universe_slug("MonkeyKing") == "monkeyking"
    assert universe_slug("Renata") == "renataglasc"


def test_parse_release_date_returns_naive_utc() -> None:
    """A Zulu timestamp becomes a naive UTC datetime and junk becomes None."""
    assert parse_release_date("2013-06-13T19:43:26.000Z") == datetime(2013, 6, 13, 19, 43, 26)
    assert parse_release_date("") is None
    assert parse_release_date(None) is None
    assert parse_release_date("not a date") is None


def test_build_factions_puts_the_synthetic_unaffiliated_faction_first() -> None:
    """The synthetic faction leads the list so the champion foreign key resolves."""
    factions = build_factions([universe_faction("noxus")])

    assert [faction.slug for faction in factions] == [UNAFFILIATED_SLUG, "noxus"]
    assert factions[0].overview is None
    assert factions[0].overview_text is None


def test_build_factions_reads_overview_from_the_short_key() -> None:
    """Faction overview comes from overview.short and overview_text is its cleaned form."""
    _, noxus = build_factions([universe_faction("noxus")])

    assert noxus.overview == NOXUS_OVERVIEW
    assert noxus.overview_text == clean_markup(NOXUS_OVERVIEW)
    assert noxus.overview_text == "Noxus is a powerful empire.\n\nStrength & ambition rule."


def test_build_factions_never_duplicates_a_published_unaffiliated_faction() -> None:
    """A source faction already slugged "unaffiliated" does not become a second row."""
    factions = build_factions([universe_faction(UNAFFILIATED_SLUG)])

    assert [faction.slug for faction in factions] == [UNAFFILIATED_SLUG]


def test_build_roles_emits_one_row_per_distinct_tag() -> None:
    """Role slugs are the lowercased Data Dragon tags, deduplicated and sorted."""
    roles = build_roles(ddragon_champion_list())

    assert [(role.slug, role.name) for role in roles] == [
        ("fighter", "Fighter"),
        ("mage", "Mage"),
        ("support", "Support"),
    ]


def test_build_champions_falls_back_to_unaffiliated_when_no_faction_is_published() -> None:
    """A champion with an empty faction slug is attached to the synthetic faction."""
    champions = build_champions(
        ddragon_champion_list(), {"aatrox": universe_champion("aatrox", faction_slug="")}
    )

    assert champions[0].faction_slug == UNAFFILIATED_SLUG


def test_build_champions_keeps_a_published_faction_slug() -> None:
    """A champion that publishes a faction slug keeps it untouched."""
    champions = build_champions(
        ddragon_champion_list(), {"aatrox": universe_champion("aatrox", faction_slug="noxus")}
    )

    assert champions[0].faction_slug == "noxus"


def test_build_champions_maps_the_renata_slug_override_to_its_ddragon_id() -> None:
    """The renataglasc Universe slug resolves to the Renata Data Dragon id."""
    champions = build_champions(
        ddragon_champion_list(),
        {"renataglasc": universe_champion("renataglasc", faction_slug="zaun")},
    )

    assert champions[0].slug == "renataglasc"
    assert champions[0].ddragon_key == "Renata"
    assert champions[0].playable is True


def test_build_champions_marks_a_lore_only_character_unplayable() -> None:
    """A character present in Universe but absent from Data Dragon is lore-only."""
    champions = build_champions(
        ddragon_champion_list(),
        {"norra": universe_champion("norra", faction_slug="bandle-city")},
    )

    assert champions[0].playable is False
    assert champions[0].ddragon_key is None


def test_build_champions_stores_raw_bio_verbatim_and_cleans_only_the_text_columns() -> None:
    """bio_full stays byte-identical while bio_full_text is its cleaned form."""
    champions = build_champions(
        ddragon_champion_list(), {"aatrox": universe_champion("aatrox", faction_slug="noxus")}
    )
    champion = champions[0]

    assert champion.bio_full == AATROX_BIO_FULL
    assert champion.bio_short == AATROX_BIO_SHORT
    assert champion.bio_full_text == clean_markup(AATROX_BIO_FULL)
    assert champion.bio_full_text == (
        'Whether mistaken for a demon or god.\n\nFew know his "real" name.'
    )
    assert champion.bio_short_text == "Once honored defenders of Shurima."


def test_build_champions_parses_the_release_date() -> None:
    """The Universe release date becomes a naive UTC datetime."""
    champions = build_champions(
        ddragon_champion_list(), {"aatrox": universe_champion("aatrox", faction_slug="noxus")}
    )

    assert champions[0].release_date == datetime(2013, 6, 13, 19, 43, 26)


def build_one_champions_abilities(champion_id: str) -> list[Ability]:
    """Build one champion's ability rows from the three fixture payloads.

    Args:
        champion_id: Data Dragon champion id every payload is built for.

    Returns:
        The five Ability rows build_abilities produces for that champion.
    """
    return build_abilities(
        {champion_id: ddragon_champion_detail(champion_id)},
        {champion_id: cdragon_champion_bin(champion_id)},
        {champion_id: cdragon_champion_record(champion_id)},
    )


def test_build_abilities_gives_the_passive_slot_p_and_no_rank() -> None:
    """The passive lands in slot P with max_rank and every tooltip column None."""
    passive = build_one_champions_abilities("Aatrox")[0]

    assert passive.slot == "P"
    assert passive.max_rank is None
    assert passive.tooltip is None
    assert passive.tooltip_text is None
    assert passive.tooltip_resolved is None
    assert passive.description == "A passive that never ranks up."


def test_build_abilities_orders_spells_into_q_w_e_r_with_their_maxrank() -> None:
    """Spells take slots Q, W, E and R in source order, each keeping its maxrank."""
    abilities = build_one_champions_abilities("Aatrox")

    assert [(ability.slot, ability.max_rank) for ability in abilities] == [
        ("P", None),
        ("Q", 5),
        ("W", 5),
        ("E", 6),
        ("R", 3),
    ]


def test_build_abilities_stores_raw_tooltip_verbatim_and_cleans_tooltip_text() -> None:
    """tooltip keeps the source markup while tooltip_text is its cleaned form."""
    spell = build_one_champions_abilities("Aatrox")[1]

    assert spell.tooltip == AATROX_Q_TOOLTIP
    assert spell.tooltip_text == clean_markup(AATROX_Q_TOOLTIP)
    assert spell.tooltip_text == "Aatrox slams his greatsword, dealing {{ qdamage }}."


def test_build_abilities_substitutes_the_community_dragon_tooltip() -> None:
    """A resolvable tooltip stores the substituted, markup-stripped text."""
    spell = build_one_champions_abilities("Aatrox")[1]

    assert spell.tooltip_resolved == "Deals 10/25/40/55/70 physical damage."


def test_build_abilities_stores_null_for_a_blocked_tooltip() -> None:
    """A tooltip naming another spell's value is dropped whole, never half filled."""
    spell = build_one_champions_abilities("Aatrox")[2]

    assert spell.slot == "W"
    assert spell.tooltip_resolved is None


def test_build_abilities_stores_null_when_no_dynamic_description_is_published() -> None:
    """A spell the source publishes no dynamicDescription for resolves to nothing."""
    spell = build_one_champions_abilities("Aatrox")[3]

    assert spell.slot == "E"
    assert spell.tooltip_resolved is None


def test_build_abilities_resolves_a_tooltip_that_carries_no_token() -> None:
    """A tooltip with nothing to substitute is still stored, cleaned."""
    spell = build_one_champions_abilities("Aatrox")[4]

    assert spell.slot == "R"
    assert spell.tooltip_resolved == R_DYNAMIC_DESCRIPTION


def test_build_abilities_uses_the_universe_slug_for_the_champion_key() -> None:
    """Abilities are keyed on the Universe slug, so the Renata override applies."""
    abilities = build_one_champions_abilities("Renata")

    assert {ability.champion_slug for ability in abilities} == {"renataglasc"}


def test_build_abilities_rejects_a_champion_without_exactly_four_spells() -> None:
    """A spell list that does not fill Q, W, E and R fails loudly."""
    detail = ddragon_champion_detail("Aatrox")
    detail["data"]["Aatrox"]["spells"] = detail["data"]["Aatrox"]["spells"][:3]

    with pytest.raises(ValueError):
        build_abilities(
            {"Aatrox": detail},
            {"Aatrox": cdragon_champion_bin("Aatrox")},
            {"Aatrox": cdragon_champion_record("Aatrox")},
        )


def test_build_items_maps_gold_plaintext_and_depth() -> None:
    """Item gold comes from the gold object and depth stays None when absent."""
    items = {item.ddragon_id: item for item in build_items(ddragon_items(), cdragon_item_bin())}

    assert items["3035"].gold_total == 1200
    assert items["3035"].gold_base == 500
    assert items["3035"].depth == 2
    assert items["1036"].depth is None
    assert items["1036"].plaintext == "Slightly increases Attack Damage"
    assert items["3035"].plaintext == ""


def test_build_items_stores_raw_description_verbatim_and_cleans_description_text() -> None:
    """description keeps the source markup while description_text is its cleaned form."""
    items = {item.ddragon_id: item for item in build_items(ddragon_items(), cdragon_item_bin())}
    raw = ddragon_items()["data"]["3035"]["description"]

    assert items["3035"].description == raw
    assert items["3035"].description_text == clean_markup(raw)
    assert items["3035"].description_text == "Armor Penetration & more"


def test_item_variant_ids_reads_only_the_declared_variants() -> None:
    """The hashed field is the only variant assertion, and it is read as a string id."""
    assert item_variant_ids(cdragon_item_bin()) == {"323035": "3035"}


def test_item_variant_ids_warns_when_the_hashed_field_matches_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A rehashed field would silently unfilter the corpus, so it must be visible."""
    payload = {"Items/3035": {"itemID": 3035, "mDisplayName": "Item_3035_Name"}}

    with caplog.at_level(logging.WARNING):
        assert item_variant_ids(payload) == {}

    assert CDRAGON_VARIANT_OF_FIELD in caplog.text


def test_item_display_name_ids_reports_only_the_records_published_under_another_id() -> None:
    """A record naming its own id publishes nothing; another id's name key is recorded."""
    assert item_display_name_ids(cdragon_item_bin()) == {"323035": "3035", "663035": "3035"}


def test_build_items_carries_both_community_dragon_variant_assertions() -> None:
    """Both columns stay NULL for an item Community Dragon says nothing special about."""
    items = {item.ddragon_id: item for item in build_items(ddragon_items(), cdragon_item_bin())}

    assert items["3035"].variant_of_id is None
    assert items["3035"].display_name_id is None


def test_build_rune_paths_keeps_data_dragon_ids_and_positions() -> None:
    """Path and rune ids come from Data Dragon and positions follow the slot layout."""
    paths = build_rune_paths(ddragon_runes())
    path = paths[0]

    assert path.id == 8100
    assert path.key == "Domination"
    assert [(rune.id, rune.row_index, rune.position_index) for rune in path.runes] == [
        (8112, 0, 0),
        (8124, 0, 1),
        (8126, 1, 0),
    ]


def test_build_rune_paths_cleans_both_description_columns() -> None:
    """Rune short and long descriptions keep their raw form beside a cleaned one."""
    path = build_rune_paths(ddragon_runes())[0]
    rune = path.runes[0]
    source = ddragon_runes()[0]["slots"][0]["runes"][0]

    assert rune.short_desc == source["shortDesc"]
    assert rune.long_desc == source["longDesc"]
    assert rune.short_desc_text == "Hitting a champion with 3 attacks."
    assert rune.long_desc_text == "Hitting a champion & dealing damage."


def test_build_summoner_spells_takes_the_first_cooldown_entry() -> None:
    """Cooldown is element zero of the cooldown list and description_text is cleaned."""
    spells = build_summoner_spells(ddragon_summoner_spells())
    spell = spells[0]

    assert spell.id == "SummonerFlash"
    assert spell.key == "4"
    assert spell.cooldown == 300
    assert spell.summoner_level == 7
    assert spell.description == ddragon_summoner_spells()["data"]["SummonerFlash"]["description"]
    assert spell.description_text == "Teleports your champion a short distance instantly."


def test_build_summoner_spells_keeps_a_fractional_cooldown() -> None:
    """A cooldown below one second survives as a float instead of truncating to zero."""
    spells = {spell.id: spell for spell in build_summoner_spells(ddragon_summoner_spells())}

    assert spells["SummonerCherryFlash"].cooldown == 0.25


def test_build_summoner_spells_skips_placeholder_entries() -> None:
    """Engine placeholder spells are dropped while real Arena spells are kept."""
    spells = build_summoner_spells(ddragon_summoner_spells())

    assert [spell.id for spell in spells] == ["SummonerFlash", "SummonerCherryFlash"]


def test_build_summoner_spells_sorts_and_deduplicates_the_modes() -> None:
    """The source order varies between spells, so a stable sorted set is stored instead."""
    spells = {spell.id: spell for spell in build_summoner_spells(ddragon_summoner_spells())}

    assert spells["SummonerFlash"].modes == ["ARAM", "CLASSIC", "NEXUSBLITZ"]
    assert spells["SummonerCherryFlash"].modes == ["CHERRY"]


def test_build_champion_stats_reads_every_published_field() -> None:
    """All twenty fields reach the row, bases and per-level figures alike."""
    rows = {row.champion_slug: row for row in build_champion_stats(ddragon_champion_list(), ROSTER)}
    aatrox = rows["aatrox"]

    assert aatrox.hp == 650
    assert aatrox.hp_per_level == 114
    assert aatrox.attack_speed == pytest.approx(0.651)
    assert aatrox.attack_speed_per_level == 2.5
    assert aatrox.spell_block_per_level == 2.05
    assert aatrox.move_speed == 345


def test_build_champion_stats_keys_rows_on_the_universe_slug() -> None:
    """Stats join to the champion table the same way build_champions does."""
    rows = build_champion_stats(ddragon_champion_list(), ROSTER)

    assert sorted(row.champion_slug for row in rows) == ["aatrox", "renataglasc"]


def test_build_champion_stats_skips_a_champion_that_is_not_on_the_roster() -> None:
    """A Data Dragon champion with no Universe page would dangle, so it yields no row."""
    rows = build_champion_stats(ddragon_champion_list(), {"aatrox": {}})

    assert [row.champion_slug for row in rows] == ["aatrox"]


def test_build_stories_walks_sections_then_subsections() -> None:
    """Both levels are walked and subsections without content are skipped."""
    stories = build_stories({"aatrox-story": universe_story("aatrox-story")})
    story = stories[0]

    assert story.subsection_count == 3
    assert story.content == "\n\n".join([STORY_BLOCK_ONE, STORY_BLOCK_TWO, STORY_BLOCK_THREE])


def test_build_stories_separates_subsections_from_paragraphs_in_content_text() -> None:
    """Cleaned blocks join on three newlines while a paragraph break inside one stays two."""
    story = build_stories({"aatrox-story": universe_story("aatrox-story")})[0]

    assert story.content_text == (
        "The blade was quiet.\n\nNobody spoke.\n\n\nThen it sang & screamed.\n\n\nNothing answered."
    )
    assert story.content_text.count("\n\n\n") == 2


def test_build_stories_counts_words_of_the_cleaned_content() -> None:
    """word_count counts whitespace-separated words of the cleaned text."""
    story = build_stories({"aatrox-story": universe_story("aatrox-story")})[0]

    assert story.word_count == 13


def test_build_stories_leaves_author_null() -> None:
    """No permitted source publishes a story author, so the column stays NULL."""
    story = build_stories({"aatrox-story": universe_story("aatrox-story")})[0]

    assert story.author is None
    assert story.title == "Story aatrox-story"
    assert story.release_date == datetime(2018, 4, 2, 15, 0, 0)


# ---------- orchestrator harness ----------

DDRAGON_DATA_PATH = "/cdn/16.14.1/data/en_US"
UNIVERSE_PATH = "/v1/en_us"
CDRAGON_BIN_PATH = "/latest/game/data/characters"
CDRAGON_ITEM_BIN_PATH = "/latest/game/items.cdtb.bin.json"
CDRAGON_CHAMPION_PATH = "/latest/plugins/rcp-be-lol-game-data/global/default/v1/champions"

CHAMPION_SLUGS = ("aatrox", "renataglasc", "norra")
FACTION_SLUGS = ("noxus",)

PLAYABLE_CHAMPION_KEYS = {"Aatrox": 266, "Renata": 888}

EXPECTED_STATS = LoadStats(
    factions=2,
    roles=3,
    champions=3,
    champion_stats=2,
    abilities=10,
    stories=3,
    items=2,
    rune_paths=1,
    runes=3,
    summoner_spells=2,
    unaffiliated_champions=1,
    tooltips_resolved=4,
    tooltips_unresolved=4,
    associations=AssociationStats(
        champion_roles=3,
        champion_related=2,
        story_champions=3,
        item_tags=3,
        item_maps=0,
        item_components=1,
        dropped_edges=0,
    ),
)


def build_routes() -> dict[str, Any]:
    """Build the URL-path to response-body map covering every endpoint load_all touches.

    Returns:
        Mapping of URL path to the JSON body served for it, spanning two
        playable champions, one lore-only character, one faction and three
        stories.
    """
    routes: dict[str, Any] = {
        f"{DDRAGON_DATA_PATH}/champion.json": ddragon_champion_list(),
        f"{DDRAGON_DATA_PATH}/item.json": ddragon_items(),
        f"{DDRAGON_DATA_PATH}/runesReforged.json": ddragon_runes(),
        f"{DDRAGON_DATA_PATH}/summoner.json": ddragon_summoner_spells(),
        CDRAGON_ITEM_BIN_PATH: cdragon_item_bin(),
        f"{UNIVERSE_PATH}/search/index.json": {
            "champions": [{"slug": slug} for slug in CHAMPION_SLUGS],
            "factions": [{"slug": slug} for slug in FACTION_SLUGS],
        },
    }
    for champion_id, champion_key in PLAYABLE_CHAMPION_KEYS.items():
        routes[f"{DDRAGON_DATA_PATH}/champion/{champion_id}.json"] = ddragon_champion_detail(
            champion_id
        )
        slug = champion_id.lower()
        routes[f"{CDRAGON_BIN_PATH}/{slug}/{slug}.bin.json"] = cdragon_champion_bin(champion_id)
        routes[f"{CDRAGON_CHAMPION_PATH}/{champion_key}.json"] = cdragon_champion_record(
            champion_id
        )
    factions = {"aatrox": "noxus", "renataglasc": "noxus", "norra": ""}
    related = {"aatrox": ("renataglasc",), "renataglasc": (), "norra": ("aatrox",)}
    for slug, faction_slug in factions.items():
        routes[f"{UNIVERSE_PATH}/champions/{slug}/index.json"] = universe_champion(
            slug, faction_slug=faction_slug, related=related[slug]
        )
        routes[f"{UNIVERSE_PATH}/story/{slug}-story/index.json"] = universe_story(f"{slug}-story")
    for slug in FACTION_SLUGS:
        routes[f"{UNIVERSE_PATH}/factions/{slug}/index.json"] = universe_faction(slug)
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
        A Session against the configured database; every change made through
        it is discarded when the test finishes, so the database stays empty.
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


def row_counts(session: Session) -> dict[str, int]:
    """Count the rows every loader writes.

    Args:
        session: Session the counts are read through.

    Returns:
        Mapping of table label to row count, covering the entity tables and
        every association table the run rewrites.
    """
    entities = {
        "factions": Faction,
        "roles": Role,
        "champions": Champion,
        "champion_stats": ChampionStats,
        "abilities": Ability,
        "stories": Story,
        "items": Item,
        "rune_paths": RunePath,
        "runes": Rune,
        "summoner_spells": SummonerSpell,
    }
    associations = {
        "champion_role": champion_role,
        "champion_related": champion_related,
        "story_champion": story_champion,
        "item_tag": item_tag,
        "item_components": item_components,
    }
    counts = {
        label: session.execute(select(func.count()).select_from(model)).scalar_one()
        for label, model in entities.items()
    }
    counts.update(
        {
            label: session.execute(select(func.count()).select_from(table)).scalar_one()
            for label, table in associations.items()
        }
    )
    return counts


EXPECTED_COUNTS = {
    "factions": 2,
    "roles": 3,
    "champions": 3,
    "champion_stats": 2,
    "abilities": 10,
    "stories": 3,
    "items": 2,
    "rune_paths": 1,
    "runes": 3,
    "summoner_spells": 2,
    "champion_role": 3,
    "champion_related": 2,
    "story_champion": 3,
    "item_tag": 3,
    "item_components": 1,
}


# ---------- orchestrator tests ----------


async def test_load_all_persists_every_entity_table(db_session: Session, tmp_path: Path) -> None:
    """One run lands the fixture corpus in every table and reports matching counts."""
    settings = build_settings(tmp_path)
    async with build_client(settings, corpus_handler()) as client:
        stats = await load_all(db_session, client, settings)

    assert stats == EXPECTED_STATS
    assert row_counts(db_session) == EXPECTED_COUNTS


async def test_load_all_persists_the_resolved_tooltip(db_session: Session, tmp_path: Path) -> None:
    """The substituted Community Dragon tooltip reaches the stored ability row."""
    settings = build_settings(tmp_path)
    async with build_client(settings, corpus_handler()) as client:
        await load_all(db_session, client, settings)
    db_session.expire_all()

    stored = {
        slot: resolved
        for slot, resolved in db_session.execute(
            select(Ability.slot, Ability.tooltip_resolved).where(Ability.champion_slug == "aatrox")
        )
    }
    assert stored == {
        "P": None,
        "Q": "Deals 10/25/40/55/70 physical damage.",
        "W": None,
        "E": None,
        "R": R_DYNAMIC_DESCRIPTION,
    }


async def test_load_all_persists_the_component_quantity_and_keeps_traversal(
    db_session: Session, tmp_path: Path
) -> None:
    """A doubled component lands as one row with quantity two, still traversable."""
    settings = build_settings(tmp_path)
    async with build_client(settings, corpus_handler()) as client:
        await load_all(db_session, client, settings)

    rows = db_session.execute(
        select(
            item_components.c.item_id,
            item_components.c.component_id,
            item_components.c.quantity,
        )
    ).all()
    assert rows == [("3035", "1036", 2)]

    last_whisper = db_session.get(Item, "3035")
    long_sword = db_session.get(Item, "1036")
    assert last_whisper is not None
    assert long_sword is not None
    assert last_whisper.components == [long_sword]
    assert long_sword.builds_into == [last_whisper]


async def test_load_all_persists_a_long_summoner_spell_id_and_fractional_cooldown(
    db_session: Session, tmp_path: Path
) -> None:
    """The Arena spell keeps its over-long id and its sub-second cooldown."""
    settings = build_settings(tmp_path)
    async with build_client(settings, corpus_handler()) as client:
        await load_all(db_session, client, settings)
    db_session.expire_all()

    stored = db_session.get(SummonerSpell, "SummonerCherryFlash")
    assert stored is not None
    assert stored.cooldown == 0.25
    assert db_session.get(SummonerSpell, "Summoner_UltBookPlaceholder") is None


async def test_load_all_resolves_the_unaffiliated_faction_foreign_key(
    db_session: Session, tmp_path: Path
) -> None:
    """The champion with no published faction points at the synthetic faction row."""
    settings = build_settings(tmp_path)
    async with build_client(settings, corpus_handler()) as client:
        await load_all(db_session, client, settings)

    norra = db_session.get(Champion, "norra")
    assert norra is not None
    assert norra.faction_slug == UNAFFILIATED_SLUG
    assert norra.faction.name == "Unaffiliated"
    assert norra.playable is False
    assert norra.ddragon_key is None


async def test_load_all_run_twice_is_idempotent(db_session: Session, tmp_path: Path) -> None:
    """A second run over the same corpus updates rows in place instead of duplicating."""
    settings = build_settings(tmp_path)
    async with build_client(settings, corpus_handler()) as client:
        first = await load_all(db_session, client, settings)
        after_first = row_counts(db_session)
        ability_ids = sorted(db_session.execute(select(Ability.id)).scalars())

        second = await load_all(db_session, client, settings)

    assert second == first
    assert row_counts(db_session) == after_first
    assert sorted(db_session.execute(select(Ability.id)).scalars()) == ability_ids
