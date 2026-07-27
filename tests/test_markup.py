import pytest

from lolrag.ingest.markup import clean_markup, clean_optional_markup

AHRI_Q_TOOLTIP = (
    "Ahri throws then pulls back her orb, dealing <magicDamage>{{ totaldamage }} magic damage"
    "</magicDamage> on the way out and <trueDamage>{{ totaldamage }} true damage</trueDamage> "
    "on the way back.{{ spellmodifierdescriptionappend }}"
)

BLOODTHIRSTER_DESCRIPTION = (
    "<mainText><stats><attention>70</attention> Attack Damage<br><attention>18%</attention> "
    "Life Steal</stats><br><br><passive>Ichorshield</passive><br>Convert excess healing from "
    "your <lifeSteal>Life Steal</lifeSteal> to a <shield>Shield</shield>.</mainText>"
)

ELECTROCUTE_LONG_DESC = (
    "Hitting a champion with 3 <b>separate</b> attacks or abilities within 3s deals bonus "
    "<lol-uikit-tooltipped-keyword key='LinkTooltip_Description_AdaptiveDmg'>"
    "<font color='#48C4B7'>adaptive damage</font></lol-uikit-tooltipped-keyword>.<br><br>"
    "Damage: 70 - 240 (+0.1 bonus AD, +0.05 AP) damage.<br>Cooldown: 20s<br><br>"
    "<i>'We called them the Thunderlords, for to speak of their lightning was to invite "
    "disaster.'</i>"
)

CORPUS_FIXTURES = [AHRI_Q_TOOLTIP, BLOODTHIRSTER_DESCRIPTION, ELECTROCUTE_LONG_DESC]


def test_ahri_q_tooltip() -> None:
    """Riot damage-type tags vanish while their inner text and placeholders remain."""
    assert clean_markup(AHRI_Q_TOOLTIP) == (
        "Ahri throws then pulls back her orb, dealing {{ totaldamage }} magic damage on the way "
        "out and {{ totaldamage }} true damage on the way back."
        "{{ spellmodifierdescriptionappend }}"
    )


def test_bloodthirster_description() -> None:
    """Item markup becomes stat lines separated by single and double newlines."""
    assert clean_markup(BLOODTHIRSTER_DESCRIPTION) == (
        "70 Attack Damage\n"
        "18% Life Steal\n"
        "\n"
        "Ichorshield\n"
        "Convert excess healing from your Life Steal to a Shield."
    )


def test_electrocute_long_desc() -> None:
    """Hyphenated keyword tags, font tags and br runs all resolve correctly."""
    assert clean_markup(ELECTROCUTE_LONG_DESC) == (
        "Hitting a champion with 3 separate attacks or abilities within 3s deals bonus "
        "adaptive damage.\n"
        "\n"
        "Damage: 70 - 240 (+0.1 bonus AD, +0.05 AP) damage.\n"
        "Cooldown: 20s\n"
        "\n"
        "'We called them the Thunderlords, for to speak of their lightning was to invite "
        "disaster.'"
    )


@pytest.mark.parametrize(
    "raw",
    [
        "<attention>70</attention>",
        "<keywordMajor>Grievous Wounds</keywordMajor>",
        "<scaleAP>ability power</scaleAP>",
        "<OnHit>on-hit</OnHit>",
        "<status>Slow</status>",
        "<spellName>Orb of Deception</spellName>",
        "<recast>Recast</recast>",
        "<rules>Rules text</rules>",
        "<a href='https://example.invalid/x'>link text</a>",
        '<span class="lore">lore text</span>',
        "<strike>old</strike>",
    ],
)
def test_unknown_tags_are_removed_and_inner_text_survives(raw: str) -> None:
    """Any tag is dropped without a whitelist, and the text it wrapped is kept."""
    cleaned = clean_markup(raw)

    assert "<" not in cleaned
    assert ">" not in cleaned
    assert cleaned != ""


def test_break_tags_produce_the_right_number_of_newlines() -> None:
    """A br is one newline and a p element is a paragraph break."""
    assert clean_markup("a<br>b") == "a\nb"
    assert clean_markup("a<br/>b") == "a\nb"
    assert clean_markup("a<br />b") == "a\nb"
    assert clean_markup("<p>first</p><p>second</p>") == "first\n\nsecond"
    assert clean_markup("one<hr>two") == "one\n\ntwo"
    assert clean_markup("<li>alpha<li>beta") == "alpha\nbeta"
    assert clean_markup("<li>alpha</li><li>beta</li>") == "alpha\n\nbeta"


def test_entities_are_decoded() -> None:
    """Numeric and named HTML entities present in the corpus decode to characters."""
    assert clean_markup("Ahri&#x2019;s orb") == "Ahri’s orb"
    assert clean_markup("don&apos;t") == "don't"
    assert clean_markup("Noxus&nbsp;rises") == "Noxus rises"
    assert clean_markup("Ionia&#x2014;the first land") == "Ionia—the first land"
    assert clean_markup("&#x201C;quoted") == "“quoted"


def test_tags_are_stripped_before_entities_are_decoded() -> None:
    """Escaped angle brackets in the source survive as literal text, not as markup."""
    cleaned = clean_markup("compare &lt;b&gt;this&lt;/b&gt; carefully")

    assert cleaned == "compare <b>this</b> carefully"
    assert "<b>" in cleaned
    assert "this" in cleaned


def test_hyphenated_tag_with_attributes_is_removed() -> None:
    """A lol-uikit tag with an underscored attribute value is removed whole."""
    raw = (
        "<lol-uikit-tooltipped-keyword key='LinkTooltip_Description_AdaptiveDmg'>"
        "adaptive damage</lol-uikit-tooltipped-keyword>"
    )

    assert clean_markup(raw) == "adaptive damage"


def test_placeholders_are_left_untouched() -> None:
    """Placeholder tokens are substituted downstream and must pass through verbatim."""
    assert clean_markup("deals {{ totaldamage }} damage") == "deals {{ totaldamage }} damage"
    assert clean_markup("<b>{{ e1 }}</b>") == "{{ e1 }}"


@pytest.mark.parametrize("raw", CORPUS_FIXTURES)
def test_cleaning_is_idempotent(raw: str) -> None:
    """Cleaning an already-cleaned corpus string changes nothing further."""
    once = clean_markup(raw)

    assert clean_markup(once) == once


def test_cleaning_is_not_idempotent_for_escaped_markup() -> None:
    """Escaped markup decodes into real markup, so a second pass would strip it.

    This is the documented consequence of stripping tags before decoding
    entities. Callers clean each raw source value exactly once.
    """
    once = clean_markup("compare &lt;b&gt;this&lt;/b&gt; carefully")

    assert clean_markup(once) == "compare this carefully"
    assert clean_markup(once) != once


def test_empty_and_none_inputs() -> None:
    """Empty text stays empty and a null column value stays null."""
    assert clean_markup("") == ""
    assert clean_markup("   \n\n  ") == ""
    assert clean_optional_markup(None) is None
    assert clean_optional_markup("") == ""
    assert clean_optional_markup("<b>Aatrox</b>") == "Aatrox"


def test_malformed_markup_does_not_raise_or_lose_words() -> None:
    """Unclosed tags are removed and a bare "<" is kept as literal text."""
    assert clean_markup("text <b>bold") == "text bold"
    assert clean_markup("a < b and c > d") == "a < b and c > d"
    assert clean_markup("</magicDamage> orphan close") == "orphan close"
    assert clean_markup("5 < 10 and 10 > 5") == "5 < 10 and 10 > 5"


def test_long_text_is_never_truncated() -> None:
    """A 50k-character paragraph comes through whole, minus only the markup."""
    filler = "Lorem ipsum dolor sit amet. " * 1800
    body = f"First sentence here. {filler}Final sentence here."
    raw = f"<p>{body}</p>"
    assert len(body) > 50_000

    cleaned = clean_markup(raw)

    assert cleaned.startswith("First sentence here.")
    assert cleaned.endswith("Final sentence here.")
    assert abs(len(cleaned) - (len(raw) - len("<p></p>"))) <= 2


def test_whitespace_is_normalised() -> None:
    """Horizontal runs collapse to one space and newline runs collapse to a blank line."""
    assert clean_markup("many     spaces\there") == "many spaces here"
    assert clean_markup("a<br><br><br><br>b") == "a\n\nb"
    assert clean_markup("a<p></p><p></p>b") == "a\n\nb"
    assert clean_markup("trailing   \nnext") == "trailing\nnext"
    assert clean_markup("  padded  ") == "padded"
    assert clean_markup("windows\r\nnewline") == "windows\nnewline"
