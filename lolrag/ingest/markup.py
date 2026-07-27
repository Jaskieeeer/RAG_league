import html
import re

# ---------- patterns ----------

_TAG_NAME = r"[a-zA-Z][a-zA-Z0-9_-]*"
_TAG_TAIL = r"(?:\s+[^<>]*?)?/?\s*>"

PARAGRAPH_TAG_RE = re.compile(rf"</?(?:p|div|hr|center){_TAG_TAIL}", re.IGNORECASE)
LINE_BREAK_TAG_RE = re.compile(rf"</?(?:br|li){_TAG_TAIL}", re.IGNORECASE)
ANY_TAG_RE = re.compile(rf"</?{_TAG_NAME}{_TAG_TAIL}")

CARRIAGE_RETURN_RE = re.compile(r"\r\n|\r")
HORIZONTAL_WHITESPACE_RE = re.compile(r"[^\S\n]+")
TRAILING_SPACE_RE = re.compile(r" +\n")
BLANK_LINE_RUN_RE = re.compile(r"\n{3,}")

PARAGRAPH_BREAK = "\n\n"
LINE_BREAK = "\n"


# ---------- helpers ----------


def _strip_tags(text: str) -> str:
    """Replace block-level tags with break markers and drop every other tag.

    Args:
        text: Raw source text, still containing markup and undecoded entities.

    Returns:
        The text with p/div/hr/center replaced by a paragraph break, br/li
        replaced by a line break, and all remaining tags removed while their
        inner text is kept. A "<" that does not begin a well-formed tag name is
        left in place as literal text.
    """
    text = PARAGRAPH_TAG_RE.sub(PARAGRAPH_BREAK, text)
    text = LINE_BREAK_TAG_RE.sub(LINE_BREAK, text)
    return ANY_TAG_RE.sub("", text)


def _normalise_whitespace(text: str) -> str:
    """Collapse redundant whitespace without removing any word content.

    Args:
        text: Text that has already had markup stripped and entities decoded.

    Returns:
        The text with carriage returns folded into newlines, runs of horizontal
        whitespace (including tabs and non-breaking spaces) collapsed to one
        space, trailing spaces removed from each line, runs of three or more
        newlines collapsed to two, and the whole string stripped.
    """
    text = CARRIAGE_RETURN_RE.sub(LINE_BREAK, text)
    text = HORIZONTAL_WHITESPACE_RE.sub(" ", text)
    text = TRAILING_SPACE_RE.sub(LINE_BREAK, text)
    text = BLANK_LINE_RUN_RE.sub(PARAGRAPH_BREAK, text)
    return text.strip()


# ---------- public api ----------


def clean_markup(text: str) -> str:
    """Convert raw Riot API text into clean text for embedding and display.

    Args:
        text: Raw value from a Riot JSON payload, possibly containing HTML,
            Riot custom tags, HTML entities and "{{ placeholder }}" tokens.

    Returns:
        The text with tags stripped, entities decoded and whitespace
        normalised. Tags are stripped before entities are decoded, so an
        escaped "&lt;b&gt;" in the source survives as literal "<b>" text
        instead of being mistaken for markup. "{{ placeholder }}" tokens are
        left untouched and no content is ever truncated or summarised.
    """
    return _normalise_whitespace(html.unescape(_strip_tags(text)))


def clean_optional_markup(text: str | None) -> str | None:
    """Apply clean_markup to a value that may be absent.

    Args:
        text: Raw value from a Riot JSON payload, or None for a nullable field.

    Returns:
        None if text is None, otherwise the result of clean_markup(text).
    """
    if text is None:
        return None
    return clean_markup(text)
