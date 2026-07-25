from pathlib import Path

from lolrag.fetch.cache import cache_path, read_cached, safe_segment, write_cached


def test_safe_segment_replaces_illegal_characters() -> None:
    """Every filesystem-illegal character and control character becomes an underscore."""
    assert safe_segment('a<b>c:d"e/f\\g|h?i*j') == "a_b_c_d_e_f_g_h_i_j"
    assert safe_segment("tab\there") == "tab_here"
    assert safe_segment("Aatrox.json") == "Aatrox.json"


def test_cache_path_composes_nested_path(tmp_path: Path) -> None:
    """cache_path mirrors the source URL structure one directory per segment."""
    path = cache_path(tmp_path, "ddragon", "16.14.1", "en_US", "champion.json")

    assert path == tmp_path / "ddragon" / "16.14.1" / "en_US" / "champion.json"


def test_cache_path_sanitises_each_segment(tmp_path: Path) -> None:
    """A segment containing a separator is sanitised rather than deepening the tree."""
    path = cache_path(tmp_path, "universe", "story/nested.json")

    assert path == tmp_path / "universe" / "story_nested.json"


def test_write_cached_writes_content_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    """The written file holds the exact bytes and no .tmp artifact survives."""
    path = cache_path(tmp_path, "ddragon", "api", "versions.json")

    write_cached(path, b'["16.14.1"]')

    assert path.read_bytes() == b'["16.14.1"]'
    assert list(tmp_path.rglob("*.tmp")) == []


def test_write_cached_overwrites_an_existing_entry(tmp_path: Path) -> None:
    """Writing over an existing entry replaces its content entirely."""
    path = cache_path(tmp_path, "ddragon", "api", "versions.json")
    write_cached(path, b'["16.14.1"]')

    write_cached(path, b'["16.15.1"]')

    assert path.read_bytes() == b'["16.15.1"]'


def test_read_cached_returns_none_for_missing_path(tmp_path: Path) -> None:
    """A cache miss is reported as None rather than raising."""
    assert read_cached(tmp_path / "ddragon" / "api" / "versions.json") is None


def test_read_cached_returns_written_bytes(tmp_path: Path) -> None:
    """A written entry reads back byte for byte."""
    path = cache_path(tmp_path, "cdragon", "latest", "champions", "266.json")
    write_cached(path, b'{"id": 266}')

    assert read_cached(path) == b'{"id": 266}'
