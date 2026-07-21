import json
from pathlib import Path

import pytest

from lolrag.ingestion import champion_json_to_documents

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURE_VERSION = "16.14.1"


def load_summary_fixture() -> dict:
    return json.loads((FIXTURES_DIR / "champion_fixture.json").read_text(encoding="utf-8"))


def load_detail_fixture() -> dict:
    return json.loads((FIXTURES_DIR / "champion_detail_fixture.json").read_text(encoding="utf-8"))


def build_documents():
    return champion_json_to_documents(
        load_summary_fixture(), load_detail_fixture(), FIXTURE_VERSION
    )


def test_returns_one_document_per_champion():
    documents = build_documents()

    assert len(documents) == 3
    champion_ids = {doc.metadata["champion_id"] for doc in documents}
    assert champion_ids == {"Aatrox", "Ahri", "Jinx"}


def test_multi_tag_champion_has_joined_tags_string():
    documents = build_documents()

    ahri = next(doc for doc in documents if doc.metadata["champion_id"] == "Ahri")
    assert ahri.metadata["tags"] == "Mage, Assassin"


def test_every_document_stamped_with_version():
    documents = build_documents()

    assert all(doc.metadata["ddragon_version"] == FIXTURE_VERSION for doc in documents)


def test_page_content_contains_name_and_title():
    documents = build_documents()

    jinx = next(doc for doc in documents if doc.metadata["champion_id"] == "Jinx")
    assert jinx.page_content
    assert "Jinx" in jinx.page_content
    assert "the Loose Cannon" in jinx.page_content


def test_page_content_contains_full_lore():
    documents = build_documents()

    aatrox = next(doc for doc in documents if doc.metadata["champion_id"] == "Aatrox")
    detail = load_detail_fixture()["Aatrox"]
    assert detail["lore"] in aatrox.page_content


def test_page_content_contains_passive_and_all_four_spells():
    documents = build_documents()

    ahri = next(doc for doc in documents if doc.metadata["champion_id"] == "Ahri")
    detail = load_detail_fixture()["Ahri"]

    assert detail["passive"]["name"] in ahri.page_content
    assert detail["passive"]["description"] in ahri.page_content
    for spell in detail["spells"]:
        assert spell["name"] in ahri.page_content
        assert spell["description"] in ahri.page_content


def test_mismatched_version_raises_value_error():
    with pytest.raises(ValueError):
        champion_json_to_documents(load_summary_fixture(), load_detail_fixture(), "16.13.1")


def test_missing_champion_detail_raises_key_error():
    details = load_detail_fixture()
    del details["Jinx"]

    with pytest.raises(KeyError):
        champion_json_to_documents(load_summary_fixture(), details, FIXTURE_VERSION)
