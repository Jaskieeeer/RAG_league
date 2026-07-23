from lolrag.indexing import compute_document_id
from tests.test_ingestion import build_documents


def test_same_inputs_produce_same_id():
    first = compute_document_id("ddragon:champion:Aatrox:16.14.1", "some lore text")
    second = compute_document_id("ddragon:champion:Aatrox:16.14.1", "some lore text")

    assert first == second


def test_different_source_produces_different_id():
    same_content = "some lore text"

    aatrox_id = compute_document_id("ddragon:champion:Aatrox:16.14.1", same_content)
    ahri_id = compute_document_id("ddragon:champion:Ahri:16.14.1", same_content)

    assert aatrox_id != ahri_id


def test_different_content_produces_different_id():
    same_source = "ddragon:champion:Aatrox:16.14.1"

    before_patch_id = compute_document_id(same_source, "old lore text")
    after_patch_id = compute_document_id(same_source, "new lore text")

    assert before_patch_id != after_patch_id


def test_id_is_32_lowercase_hex_characters():
    document_id = compute_document_id("ddragon:champion:Aatrox:16.14.1", "some lore text")

    assert len(document_id) == 32
    assert all(char in "0123456789abcdef" for char in document_id)


def test_id_is_stable_for_empty_and_unicode_content():
    empty_id = compute_document_id("ddragon:champion:Aatrox:16.14.1", "")
    unicode_id = compute_document_id("ddragon:champion:Ahri:16.14.1", "Ahri's essence theft ☃")

    assert len(empty_id) == 32
    assert len(unicode_id) == 32
    assert empty_id != unicode_id


def test_ids_for_fixture_documents_are_pairwise_distinct():
    documents = build_documents()

    ids = [compute_document_id(doc.metadata["source"], doc.page_content) for doc in documents]

    assert len(ids) == len(set(ids))
