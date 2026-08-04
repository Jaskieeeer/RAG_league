import pytest

from lolrag.__main__ import _build_parser


def test_parser_parses_ingest_command():
    args = _build_parser().parse_args(["ingest"])

    assert args.command == "ingest"
    assert args.refresh is False


def test_parser_parses_ingest_refresh_flag():
    args = _build_parser().parse_args(["ingest", "--refresh"])

    assert args.command == "ingest"
    assert args.refresh is True


def test_parser_parses_index_command():
    args = _build_parser().parse_args(["index"])

    assert args.command == "index"


def test_parser_rejects_refresh_flag_on_index_command():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["index", "--refresh"])


def test_parser_parses_ask_command_with_its_question():
    args = _build_parser().parse_args(["ask", "Who is Aatrox?"])

    assert args.command == "ask"
    assert args.question == "Who is Aatrox?"


def test_parser_rejects_ask_without_a_question():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["ask"])


def test_parser_rejects_missing_command():
    with pytest.raises(SystemExit):
        _build_parser().parse_args([])


def test_parser_rejects_unknown_command():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["frobnicate"])
