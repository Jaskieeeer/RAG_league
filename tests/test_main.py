import pytest

from lolrag.__main__ import _build_parser


def test_parser_parses_ingest_command():
    args = _build_parser().parse_args(["ingest"])

    assert args.command == "ingest"


def test_parser_parses_ask_command_with_question():
    args = _build_parser().parse_args(["ask", "some question"])

    assert args.command == "ask"
    assert args.question == "some question"


def test_parser_rejects_missing_command():
    with pytest.raises(SystemExit):
        _build_parser().parse_args([])


def test_parser_rejects_unknown_command():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["frobnicate"])
