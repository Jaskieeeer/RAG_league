import argparse

import truststore

from lolrag.config import Settings, get_settings
from lolrag.indexing import build_index
from lolrag.ingestion import ingest
from lolrag.pipeline import answer_question

truststore.inject_into_ssl()


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with ingest and ask subcommands.

    Returns:
        ArgumentParser whose parsed namespace carries a "command" attribute set
        to "ingest" or "ask", and for "ask" a "question" attribute with the
        positional question text.
    """
    parser = argparse.ArgumentParser(
        prog="lolrag",
        description="League of Legends RAG pipeline over Data Dragon champion data.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "ingest",
        help="Fetch all champions from Data Dragon and build the vector index.",
    )

    ask_parser = subparsers.add_parser(
        "ask",
        help="Answer a question using the indexed champion corpus.",
    )
    ask_parser.add_argument("question", help="Question to answer.")

    return parser


def _run_ingest(settings: Settings) -> None:
    """Fetch the full champion corpus and upsert it into the vector store.

    Args:
        settings: Application settings for ingestion and indexing.
    """
    documents = ingest(settings)
    build_index(documents, settings)
    print(f"Indexed {len(documents)} documents into {settings.chroma_persist_dir}")


def _run_ask(question: str, settings: Settings) -> None:
    """Answer a question and print the answer with its sources.

    Args:
        question: Question to answer.
        settings: Application settings for retrieval and generation.
    """
    response = answer_question(question, settings)
    print(response.answer)
    print()
    print("Sources:")
    for source in response.sources:
        label = source.name if source.name is not None else source.source
        print(f"- {label} ({source.source})")


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and dispatch to the selected command.

    Args:
        argv: Argument list to parse, or None to use sys.argv.
    """
    args = _build_parser().parse_args(argv)
    settings = get_settings()
    if args.command == "ingest":
        _run_ingest(settings)
    elif args.command == "ask":
        _run_ask(args.question, settings)


if __name__ == "__main__":
    main()
