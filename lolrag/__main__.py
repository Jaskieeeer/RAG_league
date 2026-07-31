import argparse
import asyncio
import logging

import truststore

from lolrag.config import Settings, get_settings
from lolrag.db.session import get_session
from lolrag.fetch.client import FetchClient
from lolrag.ingest.documents import DocumentLoadStats, load_documents
from lolrag.ingest.run import IngestReport, run_ingest

truststore.inject_into_ssl()


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with the ingest and index subcommands.

    Returns:
        ArgumentParser whose parsed namespace carries a "command" attribute set
        to "ingest" or "index", and a "refresh" flag on "ingest".
    """
    parser = argparse.ArgumentParser(
        prog="lolrag",
        description="League of Legends RAG pipeline over Riot first-party JSON APIs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Warm the corpus cache and load every entity and value row into Postgres.",
    )
    ingest_parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refetch every cache entry instead of reusing the ones on disk.",
    )

    subparsers.add_parser(
        "index",
        help="Rebuild documents and chunks from the entity rows already in Postgres.",
    )

    return parser


async def _ingest(settings: Settings, *, refresh: bool) -> IngestReport:
    """Run one ingest inside a session that is committed on success.

    Args:
        settings: Application settings for fetching and loading.
        refresh: Whether to refetch every cache entry.

    Returns:
        The IngestReport of the committed run.
    """
    with get_session(settings) as session:
        async with FetchClient(settings) as client:
            report = await run_ingest(session, client, settings, refresh=refresh)
        session.commit()
    return report


def _index(settings: Settings) -> DocumentLoadStats:
    """Run the document stage alone inside a session that is committed on success.

    Args:
        settings: Application settings providing database_url and
            embedding_model_name.

    Returns:
        The DocumentLoadStats of the committed run.
    """
    with get_session(settings) as session:
        stats = load_documents(session, settings)
        session.commit()
    return stats


def _print_document_stats(stats: DocumentLoadStats, width: int) -> None:
    """Print the document stage counts as aligned rows.

    Args:
        stats: The DocumentLoadStats to render.
        width: Label column width the rows are padded to.
    """
    print(f"{'documents built':<{width}}  {stats.documents_built:>6}")
    print(f"{'documents changed':<{width}}  {stats.documents_changed:>6}")
    print(f"{'chunks written':<{width}}  {stats.chunks_written:>6}")
    print(f"{'chunks skipped':<{width}}  {stats.chunks_skipped:>6}")


def _print_report(report: IngestReport) -> None:
    """Print one ingest run as an aligned per-table row count table.

    Args:
        report: The IngestReport to render.
    """
    counts = {
        "factions": report.entities.factions,
        "roles": report.entities.roles,
        "champions": report.entities.champions,
        "abilities": report.entities.abilities,
        "stories": report.entities.stories,
        "items": report.entities.items,
        "rune_paths": report.entities.rune_paths,
        "runes": report.entities.runes,
        "summoner_spells": report.entities.summoner_spells,
        "ability_values": report.values.ability_values,
        "item_values": report.values.item_values,
        "champion_role": report.entities.associations.champion_roles,
        "champion_related": report.entities.associations.champion_related,
        "story_champion": report.entities.associations.story_champions,
        "item_tag": report.entities.associations.item_tags,
        "item_map": report.entities.associations.item_maps,
        "item_components": report.entities.associations.item_components,
    }
    width = max(len(name) for name in counts)
    for name, count in counts.items():
        print(f"{name:<{width}}  {count:>6}")
    print()
    _print_document_stats(report.documents, width)
    print()
    print(f"{'cache hits':<{width}}  {report.cache.cache_hits:>6}")
    print(f"{'cache misses':<{width}}  {report.cache.cache_misses:>6}")
    print(f"{'dropped edges':<{width}}  {report.entities.associations.dropped_edges:>6}")


def _run_ingest(settings: Settings, *, refresh: bool) -> None:
    """Ingest the full corpus into Postgres and print the resulting report.

    Args:
        settings: Application settings for fetching and loading.
        refresh: Whether to refetch every cache entry.
    """
    _print_report(asyncio.run(_ingest(settings, refresh=refresh)))


def _run_index(settings: Settings) -> None:
    """Rebuild documents and chunks from the stored entities and print the counts.

    Args:
        settings: Application settings providing database_url and
            embedding_model_name.
    """
    stats = _index(settings)
    _print_document_stats(stats, len("documents changed"))


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and dispatch to the selected command.

    Args:
        argv: Argument list to parse, or None to use sys.argv.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    if args.command == "ingest":
        _run_ingest(settings, refresh=args.refresh)
    elif args.command == "index":
        _run_index(settings)


if __name__ == "__main__":
    main()
