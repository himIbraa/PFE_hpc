"""CLI entry point: akn-rlm run / eval / index"""
from __future__ import annotations

import sys
import click


@click.group()
def main():
    """AKN-RLM: Recursive-Language-Model legal IR for AlgerianLegalBench v3.0."""


@main.command()
@click.option("--query", "-q", required=True, help="Legal query (Arabic or French).")
def run(query: str):
    """Run a single query through the full RLM pipeline."""
    click.echo(f"[akn-rlm run] Query: {query}")
    click.echo("Pipeline not yet implemented — Phase A only.")


@main.command()
@click.option("--output", "-o", default="results/eval.json", help="Output JSON path.")
def eval(output: str):
    """Run full benchmark evaluation."""
    click.echo(f"[akn-rlm eval] Output: {output}")
    click.echo("Eval runner not yet implemented — Phase A only.")


@main.command()
@click.option("--force", is_flag=True, help="Rebuild even if indices exist.")
def index(force: bool):
    """Build all retrieval indices from the AKN corpus."""
    from akn_rlm.corpus.akn_parser import parse_all
    from akn_rlm.corpus.article_registry import get_registry
    from akn_rlm.config import ARTICLE_REGISTRY_PATH, INDICES_DIR

    INDICES_DIR.mkdir(parents=True, exist_ok=True)

    click.echo("[akn-rlm index] Parsing AKN XML files ...")
    articles = parse_all()
    click.echo(f"  {len(articles):,} articles parsed.")

    registry = get_registry(articles=articles, force_rebuild=force)
    click.echo(f"  {registry.doc_count} canonical documents registered.")

    registry.save(ARTICLE_REGISTRY_PATH)
    click.echo(f"  Registry saved to {ARTICLE_REGISTRY_PATH}")
    click.echo("[akn-rlm index] Phase A complete.")


if __name__ == "__main__":
    main()
