"""
LeadCraft Entry Point

This module provides the main programmatic and CLI entry point for
bioactivity retrieval and molecular similarity search.
"""

import argparse
import logging
from typing import Optional

import pandas as pd

from src.mol_activity.utils.config import DB_PATH
from src.mol_activity.utils.files_and_SQL_utils import MolecularActivityDatabase
from src.mol_activity.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def basic_query(
    target: str,
    smiles: str,
    min_similarity: float = 0.8,
    max_molecules: int = 10,
    organism: Optional[str] = "Homo sapiens",
    force_refresh: bool = False,
    db_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Main pipeline entry point for molecular similarity search against bioactivity data.

    Checks local cache first and fetches data from ChEMBL if needed.

    Args:
        target: Target name to search, e.g. "CDK2" or "EGFR"
        smiles: Query molecule SMILES string
        min_similarity: Minimum Tanimoto similarity threshold in range [0, 1]
        max_molecules: Maximum number of similar molecules to return
        organism: Organism filter (default: "Homo sapiens", <None> for all organisms)
        force_refresh: If True, re-fetch data from ChEMBL even if cached
        db_path: Path to SQLite database file (default: project database path)

    Returns:
        DataFrame with columns: activity_id, molecule_chembl_id, target_chembl_id,
        assay_chembl_id, pchembl_value, context, canonical_smiles, is_active,
        tanimoto_similarity — sorted by similarity descending.
    """
    logger.info(f"Starting query pipeline for target: '{target}'")

    path = db_path if db_path is not None else str(DB_PATH)
    db = MolecularActivityDatabase(path)

    results = db.process_query(
        target_name=target,
        query_smiles=smiles,
        min_similarity=min_similarity,
        max_molecules=max_molecules,
        organism=organism,
        force_refresh=force_refresh,
    )

    logger.info(f"Query complete: {len(results)} activity records returned")

    return results


def _build_parser() -> argparse.ArgumentParser:
    """
    Build CLI argument parser for LeadCraft entry point.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="leadcraft",
        description="LeadCraft: bioactivity retrieval and similarity search via ChEMBL.",
    )
    parser.add_argument("--target", required=True, help="Target name (e.g. CDK2, EGFR)")
    parser.add_argument("--smiles", required=True, help="Query molecule SMILES string")
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.8,
        help="Minimum Tanimoto similarity threshold [0-1] (default: 0.8)",
    )
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=10,
        help="Maximum number of similar molecules to return (default: 10)",
    )
    parser.add_argument(
        "--organism",
        default="Homo sapiens",
        help='Organism filter for ChEMBL (default: "Homo sapiens", use "all" for no filter)',
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-fetch data from ChEMBL even if cached",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to SQLite database file (default: project database path)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    return parser


def main() -> None:
    """
    Run LeadCraft CLI workflow.

    Parses CLI arguments, initializes logging, runs the query pipeline,
    and prints formatted results to stdout.
    """
    parser = _build_parser()
    args = parser.parse_args()

    setup_logging(level=args.log_level)

    organism = None if args.organism.lower() == "all" else args.organism

    results = basic_query(
        target=args.target,
        smiles=args.smiles,
        min_similarity=args.min_similarity,
        max_molecules=args.max_molecules,
        organism=organism,
        force_refresh=args.force_refresh,
        db_path=args.db_path,
    )

    if results.empty:
        logger.warning("Query returned no matching activity records")
        print("\nNo activity records found.")
        return

    print(
        f"\nFound {len(results)} activity records from "
        f"{results['molecule_chembl_id'].nunique()} unique molecules"
    )
    print(
        f"Similarity range: "
        f"{results['tanimoto_similarity'].min():.3f} - {results['tanimoto_similarity'].max():.3f}"
    )
    if results["pchembl_value"].notna().any():
        print(
            f"pChEMBL range:   "
            f"{results['pchembl_value'].min():.2f} - {results['pchembl_value'].max():.2f}"
        )
    print()
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
