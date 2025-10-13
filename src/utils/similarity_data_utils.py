import logging
import pandas as pd
from typing import Optional, Callable
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors

logger = logging.getLogger(__name__)


def calculate_tanimoto_similarity(
        mol_search: Chem.rdchem.Mol,
        smiles_from_df: str
) -> Optional[float]:
    """
    Calculate Tanimoto similarity between query molecule and target SMILES.
    Uses MACCS keys fingerprints for similarity calculation.
    Args:
        mol_search: RDKit molecule object (query)/ mol is used not to convert many times same smiles to mol
        smiles_from_df: SMILES string (target)/ smiles used not to generate column with data-heavy mol info
    Returns:
        Tanimoto similarity coefficient (0-1) or None if target is invalid
    Raises:
        ValueError: If search molecule is invalid
    """
    if not mol_search:
        logger.error("Invalid search molecule provided")
        raise ValueError("Invalid search SMILES")

    try:
        molecule_from_df = Chem.MolFromSmiles(smiles_from_df)
        if not molecule_from_df:
            return None
    except TypeError:
        return None

    fingerprints_search_mol = rdMolDescriptors.GetMACCSKeysFingerprint(mol_search)
    fingerprints_for_df_mol = rdMolDescriptors.GetMACCSKeysFingerprint(molecule_from_df)

    return DataStructs.TanimotoSimilarity(fingerprints_search_mol, fingerprints_for_df_mol)


def get_tanimoto_similarity_for_query(
        smiles: str
) -> Callable[[str], Optional[float]]:
    """An auxiliary function to use similarity_tanimoto_search with dataframe map.
    Args:
        smiles: Query SMILES string
    Returns:
        Callable that calculates similarity to the query molecule or returns None if smiles str in df is invalid.
    Raises:
        ValueError: If query SMILES is invalid"""
    mol_search = Chem.MolFromSmiles(smiles)
    if not mol_search:
        raise ValueError("Invalid SMILES")

    return lambda smi: calculate_tanimoto_similarity(mol_search, smi)


def generate_similarity_column(
        df: pd.DataFrame,
        query_smiles: str,
        smiles_col: str = "canonical_smiles"
) -> pd.DataFrame:
    """
    Add tanimoto_similarity column to DataFrame based on query SMILES.
    Args:
        df: DataFrame with column containing activities of compounds and their SMILES string
        query_smiles: Query molecule SMILES string
        smiles_col: SMILES string column
    Returns:
        DataFrame copy with added 'tanimoto_similarity' column
    """
    logger.info(f"Generating similarity column for {query_smiles} smiles")
    scorer = get_tanimoto_similarity_for_query(query_smiles)
    df = df.copy()
    df["tanimoto_similarity"] = df[smiles_col].map(scorer)
    tanimoto_sim_total = df["tanimoto_similarity"].notna().sum()
    smiles_total = df[smiles_col].notna().sum()
    logger.info(f"Generated {tanimoto_sim_total} tanimoto similarity values out of {smiles_total} smiles")
    if smiles_total > tanimoto_sim_total:
        logger.warning(f"Not converted: {smiles_total - tanimoto_sim_total} smiles")

    return df


def similarity_filter(
    df: pd.DataFrame,
    min_similarity: float = 0.8,
    max_molecules: Optional[int] = 10
) -> pd.DataFrame:
    """
    Filter by Tanimoto similarity, rank molecules by their *best* similarity,
    keep up to `max_molecules` unique molecules, and return *all* rows for them.
    Uses a MultiIndex for clean selection.

    Args:
        df: DataFrame with columns ['tanimoto_similarity','molecule_chembl_id','pchembl_value', ...]
        min_similarity: Threshold in [0,1]
        max_molecules: Max unique molecules to keep (None = all)

    Returns:
        DataFrame of all activity rows for the top molecules, sorted by similarity desc.
    """
    if df.empty:
        raise ValueError("Cannot filter empty DataFrame")

    required = ["tanimoto_similarity", "molecule_chembl_id", "pchembl_value"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame must contain columns: {missing}")
    if not 0.0 <= float(min_similarity) <= 1.0:
        raise ValueError(f"min_similarity must be between 0 and 1, got {min_similarity}")
    if max_molecules is not None and max_molecules < 1:
        raise ValueError(f"max_molecules must be positive or None, got {max_molecules}")

    # Ensure numeric similarity (avoid lexicographic sorts)
    df = df.copy()
    df["tanimoto_similarity"] = pd.to_numeric(df["tanimoto_similarity"], errors="coerce")

    # Threshold
    df_filt = df[df["tanimoto_similarity"] >= min_similarity]
    logger.info(
        "Similarity threshold %.3f: %d/%d activity rows passed",
        min_similarity, len(df_filt), len(df)
    )
    if df_filt.empty:
        logger.warning("No rows passed the similarity threshold %.3f", min_similarity)
        return df_filt.reset_index(drop=True)

    # Require at least one pChEMBL per molecule
    has_pchembl = df_filt.groupby("molecule_chembl_id")["pchembl_value"].apply(lambda s: s.notna().any())
    valid_mols = has_pchembl[has_pchembl].index.tolist()
    if not valid_mols:
        logger.warning("No molecules with valid pChEMBL values found after thresholding")
        return pd.DataFrame(columns=df.columns).reset_index(drop=True)

    # Keep only valid molecules
    df_filt = df_filt[df_filt["molecule_chembl_id"].isin(valid_mols)]

    # ---- MultiIndex flow ----
    # Index: (molecule_chembl_id, tanimoto_similarity). Sort so first per molecule is the max similarity.
    df_mi = (
        df_filt.set_index(["molecule_chembl_id", "tanimoto_similarity"])
               .sort_index(level=[0, 1], ascending=[True, False])
    )

    # Best row per molecule = first row within each first-level key (thanks to sort desc on similarity)
    best_per_mol = (
        df_mi.groupby(level=0, sort=False)
             .head(1)                       # one row per molecule (the max similarity)
             .reset_index()
             .sort_values("tanimoto_similarity", ascending=False)
    )

    # Choose top molecules
    if max_molecules is not None:
        top_mols = best_per_mol["molecule_chembl_id"].head(max_molecules).tolist()
    else:
        top_mols = best_per_mol["molecule_chembl_id"].tolist()

    # Select *all* rows for the chosen molecules via first-level .loc
    df_top = df_mi.loc[top_mols]  # MultiIndex selection by first level
    if not isinstance(df_top, pd.DataFrame):  # safety when a single molecule returns a Series-like view
        df_top = df_top.to_frame().T

    # Final ordering: similarity desc, then molecule id for stability
    out = (
        df_top.reset_index()
              .sort_values(["tanimoto_similarity", "molecule_chembl_id"], ascending=[False, True])
              .reset_index(drop=True)
    )

    logger.info(
        "Returning %d rows from %d unique molecules (max_molecules=%s).",
        len(out), out["molecule_chembl_id"].nunique(), str(max_molecules)
    )
    return out

#
#
# def similarity_filter(
#         df: pd.DataFrame,
#         min_similarity: float = 0.8,
#         max_molecules: Optional[int] = 10
# ) -> pd.DataFrame:
#     """
#     Filter and rank molecules by Tanimoto similarity, keeping all activity records.
#     Returns all rows for the top unique molecules (e.g., 10 molecules might return 50 rows).
#
#     Args:
#         df: DataFrame with 'tanimoto_similarity', 'molecule_chembl_id', and 'pchembl_value' columns
#         min_similarity: Minimum similarity threshold (0.0-1.0), default 0.8
#         max_molecules: Maximum number of unique molecules. Returns all rows for these molecules.
#
#     Returns:
#         Filtered DataFrame with all rows for top unique molecules, sorted by similarity
#
#     Raises:
#         ValueError: If dataframe is empty or missing required columns
#         ValueError: If min_similarity not in range [0, 1]
#     """
#     if df.empty:
#         raise ValueError("Cannot filter empty DataFrame")
#     required_cols = ["tanimoto_similarity", "molecule_chembl_id", "pchembl_value"]
#     missing_cols = [col for col in required_cols if col not in df.columns]
#     if missing_cols:
#         raise ValueError(f"DataFrame must contain columns: {missing_cols}")
#     if not 0 <= min_similarity <= 1:
#         raise ValueError(f"min_similarity must be between 0 and 1, got {min_similarity}")
#     if max_molecules is not None and max_molecules < 1:
#         raise ValueError(f"max_molecules must be positive or None, got {max_molecules}")
#     df_filtered = df[df["tanimoto_similarity"] >= min_similarity].copy()
#
#     logger.info(
#         f"Similarity threshold {min_similarity}: "
#         f"{len(df_filtered)}/{len(df)} activity records passed"
#     )
#
#     if df_filtered.empty:
#         logger.warning(f"No molecules passed similarity threshold {min_similarity}")
#         return df_filtered.reset_index(drop=True)
#     molecules_with_pchembl = (
#         df_filtered.groupby("molecule_chembl_id")["pchembl_value"]
#         .apply(lambda x: x.notna().any())
#     )
#     valid_molecules = molecules_with_pchembl[molecules_with_pchembl].index.tolist()
#
#     removed_molecules = len(df_filtered["molecule_chembl_id"].unique()) - len(valid_molecules)
#     if removed_molecules > 0:
#         logger.info(f"Removed {removed_molecules} molecules with no pChEMBL values")
#
#     if not valid_molecules:
#         logger.warning("No molecules with valid pChEMBL values found")
#         return pd.DataFrame(columns=df.columns).reset_index(drop=True)
#
#     if max_molecules is not None:
#         molecule_rankings = (
#             df_filtered[df_filtered["molecule_chembl_id"].isin(valid_molecules)]
#             .drop_duplicates(subset=["molecule_chembl_id"])
#             .sort_values("tanimoto_similarity", ascending=False)
#             .head(max_molecules)
#         )
#         top_molecules = molecule_rankings["molecule_chembl_id"].tolist()
#     else:
#         top_molecules = valid_molecules
#     df_result = (
#         df_filtered[df_filtered["molecule_chembl_id"].isin(top_molecules)]
#         .sort_values(["tanimoto_similarity", "molecule_chembl_id"], ascending=[False, True])
#         .reset_index(drop=True)
#     )
#
#     logger.info(
#         f"Returning {len(df_result)} activity records from "
#         f"{len(top_molecules)} unique molecules"
#     )
#
#     return df_result
