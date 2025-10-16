"""
Molecular Similarity Utilities

This module provides utilities for calculating Tanimoto molecular similarity
using MACCS keys fingerprints and filtering molecular activity datasets by similarity.
"""

import logging
import pandas as pd
from typing import Optional, Callable
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors

logger = logging.getLogger(__name__)


def calculate_tanimoto_similarity(
        mol_search: Chem.rdchem.Mol,
        smiles_from_df: Optional[str]
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
        if smiles_from_df is None:
            return None

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
    df["tanimoto_similarity"] = pd.to_numeric(df["tanimoto_similarity"], errors="coerce")
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
    Filter molecules by Tanimoto similarity and return all activity rows for the most similar molecules.
    Args:
        df: DataFrame molecules and similarity values
        min_similarity: Threshold in [0,1], default 0.8
        max_molecules: Maximum unique molecules to keep (None = all), default 10
    Returns:
        DataFrame with all activity rows for the best fit molecules, sorted by similarity descending
    Raises:
        ValueError: If DataFrame is empty
        ValueError: If required columns are missing from DataFrame
        ValueError: If min_similarity is not in range [0, 1]
        ValueError: If max_molecules is not positive or None
    """
    if df.empty:
        raise ValueError("Cannot filter empty DataFrame")

    required_columns = ["tanimoto_similarity", "molecule_chembl_id", "pchembl_value"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame must contain columns: {missing_columns}")

    if not 0.0 <= float(min_similarity) <= 1.0:
        raise ValueError(f"min_similarity must be between 0 and 1, got {min_similarity}")

    if max_molecules is not None and max_molecules < 1:
        raise ValueError(f"max_molecules must be positive or None, got {max_molecules}")
    filtered_by_similarity = df[df["tanimoto_similarity"] >= min_similarity].copy()
    logger.info(
        f"Similarity threshold {min_similarity}: "
        f"{len(filtered_by_similarity)}/{len(df)} activity rows passed"
    )
    if filtered_by_similarity.empty:
        logger.warning(f"No rows passed similarity threshold {min_similarity}")

        return filtered_by_similarity.reset_index(drop=True)

    molecules_with_activity = (
        filtered_by_similarity
        .groupby("molecule_chembl_id")["pchembl_value"]
        .apply(lambda values: values.notna().any())
    )
    valid_molecule_ids = molecules_with_activity[molecules_with_activity].index.tolist()
    if not valid_molecule_ids:
        logger.warning("No molecules with valid pChEMBL values found")

        return pd.DataFrame(columns=df.columns).reset_index(drop=True)

    molecules_with_valid_activity = filtered_by_similarity[
        filtered_by_similarity["molecule_chembl_id"].isin(valid_molecule_ids)
    ]
    best_similarity_per_molecule = (
        molecules_with_valid_activity
        .groupby("molecule_chembl_id")["tanimoto_similarity"]
        .max()
        .sort_values(ascending=False)
    )
    if max_molecules is not None:
        top_molecule_ids = best_similarity_per_molecule.head(max_molecules).index.tolist()
    else:
        top_molecule_ids = best_similarity_per_molecule.index.tolist()
    result = molecules_with_valid_activity[
        molecules_with_valid_activity["molecule_chembl_id"].isin(top_molecule_ids)
    ]
    result = result.sort_values(
        ["tanimoto_similarity", "molecule_chembl_id"],
        ascending=[False, True]
    ).reset_index(drop=True)
    logger.info(
        f"Returning {len(result)} rows from {result['molecule_chembl_id'].nunique()} unique molecules"
    )

    return result
