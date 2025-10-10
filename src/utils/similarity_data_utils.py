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