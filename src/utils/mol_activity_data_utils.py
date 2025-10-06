"""The workflow process so far:
8. Add a "similarity" column and fill it using search_smiles from the user query
    and dataframe from SQL similarity_column_generation
9. Filter the dataframe by the "similarity" column and give if there are >= 5
    molecules with similarity >= 0.7 return similarity based on the activity
    of those molecules, otherwise return "uncertain"
Consider points 7, 8, 9 to be done not with pandas but with SQLAlchemy if FastAPI
will be better for optimisation/next steps
------------------------------------------
1. Implement REST API logic <user gives target and smile -> check if a target
    already exists as SQL file -> start from SQL (point 7), if no -> start from scratch
    (point 1)
2. Consider the 10 closest related targets to score with automatic docking (Autodock Vina
    (or other options), and work with the docking score as follows:
    if the docking score is similar or better than for active, closely related compounds
    (or worse/similar to inactive closely related compounds)
    return active/inactive/unknown status, scores, docking files, and images
    (if possible).
3. Make proposals for lead compound optimisation.

IMPORTANT todos list:
TODO add logger
TODO add tests
TODO add docker
TODO add ci/cd
TODO add readme
TODO add example jupiter notebook

TODO check if all return targets are relevant to the query
"""

"""
Molecular Activity Data Utilities

This module provides utilities for fetching, processing, and analyzing molecular activity data
from ChEMBL database, including target search, activity retrieval, and bioactivity classification.
"""

from typing import Optional, Callable, List
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdMolDescriptors
from functools import partial
import pandas as pd
import re
import math
import logging

# logger config and setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# constants
UNIT_FACTORS = {  # Unit conversion concentration to standart molar
    "m": 1.0,
    "mm": 1e-3,
    "um": 1e-6,
    "nm": 1e-9,
    "pm": 1e-12,
}


def find_targets(
        query: str,
        organism: str = "Homo sapiens",
        limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Search for biological targets in ChEMBL database matching the query.
    # see more here: https://github.com/chembl/chembl_webresource_client

    Args:
        query: Search term, e.g. receptor or enzyme
        organism: Organism filter (default: "Homo sapiens")
        limit: Maximum number of results to return

    Returns:
        DataFrame with columns: target_chembl_id, pref_name, organism, target_type
    """
    logger.info(f"Searching for targets with query: '{query}', organism: '{organism}', limit: {limit}")
    try:
        t = new_client.target
        hits = t.search(query)
        rows = []
        for hit in hits:
            if organism and hit.get("organism") == organism:
                rows.append({
                    "target_chembl_id": hit["target_chembl_id"],
                    "pref_name": hit.get("pref_name"),
                    "organism": hit.get("organism"),
                    "target_type": hit.get("target_type"),
                })
                if limit and len(rows) >= limit:
                    break

        logger.info(f"Found {len(rows)} targets matching criteria")
        return pd.DataFrame(rows)
    except Exception as e:
        logger.error(f"Error searching for targets: {e}")
        raise


def get_activities_for_target(
        target_chembl_id: str,
        types: tuple[str] = ("IC50", "Ki", "Kd", "EC50")
) -> list[Optional[dict[str, str]]]:
    """
    Retrieve bioactivity data for a specific target from ChEMBL.

    Args:
        target_chembl_id: ChEMBL target identifier
        types: Tuple of activity types to retrieve (default: IC50, Ki, Kd, EC50)

    Returns:
        List of activity dictionaries containing assay and measurement data
    """
    logger.info(f"Fetching activities for target: {target_chembl_id}, types: {types}")

    try:
        act = new_client.activity
        fields = [
            "activity_id", "assay_chembl_id", "assay_type", "assay_confidence_score",
            "molecule_chembl_id", "standard_type", "standard_value", "standard_units",
            "relation", "pchembl_value", "target_chembl_id"
        ]
        activities_found = act.filter(
            target_chembl_id=target_chembl_id,
            standard_type__in=list(types)
        ).only(fields)
        activities = list(activities_found)
        logger.info(f"Retrieved {len(activities)} activities for {target_chembl_id}")
        return activities

    except Exception as e:
        logger.error(f"Error fetching activities for {target_chembl_id}: {e}")


def combine_activities_for_targets(
        target_ids: list[str],
        types: tuple[str] = ("IC50", "Ki", "Kd", "EC50")
) -> list[Optional[dict[str, str]]]:
    """
    Combines bioactivity data from multiple targets.
    Args:
        target_ids: List of ChEMBL target identifiers
        types: Tuple of activity types to retrieve (default: IC50, Ki, Kd, EC50)

    Returns:
        Combined list of all activities from all targets

    Raises:
        ValueError: If no activities are found for any target
    """
    logger.info(f"Combining activities for {len(target_ids)} targets")

    all_activities = []

    for i, target_id in enumerate(target_ids):
        logger.debug(f"Processing target {i + 1}/{len(target_ids)}: {target_id}")
        activities = get_activities_for_target(target_id, types=types)
        if activities:
            all_activities.extend(activities)

    if not all_activities:
        logger.error("No activities found for any target")
        raise ValueError("No activities found")

    logger.info(f"Combined {len(all_activities)} total activities from all targets")
    return all_activities


def standard_unit_convertor_to_pchembl(units: str, value: float) -> float:
    """
    Convert activity value in various units to pChEMBL scale (-log10(Molar)).
    Args:
        units: Unit string (e.g., "nM", "uM", "mM")
        value: Numeric activity value

    Returns:
        pChEMBL value (-log10 of molar concentration)

    Raises:
        ValueError: If value is non-positive or unit is unknown
    Todo:
        * now code raises ValueError if value is negative or unit is unknown -> better return None and log?
    """
    if value <= 0:
        logger.error(f"Invalid value for pChEMBL conversion: {value}")
        raise ValueError("value must be positive")

    u = units.strip().lower()
    u = u.replace("µ", "u")
    u = re.sub(r"\s+", "", u)
    u = u.replace("mol/l", "").replace("molperl", "")

    if u in UNIT_FACTORS:
        standard_value = UNIT_FACTORS[u] * value
        pchembl = -math.log10(standard_value)
        logger.info(f"Converted {value} {units} to pChEMBL {pchembl:.2f}")
        return pchembl
    else:
        logger.error(f"Unknown unit: {units}")
        raise ValueError(f"Unknown unit {units}")


def pchembl_extractor(activity_entry: dict) -> Optional[float]:
    """
    Extract or calculate pChEMBL value from activity data.

    The workflow utilizes the following stratagies stratagies:
    use existing pchembl_value if available; not available? -> calculate from standard_value/value or return None

    Args:
        activity_entry: ChEMBL dictionary containing activity data

    Returns:
        pChEMBL value or None if cannot be determined
    """
    if activity_entry.get("pchembl_value"):
        try:
            pchembl_value = float(activity_entry["pchembl_value"])
        except ValueError as e:
            logger.debug(f"pchembl_value is not possible to convert to float:{e}")
        else:
            return pchembl_value

    if (activity_entry.get("standard_value") and str(activity_entry["standard_value"]).
            replace('.', '', 1).isdigit()):
        units = activity_entry.get("standard_units")
        if units:
            try:
                pchembl_value_su = standard_unit_convertor_to_pchembl(
                    units=units,
                    value=float(activity_entry["standard_value"])
                )
            except (ValueError, KeyError) as e:
                logger.debug(f"Could not convert standard_value: {e}")
            else:
                return pchembl_value_su

    if activity_entry.get("value") and str(activity_entry["value"]).replace('.', '', 1).isdigit():
        units = activity_entry.get("units")
        if units:
            try:
                pchembl_value_u = standard_unit_convertor_to_pchembl(
                    units=units,
                    value=float(activity_entry["value"])
                )
            except (ValueError, KeyError) as e:
                logger.debug(f"Could not convert value: {e}")
            else:
                return pchembl_value_u

    logger.debug(f"Could not extract pChEMBL for activity {activity_entry.get('activity_id')}")
    return None


def activities_to_dataframe(activities: list[dict[str, str]]) -> pd.DataFrame:
    """
    Convert list of activity dictionaries to DataFrame, cut off non-essential columns and
     generate pChEMBL values.
    Args:
        activities: list of activity dictionaries from ChEMBL
    Returns:
        DataFrame with activities (without duplicated) and calculated pChEMBL values
    Raises:
        ValueError: If activities list is empty
    """
    logger.info(f"Converting {len(activities)} activities to DataFrame")

    if not activities:
        logger.error("Cannot create DataFrame from empty activities list")
        raise ValueError("No activities found")

    try:
        activities_df = pd.DataFrame(activities, index=None)
        final_cols = [
            "molecule_chembl_id", "activity_id", "assay_chembl_id",
            "assay_type", "standard_type", "relation"
        ]
        activities_df = activities_df[final_cols]
        activities_df["pchembl_value"] = [pchembl_extractor(act) for act in activities]

        before_dedup = len(activities_df)
        activities_df = activities_df.drop_duplicates()
        after_dedup = len(activities_df)
        logger.info(f"Removed {before_dedup - after_dedup} duplicate activities")

        valid_pchembl = activities_df["pchembl_value"].notna().sum()
        logger.info(f"Successfully calculated pChEMBL for {valid_pchembl}/{len(activities_df)} activities")

        return activities_df

    except Exception as e:
        logger.error(f"Error converting activities to DataFrame: {e}")
        raise


def attach_smiles(df: pd.DataFrame, batch_size=100) -> pd.DataFrame:
    """
    Fetch and attach canonical SMILES strings to molecules in the DataFrame.
    Retrieves SMILES from ChEMBL in batches of batch_size (default 100) for efficiency.
    Args:
        df: DataFrame with 'molecule_chembl_id' column
    Returns:
        DataFrame with added 'canonical_smiles' column
    """
    logger.info("Fetching SMILES strings for molecules")

    try:
        # mol = new_client.molecule
        # ids = df["molecule_chembl_id"].dropna().unique().tolist()
        # logger.info(f"Fetching SMILES for {len(ids)} unique molecules")
        #
        # chunks = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
        # id_to_smiles = {}
        #
        # for idx, chunk in enumerate(chunks):
        #     mres = mol.filter(molecule_chembl_id__in=chunk).only(
        #         ["molecule_chembl_id", "molecule_structures"]
        #     )
        #     for m in mres:
        #         smi = None
        #         if m.get("molecule_structures"):
        #             smi = m["molecule_structures"].get("canonical_smiles")
        #         id_to_smiles[m["molecule_chembl_id"]] = smi

        df = df.copy()
        df["canonical_smiles"] = df["molecule_chembl_id"].map(id_to_smiles)

        valid_smiles = df["canonical_smiles"].notna().sum()
        logger.info(f"Successfully retrieved SMILES for {valid_smiles}/{len(df)} activities")

        return df

    except Exception as e:
        logger.error(f"Error fetching SMILES: {e}")
        raise



def morgan_finger_prints_from_smiles(smiles: str):  # TODO: add return value
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) if mol else None


def add_mols_and_fps(
        df: pd.DataFrame,
        smiles_col: str = "canonical_smiles",
        mol_col: str = "mol",
        fp_col: str = "ecfp4"
) -> pd.DataFrame:
    mols = []
    fps = []
    for smi in df[smiles_col].astype(str):
        m = Chem.MolFromSmiles(smi)
        mols.append(m)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) if m else None)
    out = df.copy()
    out[mol_col] = mols
    out[fp_col] = fps
    out = out[out[mol_col].notna()].reset_index(drop=True)
    return out


def similarity_tanimoto_search(
        mol_search: Chem.rdchem.Mol,
        smiles_df: str) -> Optional[float]:
    if not mol_search:
        raise ValueError("Invalid search SMILES")
    mol_df = Chem.MolFromSmiles(smiles_df)
    if not mol_df:
        return None
    fp_search = rdMolDescriptors.GetMACCSKeysFingerprint(mol_search)
    fp_df = rdMolDescriptors.GetMACCSKeysFingerprint(mol_df)
    return DataStructs.TanimotoSimilarity(fp_search, fp_df)


def similarity_search_for_query(smiles: str) -> Callable[[str], Optional[float]]:
    mol_search = Chem.MolFromSmiles(smiles)
    if not mol_search:
        raise ValueError("Invalid SMILES")
    return partial(similarity_tanimoto_search, mol_search)


def similarity_column_generation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["similarity"] = df["smiles"].apply(similarity_search_for_query, axis=1)
    return df


def assay_info_extractor(activities_df: pd.DataFrame) -> list[dict[str, str]]:
    assays = activities_df["assay_chembl_id"].dropna().unique().tolist()
    all_assays_info = list(new_client.assay.filter(assay_chembl_id__in=assays)
    .only(
        ["assay_chembl_id", "assay_cell_type", "bao_label"]))
    return all_assays_info


def assay_type_auxiliary(assay_info: dict[str, str]) -> Optional[str]:
    fmt = (assay_info.get("bao_label")).lower()
    BAO_TO_CONTEXT = {
        "cell-based format": "cellular",
        "organism-based format": "organism",
        "single protein format": "biochemical",
        "protein format": "biochemical",
        "cell-free format": "biochemical",
        "cell membrane format": "cellular",
        "subcellular format": "cellular",
        "assay format": None,
    }
    content = BAO_TO_CONTEXT.get(fmt)
    if not content and assay_info.get("assay_cell_type"):
        return "cellular"
    return BAO_TO_CONTEXT.get(fmt)


def certain_activity_mapper(all_assays_info: list[dict[str, str]]) -> dict[str, Optional[str]]:
    assay_type = dict()
    for assay_info in all_assays_info:
        if assay_info.get("assay_chembl_id") and assay_info.get("assay_chembl_id") not in assay_type.keys():
            assay_type[assay_info["assay_chembl_id"]] = assay_type_auxiliary(assay_info)
    return assay_type


def assay_exact_type_generator(activities: pd.DataFrame) -> pd.DataFrame:
    df = activities.copy()
    all_assays_info = assay_info_extractor(df)
    ctx_map = certain_activity_mapper(all_assays_info)
    df["context"] = df["assay_chembl_id"].map(ctx_map)
    return df


def assay_approx_type_generator_for_row(activities: pd.DataFrame) -> pd.DataFrame:
    df = activities.copy()
    stype = df["standard_type"].astype(str).str.upper()
    atype = df["assay_type"].astype(str).str.upper()
    unknown = df["context"].isna()
    df.loc[unknown & stype.isin(["KI", "KD"]), "context"] = "biochemical"
    df.loc[unknown & (atype == "F") & (stype == "EC50"), "context"] = "cellular"
    df.loc[unknown & (atype == "B") & (stype == "IC50"), "context"] = "biochemical"
    return df


def activity_status(activities: pd.DataFrame) -> pd.DataFrame:
    THRESHOLD_ACTIVE = {
        "biochemical": 6.0,  # <= 1 µM
        "cellular": 5.0,  # <= 10 µM
        "organism": 4.5,  # <= ~33 µM
        "unknown": 6.0,  # same as the worst (<= 1 µM)
    }
    ALLOWED_REL = {
        "=", "~", None, "<", "<="

    }
    out = activities.copy()
    out["pchembl_value"] = pd.to_numeric(out["pchembl_value"], errors="coerce")
    ctx = out.get("context", pd.Series(index=out.index, dtype=object)).fillna("unknown").astype(str)
    cutoff = ctx.map(THRESHOLD_ACTIVE).fillna(THRESHOLD_ACTIVE["unknown"])
    rel_ok = out.get("relation", pd.Series(index=out.index, dtype=object)).isin(ALLOWED_REL)
    out["is_active"] = (out["pchembl_value"] >= cutoff) & rel_ok
    return out


def main_generator(query: str) -> pd.DataFrame:
    targets = find_targets(query)
    print(targets)
    combined_activities = combine_activities_for_targets(targets["target_chembl_id"].tolist())
    print(combined_activities[0:3])
    activities = activities_to_dataframe(combined_activities)
    print(activities.head())
    activities_with_smiles = attach_smiles(activities)
    print(activities_with_smiles.head())
    activities_with_exact_assays = assay_exact_type_generator(activities_with_smiles)
    print(activities_with_exact_assays.head())
    activities_with_all_assays = assay_approx_type_generator_for_row(activities_with_exact_assays)
    print(activities_with_all_assays.head())
    activities_all = activity_status(activities_with_all_assays)
    print(activities_all.head())
    return activities_all[
        ["molecule_chembl_id", "activity_id", "assay_chembl_id", "pchembl_value", "context", "canonical_smiles",
         "is_active"]]
