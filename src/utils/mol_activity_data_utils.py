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

from typing import Optional, Callable
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import DataStructs, rdMolDescriptors
from logging.config import dictConfig
import pandas as pd
import re
import math
import logging
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


def setup_logging(
        level: str = "INFO",
        mute_chembl: bool = True
) -> None:
    """
    Configure logging for the application with suppression of noisy external libraries.
    Args:
        level: Logging level for the application ("DEBUG", "INFO", "WARNING", "ERROR")
        mute_chembl: If True, suppress verbose output from ChEMBL and HTTP libraries
    """
    dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "std": {"format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s"}
        },
        "handlers": {
            "console": {"class": "logging.StreamHandler", "formatter": "std"}
        },
        "root": {"level": level.upper(), "handlers": ["console"]},
        "loggers": {
            "chembl_webresource_client": {"level": "ERROR"},
        } if mute_chembl else {}
    })


@dataclass
class ConversionStatistics:
    """Statistics/errors collected during pChEMBL fetching and conversion process"""
    chembl_targets: dict = field(default_factory=dict)
    unknown_units: dict = field(default_factory=lambda: defaultdict(int))
    invalid_values: int = 0
    pchembl_negative_values: int = 0
    no_activity: int = 0


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
        types: tuple[str] = ("IC50", "Ki", "Kd", "EC50"),
        stats: Optional[ConversionStatistics] = None
) -> list[Optional[dict[str, str]]]:
    """
    Retrieve bioactivity data for a specific target from ChEMBL.

    Args:
        target_chembl_id: ChEMBL target identifier
        types: Tuple of activity types to retrieve (default: IC50, Ki, Kd, EC50)
        stats: Optional statistics object to track fetching results

    Returns:
        List of activity dictionaries containing assay and measurement data
    """
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
        if stats:
            stats.chembl_targets[target_chembl_id] = len(activities)

        return activities

    except Exception as e:
        logger.error(f"Error fetching activities for {target_chembl_id}: {e}")
        if stats:
            stats.chembl_targets[target_chembl_id] = 0
        raise


def combine_activities_for_targets(
        target_ids: list[str],
        types: tuple[str] = ("IC50", "Ki", "Kd", "EC50"),
        stats: Optional[ConversionStatistics] = None
) -> list[Optional[dict[str, str]]]:
    """
    Combines bioactivity data from multiple targets.
    Args:
        target_ids: List of ChEMBL target identifiers
        types: Tuple of activity types to retrieve (default: IC50, Ki, Kd, EC50)
        stats: Optional statistics object to track fetching results

    Returns:
        Combined list of all activities from all targets

    Raises:
        ValueError: If no activities are found for any target
    """
    logger.info(f"Retrieving and combining activities for {len(target_ids)} targets")

    all_activities = []

    for i, target_id in enumerate(target_ids):
        activities = get_activities_for_target(target_chembl_id=target_id, types=types, stats=stats)
        if activities:
            all_activities.extend(activities)

    if stats and stats.chembl_targets:
        sorted_targets = sorted(
            stats.chembl_targets.items(),
            key=lambda x: x[1],
            reverse=True
        )
        targets_per_line = 5
        lines = [f"Retrieved activities from {len(sorted_targets)} targets:"]
        for i in range(0, len(sorted_targets), targets_per_line):
            batch = sorted_targets[i:i + targets_per_line]
            line = ", ".join([f"{tid}: {count}" for tid, count in batch])
            lines.append(f"  {line}")
        logger.info("\n".join(lines))

    logger.info(f"Found {len(all_activities)} activities combined")
    if not all_activities:
        logger.error("No activities found for any target")
        raise ValueError("No activities found")

    return all_activities


def convert_standard_units_to_pchembl(
        units: str,
        value: float,
        stats: Optional[ConversionStatistics] = None
) -> Optional[float]:
    """
    Convert activity value in various units to pChEMBL scale (-log10(Molar)).
    Args:
        units: Unit string (e.g., "nM", "uM", "mM")
        value: Numeric activity value
        stats: Optional statistics object to track fetching results

    Returns:
        pChEMBL value (-log10 of molar concentration)

    Raises:
        ValueError: If value is non-positive or unit is unknown
    """
    unit_factors = {  # Unit conversion concentration to standard molar
        "m": 1.0,
        "mm": 1e-3,
        "um": 1e-6,
        "nm": 1e-9,
        "pm": 1e-12,
    }
    if value < 0:
        if stats:
            stats.pchembl_negative_values += 1
        return None

    if value == 0:
        return None

    u = units.strip().lower()
    u = u.replace("µ", "u")
    u = re.sub(r"\s+", "", u)
    u = u.replace("mol/l", "").replace("molperl", "")

    if u in unit_factors:
        standard_value = unit_factors[u] * value
        pchembl = -math.log10(standard_value)
        return pchembl
    else:
        if stats:
            stats.unknown_units[units] += 1
        return None


def retrieve_pchembl_value(
        activity_entry: dict,
        stats: Optional[ConversionStatistics] = None
) -> Optional[float]:
    """
    Extract or calculate pChEMBL value from activity data.

    The workflow utilizes the following strategies:
    use existing pchembl_value if available; not available? -> calculate from standard_value/value or return None

    Args:
        activity_entry: ChEMBL dictionary containing activity data
        stats: Optional statistics object to track fetching results

    Returns:
        pChEMBL value or None if it cannot be determined
    """

    if activity_entry.get("pchembl_value"):
        try:
            pchembl_value = float(activity_entry["pchembl_value"])
        except ValueError as e:
            logger.debug(
                f"pchembl_value {activity_entry['pchembl_value']} is not possible to convert to float: {e}")
        else:
            return pchembl_value

    try:
        value = float(activity_entry["standard_value"])
    except (ValueError, KeyError):
        pass
    else:
        units = activity_entry.get("standard_units")
        if units:
            try:
                pchembl_value_su = convert_standard_units_to_pchembl(
                    units=units,
                    value=value,
                    stats=stats
                )
            except (ValueError, KeyError) as e:
                logger.debug(f"Could not convert value: {e}")
            else:
                return pchembl_value_su

    try:
        value = float(activity_entry["value"])
    except (ValueError, KeyError):
        pass
    else:
        units = activity_entry.get("units")
        if units:
            try:
                pchembl_value_u = convert_standard_units_to_pchembl(
                    units=units,
                    value=value,
                    stats=stats
                )
            except (ValueError, KeyError) as e:
                logger.debug(f"Could not convert value: {e}")
            else:
                return pchembl_value_u

    logger.debug(f"Could not extract pchembl for activity {activity_entry.get('activity_id')}")
    if stats:
        stats.no_activity += 1
    return None


def attach_smiles(
        df: pd.DataFrame,
        batch_size=100
) -> pd.DataFrame:
    """
    Fetch and attach canonical SMILES strings to molecules in the DataFrame.
    Retrieves SMILES from ChEMBL in batches of batch_size (default 100) for efficiency.
    Args:
        df: DataFrame with 'molecule_chembl_id' column
        batch_size: Number of molecules to fetch in each batch
    Returns:
        DataFrame with added 'canonical_smiles' column
    """
    logger.info("Fetching SMILES strings for molecules")

    try:
        molecule_client = new_client.molecule
        molecule_ids = df["molecule_chembl_id"].dropna().unique().tolist()
        logger.info(f"Fetching SMILES for {len(molecule_ids)} unique molecules")

        # batches are needed to speed up the process and make less api calls to chembl
        batches = [molecule_ids[i:i + batch_size] for i in range(0, len(molecule_ids), batch_size)]
        id_to_smiles = {}

        for idx, batch in enumerate(batches):
            mol_with_strs = molecule_client.filter(molecule_chembl_id__in=batch).only(
                ["molecule_chembl_id", "molecule_structures"]
            )
            # mol_with_strs is a list of dicts of size batch_size or less for last batch
            # dicts inside mol_with_strs in form
            # {'molecule_chembl_id': 'some_id',
            # 'molecule_structures': {'canonical_smiles': 'smiles string',
            # 'molfile': 'mol file adj. matrix',
            # 'standard_inchi': 'inchi string',
            # 'standard_inchi_key': 'inchi key'}}
            for mol in mol_with_strs:
                smi = None  # if smiles is not found, it will be dropped later
                if mol.get("molecule_structures"):
                    smi = mol["molecule_structures"].get("canonical_smiles")
                id_to_smiles[mol["molecule_chembl_id"]] = smi

        df = df.copy()
        df["canonical_smiles"] = df["molecule_chembl_id"].map(id_to_smiles)

        valid_smiles = df["canonical_smiles"].notna().sum()
        logger.info(f"Successfully retrieved SMILES for {valid_smiles}/{len(df)} activities")

        return df

    except Exception as e:
        logger.error(f"Error fetching SMILES: {e}")
        raise


def create_base_dataframe(
        activities: list[dict]
) -> pd.DataFrame:
    """
    Extract essential columns from activities list and create initial DataFrame.

    Args:
        activities: List of activity dictionaries from ChEMBL

    Returns:
        DataFrame with essential activity columns
    """
    final_cols = [
        "molecule_chembl_id", "activity_id", "assay_chembl_id",
        "assay_type", "standard_type", "relation"
    ]

    return pd.DataFrame(activities)[final_cols]


def add_pchembl_values(
        df: pd.DataFrame,
        activities: list[dict],
        stats: Optional[ConversionStatistics] = None
) -> pd.DataFrame:
    """
    Calculate and add pChEMBL values to DataFrame.
    Args:
        df: DataFrame with activity data
        activities: Original list of activity dictionaries for pChEMBL extraction
        stats: Optional statistics object to track conversion results
    Returns:
        DataFrame with added 'pchembl_value' column
    """
    df = df.copy()
    df["pchembl_value"] = [retrieve_pchembl_value(act, stats=stats) for act in activities]

    if stats:
        logger.warning(f"  Negative values skipped: {stats.pchembl_negative_values}")
        logger.warning(f"  Unknown units found: {len(stats.unknown_units)} types")
        if len(stats.unknown_units.items()) >= 1:
            sorted_units = sorted(stats.unknown_units.items(), key=lambda x: x[1], reverse=True)
            units_per_line = 5
            lines = ["Unknown units found:"]
            for i in range(0, len(sorted_units), units_per_line):
                batch = sorted_units[i:i + units_per_line]
                line = ", ".join([f"{unit}: {count}" for unit, count in batch])
                lines.append(f"  {line}")
            logger.info("\n".join(lines))

    valid_pchembl = df["pchembl_value"].notna().sum()
    logger.info(f"Successfully calculated pChEMBL for {valid_pchembl}/{len(df)} activities")

    return df


def remove_duplicate_activities(
        df: pd.DataFrame
) -> pd.DataFrame:
    """
    Remove duplicate activities from DataFrame.
    Args:
        df: DataFrame with activity data
    Returns:
        DataFrame without duplicates
    TODO:
        * it usually does nothing because work on every column, idea could be to remove duplicates
            based on smiles (or molecular chembl id and context type later, leaving compound with the
            best activity
    """
    before_dupl_rem = len(df)
    df = df.drop_duplicates()
    after_dupl_rem = len(df)
    logger.info(f"Removed {before_dupl_rem - after_dupl_rem} duplicate activities")

    return df


def save_activities_in_dataframe(
        activities: list[dict[str, str]],
        stats: Optional[ConversionStatistics] = None,
) -> pd.DataFrame:
    """
    Convert list of activity dictionaries to DataFrame, cut off non-essential columns and
    generate pChEMBL values.
    Args:
        activities: List of activity dictionaries from ChEMBL
        stats: Optional statistics object to track fetching results
    Returns:
        DataFrame with activities (without duplicates) and calculated pChEMBL values
    Raises:
        ValueError: If activities list is empty
    """
    logger.info(f"Converting {len(activities)} activities to DataFrame")

    if not activities:
        logger.error("Cannot create DataFrame from empty activities list")
        raise ValueError("No activities found")

    try:
        df = create_base_dataframe(activities)
        df = attach_smiles(df)
        df = add_pchembl_values(df, activities, stats)
        df = remove_duplicate_activities(df)

        return df

    except Exception as e:
        logger.error(f"Error converting activities to DataFrame: {e}")
        raise


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


def retrieve_assay_info(
        activities_df: pd.DataFrame
) -> list[dict]:
    """
    Extract detailed assay information from ChEMBL for all unique assays.
    Args:
        activities_df: DataFrame with 'assay_chembl_id' column
    Returns:
        List of assay information dictionaries
    """
    assays = activities_df["assay_chembl_id"].dropna().unique().tolist()
    logger.info(f"Extracting assay information for {len(assays)} unique assays")
    try:
        all_assays_info = list(
            new_client.assay.filter(assay_chembl_id__in=assays)
            .only(["assay_chembl_id", "assay_cell_type", "bao_label"])
        )
        logger.info(f"Retrieved information for {len(all_assays_info)} assays")

        return all_assays_info

    except Exception as e:
        logger.error(f"Error extracting assay information: {e}")
        raise


def determine_assay_type_auxiliary(
        assay_info: dict
) -> Optional[str]:
    """
    Determine assay context (biochemical/cellular/organism) from BAO label and cell type.
    Args:
        assay_info: Dictionary with assay metadata including 'bao_label' and 'assay_cell_type'
    Returns:
        Context string: "biochemical", "cellular", "organism", or None
    """
    if not assay_info.get("bao_label"):
        return None

    fmt = assay_info["bao_label"].lower()

    bao_to_context = {
        "cell-based format": "cellular",
        "organism-based format": "organism",
        "single protein format": "biochemical",
        "protein format": "biochemical",
        "cell-free format": "biochemical",
        "cell membrane format": "cellular",
        "subcellular format": "cellular",
        "assay format": None,
    }
    context = bao_to_context.get(fmt)
    if not context and assay_info.get("assay_cell_type"):
        return "cellular"

    return context


def create_certain_activity_mapper(
        all_assays_info: list[dict]
) -> dict:
    """
    Create mapping of assay IDs to their experimental contexts.
    Args:
        all_assays_info: List of assay information dictionaries
    Returns:
        Dictionary mapping assay_chembl_id to context type
    """
    logger.info("Mapping assay contexts")

    assay_type = {}
    for assay_info in all_assays_info:
        assay_id = assay_info.get("assay_chembl_id")
        if assay_id and assay_id not in assay_type:
            assay_type[assay_id] = determine_assay_type_auxiliary(assay_info)

    logger.debug(f"Mapped {len(assay_type)} assays to contexts")

    return assay_type


def generate_exact_assay_type(
        activities: pd.DataFrame
) -> pd.DataFrame:
    """
    Add 'context' column to activities based on exact assay metadata from ChEMBL where assay type can be directly taken
    from metadata
    Args:
        activities: DataFrame with activities and assay_id
    Returns:
        DataFrame with added 'context' column
    """
    logger.info("Generating exact assay context types")
    try:
        df = activities.copy()
        all_assays_info = retrieve_assay_info(df)
        ctx_map = create_certain_activity_mapper(all_assays_info)
        df["context"] = df["assay_chembl_id"].map(ctx_map)
        mapped_count = df["context"].notna().sum()
        logger.info(f"Assigned exact context for {mapped_count}/{len(df)} activities exactly from metadata")

        return df

    except Exception as e:
        logger.error(f"Error generating exact assay types: {e}")
        raise


def generate_approx_assay_type_for_row(
        activities: pd.DataFrame
) -> pd.DataFrame:
    """
    Infer missing assay contexts using heuristics based on standard_type and assay_type.
    Heuristics:
    - Ki, Kd → biochemical (not known for cellular or in vivo models)
    - EC50 with functional assay (F) → cellular (used to describe the concentration of active compound to reach
        50% of desirable biological effect, not known for in vivo and IC50 alternative used for biochemical assays)
    - IC50 with binding assay (B) → biochemical (binding assay used for 2 molecules, so usually not known
        for in vivo and cellular effect, IC50 alone very often used for both cellular and biochemical assays)
    Args:
        activities: DataFrame with 'context' (filled partially with assay_exact_type_generator),
            'standard_type', and 'assay_type' columns
    Returns:
        DataFrame with filled 'context' column
    """
    logger.info("Inferring missing assay contexts using heuristics")

    df = activities.copy()
    before_inference = df["context"].isna().sum()

    stype = df["standard_type"].astype(str).str.upper()
    assay_type = df["assay_type"].astype(str).str.upper()
    unknown = df["context"].isna()

    df.loc[unknown & stype.isin(["KI", "KD"]), "context"] = "biochemical"
    df.loc[unknown & (assay_type == "F") & (stype == "EC50"), "context"] = "cellular"
    df.loc[unknown & (assay_type == "B") & (stype == "IC50"), "context"] = "biochemical"
    after_inference = df["context"].isna().sum()
    delta = before_inference - after_inference
    logger.info(f"Assigned exact context for {delta} activities approximately from metadata")

    return df


def retrieve_activity_status(
        activities: pd.DataFrame
) -> pd.DataFrame:
    """
    Classify activities as active/inactive based on pChEMBL value and assay context.

    Activity thresholds by context:
    - biochemical: pChEMBL >= 6.0 (≤ 1 µM)
    - cellular: pChEMBL >= 5.0 (≤ 10 µM)
    - organism: pChEMBL >= 4.5 (≤ ~33 µM)
    - unknown: pChEMBL >= 6.0 (conservative)
    Only considers exact relations (=, ~, <, <=) for reliable classification to filter not active until >
        (which could be concentration or assay limit)
    Args:
        activities: DataFrame with activity data
    Returns:
        DataFrame with added 'is_active' boolean column used together with similarity to identify activity of
            searched hit/lead
    """

    threshold_active = {
        "biochemical": 6.0,
        "cellular": 5.0,
        "organism": 4.5,
        "unknown": 6.0,
    }

    allowed_rel = {"=", "~", None, "<", "<="}

    try:
        out = activities.copy()
        out["pchembl_value"] = pd.to_numeric(out["pchembl_value"], errors="coerce")

        ctx = out.get("context", pd.Series(index=out.index, dtype=object))
        ctx = ctx.fillna("unknown").astype(str)

        cutoff = ctx.map(threshold_active).fillna(threshold_active["unknown"])

        rel_ok = out.get("relation", pd.Series(index=out.index, dtype=object)).isin(allowed_rel)

        out["is_active"] = (out["pchembl_value"] >= cutoff) & rel_ok
        active_count = out["is_active"].sum()
        inactive_count = (~out["is_active"]).sum()
        logger.info(f"Classification complete: {active_count} active, {inactive_count} inactive")

        return out

    except Exception as e:
        logger.error(f"Error classifying activity status: {e}")
        raise


def generate_complete_activity_dataframe(query: str,
                                         organism: str = "Homo sapiens",
                                         limit: Optional[int] = None,
                                         stats: Optional[ConversionStatistics] = None
                                         ) -> pd.DataFrame:
    """
    Main pipeline to generate complete molecular activity dataset for a target query.
    Pipeline steps:
    1. Find targets matching query
    2. Retrieve all bioactivity data
    3. Calculate pChEMBL values and fetch smiles
    4. Determine assay contexts
    5. Determine activity status (active/inactive/None)
    Args:
        query: Target search query (e.g., "CDK2", "kinase")
        organism: Organism filter (default: "Homo sapiens")
        limit: Maximum number of targets to process (default: None, search for all available targets)
        stats: Optional statistics object to track fetching results
    Returns:
        DataFrame with columns: molecule_chembl_id, activity_id, assay_chembl_id,
        pchembl_value, context, canonical_smiles, is_active
    """
    logger.info(f"Starting main pipeline to generate activities for query: '{query}'\n{"=" * 60}\n")

    logger.info(f"Step 1/6: Search for targets matching query")
    targets = find_targets(query, organism=organism, limit=limit)

    logger.info(f"Step 2/6: Retrieving combined activities")
    combined_activities = combine_activities_for_targets(targets["target_chembl_id"].tolist(), stats=stats)

    logger.info(f"Step 3/6: Creating Dataframe with activities")
    activities = save_activities_in_dataframe(combined_activities, stats=stats)

    logger.info(f"Step 4/6: Determining exact assay contexts")
    activities_with_exact_assays = generate_exact_assay_type(activities)

    logger.info(f"Step 5/6: Determining remaining assay contexts")
    activities_with_all_assays = generate_approx_assay_type_for_row(activities_with_exact_assays)

    logger.info(f"Step 6/6: Classifying activity status")
    activities_all = retrieve_activity_status(activities_with_all_assays)

    final_df = activities_all[[
        "molecule_chembl_id", "activity_id", "assay_chembl_id",
        "pchembl_value", "context", "canonical_smiles", "is_active"
    ]]

    return final_df
