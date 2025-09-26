"""The workflow process so far:
1. Generate a list of all target IDs by using find_targets
2. Generate activities using get_activities_for_target for every target
    and save them in the combined list of all activities for every target
    using combine_activities_for_target
    TODO write combine_activities_for_target
3. Create a dataframe from the combined activities list and filter it to
    remove duplicates
    TODO get_activities_for_target needs to be rewritten to make a dataframe earlier
    TODO write filter_duplicated_activities to filter duplicates
4. Generate pchembl values and save them in a separate column
    TODO rewrite pchembl_extractor to work with a dataframe
    TODO write activities_df_clean_up to drop unimportant columns
5. Fetch smiles and save them in a separate column using the attach_smiles function
6. Generate "is_active" column using activity_status_generator
    TODO write activity_status_generator
7. Make a function to save a dataframe with activities to the SQL db (SQLite) and
    extract it from the database
    TODO write save_to_SQL and extract_from_SQL
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
"""

from typing import Optional, Callable
from chembl_webresource_client.new_client import new_client
import pandas as pd
import re
import math
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdMolDescriptors
from functools import partial

UNIT_FACTORS = {  # TODO create and save in constants.py file
    "m": 1.0,
    "mm": 1e-3,
    "um": 1e-6,
    "nm": 1e-9,
    "pm": 1e-12,
}


def find_targets(query: str, organism="Homo sapiens", limit=None):
    t = new_client.target
    hits = t.search(query)
    rows = []
    for h in hits:
        if organism and h.get("organism") == organism:
            rows.append(
                {
                    "target_chembl_id": h["target_chembl_id"],
                    "pref_name": h.get("pref_name"),
                    "organism": h.get("organism"),
                    "target_type": h.get("target_type"),
                }
            )
            if limit and len(rows) >= limit:
                break
    return pd.DataFrame(rows)


def get_activities_for_target(target_chembl_id: str,
                              types=("IC50", "Ki", "Kd", "EC50"),
                              ) -> list[dict[str, str]]:
    act = new_client.activity
    fields = [
        "activity_id", "assay_chembl_id", "assay_type", "assay_confidence_score",
        "molecule_chembl_id",
        "standard_type", "standard_value", "standard_units", "relation",
        "pchembl_value", "target_chembl_id"
    ]
    q = act.filter(
        target_chembl_id=target_chembl_id,
        standard_type__in=list(types)
    ).only(fields)
    data = list(q)
    return data


def standard_unit_convertor_to_pchembl(units: str, value: float) -> float:
    if value <= 0:
        raise ValueError("value must be positive")
    u = units.strip().lower()
    u = u.replace("µ", "u")
    u = re.sub(r"\s+", "", u)
    u = u.replace("mol/l", "").replace("molperl", "")

    if u in UNIT_FACTORS.keys():
        standard_value = UNIT_FACTORS[units] * value
        return - math.log10(standard_value)
    else:
        raise ValueError(f"Unknown unit {units}")


def pchembl_extractor(activity_entry: dict[str, str]) -> Optional[float]:
    if activity_entry.get("pchembl_value"):
        return float(activity_entry["pchembl_value"])
    if activity_entry.get("standard_value") and activity_entry["standard_value"].isnumeric():
        units = activity_entry.get("standard_units")
        if units:
            return standard_unit_convertor_to_pchembl(units=units,
                                                      value=float
                                                      (activity_entry["standard_value"])
                                                      )

    if activity_entry.get("value") and activity_entry["value"].isnumeric():
        units = activity_entry.get("units")
        if units:
            return standard_unit_convertor_to_pchembl(units=units,
                                                      value=float(activity_entry["value"])
                                                      )

    return None


def activities_to_dataframe(activities: list[dict[str, Optional[str]]]) -> pd.DataFrame:
    if not activities:
        raise ValueError("No activities found")
    activities_df = pd.DataFrame(activities, index=None)
    activities_df = activities_df[["molecule_chembl_id",
                                   "activity_id",
                                   "assay_chembl_id",
                                   "assay_type",
                                   ]]
    activities_df["pchembl_value"] = [pchembl_extractor(activity_entry) for activity_entry in activities]
    return activities_df


def attach_smiles(df: pd.DataFrame) -> pd.DataFrame:
    mol = new_client.molecule
    ids = df["molecule_chembl_id"].dropna().unique().tolist()
    chunks = [ids[i:i + 100] for i in range(0, len(ids), 100)]
    id_to_smiles = {}
    for chunk in chunks:
        mres = mol.filter(molecule_chembl_id__in=chunk).only(
            ["molecule_chembl_id", "molecule_structures"]
        )
        for m in mres:
            smi = None
            if m.get("molecule_structures"):
                smi = m["molecule_structures"].get("canonical_smiles")
            id_to_smiles[m["molecule_chembl_id"]] = smi
    df = df.copy()
    df["canonical_smiles"] = df["molecule_chembl_id"].map(id_to_smiles)
    return df


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
