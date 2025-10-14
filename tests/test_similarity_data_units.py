import pandas as pd
import pytest
from rdkit import Chem, RDLogger
from src.utils.similarity_data_utils import calculate_tanimoto_similarity, get_tanimoto_similarity_for_query, \
    generate_similarity_column, similarity_filter


@pytest.fixture(scope="session", autouse=True)
def suppress_rdkit_logs():
    """
    Suppress RDKit warnings and error messages for all tests.

    Disables rdApp logging to prevent noise in test output from
    invalid SMILES parsing and molecule operations.
    """
    RDLogger.DisableLog('rdApp.*')

    yield

    RDLogger.EnableLog('rdApp.*')


class TestCalculateTanimotoSimilarity:
    """Tests for Tanimoto similarity calculation between molecules"""

    def test_identical_molecules(self):
        """Test that identical molecules have similarity of 1.0"""
        smiles = "CC(N(C([C@H]1CC[C@@H](CC1)C)=O)c2c(C([O-])=O)cc(Oc3c(C(F)(F)F)cccn3)cc2)C"
        mol = Chem.MolFromSmiles(smiles)
        similarity = calculate_tanimoto_similarity(mol, smiles)

        assert similarity == pytest.approx(1.0, abs=0.001)

    def test_different_molecules(self):
        """Test similarity between different molecules is between 0 and 1"""
        mol1 = Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1")
        smiles2 = "O=C(C)Oc1ccccc1C(=O)O"
        similarity = calculate_tanimoto_similarity(mol1, smiles2)

        assert 0.0 <= similarity <= 1.0
        assert similarity < 1.0

    def test_similar_molecules(self):
        """Test that similar molecules have high similarity"""
        mol1 = Chem.MolFromSmiles("CC(OC1=C(O[C@@H]2[C@]34CCN(C)[C@@H]([C@@H]4C=C[C@@H]2OC(C)=O)C5)C3=C5C=C1)=O")
        smiles2 = "CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"
        similarity = calculate_tanimoto_similarity(mol1, smiles2)

        assert similarity > 0.5

    def test_invalid_search_molecule_raises_error(self):
        """Test that invalid search molecule raises ValueError"""
        with pytest.raises(ValueError, match="Invalid search SMILES"):
            calculate_tanimoto_similarity(None, "O=C(C)Oc1ccccc1C(=O)O")

    def test_none_target_smiles_returns_none(self):
        """Test that None target SMILES returns None"""
        real_mol = Chem.MolFromSmiles("O=C(C)Oc1ccccc1C(=O)O")
        similarity = calculate_tanimoto_similarity(real_mol, None)

        assert similarity is None


class TestGetTanimotoSimilarityForQuery:
    """Tests for creating similarity scoring function"""

    def test_returns_callable(self):
        """Test that function returns a callable"""
        scorer = get_tanimoto_similarity_for_query("O=C1C(C(OC)=O)CCCC1")

        assert callable(scorer)

    def test_scorer_calculates_similarity(self):
        """Test that returned callable calculates similarity correctly"""
        scorer = get_tanimoto_similarity_for_query("O=C1C(C(OC)=O)CCCC1")
        similarity = scorer("O=C1C(C(OC)=O)CCCC1")

        assert similarity == pytest.approx(1.0, abs=0.001)

    def test_scorer_handles_invalid_smiles(self):
        """Test that scorer returns None for invalid SMILES"""
        scorer = get_tanimoto_similarity_for_query("O=C1C(C(OC)=O)CCCC1")
        similarity = scorer("invalid_smiles")

        assert similarity is None

    def test_invalid_query_smiles_raises_error(self):
        """Test that invalid query SMILES raises ValueError"""
        with pytest.raises(ValueError, match="Invalid SMILES"):
            get_tanimoto_similarity_for_query("invalid_smiles")


class TestGenerateSimilarityColumn:
    """Tests for adding similarity column to DataFrame"""

    def test_adds_similarity_column(self):
        """Test that similarity column is added to DataFrame"""
        df = pd.DataFrame({
            "canonical_smiles": ["CCO", "CCCO", "c1ccccc1"],
            "molecule_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3"]
        })

        result = generate_similarity_column(df, "CCO")

        assert "tanimoto_similarity" in result.columns
        assert len(result) == 3

    def test_similarity_values_correct(self):
        """Test that similarity values are calculated correctly"""
        df = pd.DataFrame({
            "canonical_smiles": ["CCO", "CCCO"],
            "molecule_chembl_id": ["CHEMBL1", "CHEMBL2"]
        })

        result = generate_similarity_column(df, "CCO")

        assert result.loc[0, "tanimoto_similarity"] == pytest.approx(1.0, abs=0.001)
        assert 0.0 < result.loc[1, "tanimoto_similarity"] < 1.0

    def test_handles_invalid_smiles_in_dataframe(self):
        """Test that invalid SMILES in DataFrame result in NaN"""
        df = pd.DataFrame({
            "canonical_smiles": ["CCO", "invalid_smiles", "CCCO"],
            "molecule_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3"]
        })

        result = generate_similarity_column(df, "CCO")

        assert result.loc[0, "tanimoto_similarity"] == pytest.approx(1.0, abs=0.001)
        assert pd.isna(result.loc[1, "tanimoto_similarity"])
        assert result.loc[2, "tanimoto_similarity"] > 0

    def test_custom_smiles_column(self):
        """Test using custom SMILES column name"""
        df = pd.DataFrame({
            "my_smiles": ["CCO", "CCCO"],
            "molecule_chembl_id": ["CHEMBL1", "CHEMBL2"]
        })

        result = generate_similarity_column(df, "CCO", smiles_col="my_smiles")

        assert "tanimoto_similarity" in result.columns
        assert result.loc[0, "tanimoto_similarity"] == pytest.approx(1.0, abs=0.001)

    def test_does_not_modify_original_dataframe(self):
        """Test that original DataFrame is not modified"""
        df = pd.DataFrame({
            "canonical_smiles": ["CCO", "CCCO"],
            "molecule_chembl_id": ["CHEMBL1", "CHEMBL2"]
        })

        original_columns = df.columns.tolist()
        result = generate_similarity_column(df, "CCO")

        assert df.columns.tolist() == original_columns
        assert "tanimoto_similarity" not in df.columns
        assert "tanimoto_similarity" in result.columns


class TestSimilarityFilter:
    """Tests for filtering molecules by similarity"""

    def test_filters_by_minimum_similarity(self):
        """Test that molecules below threshold are filtered out"""
        df = pd.DataFrame({
            "tanimoto_similarity": [0.9, 0.7, 0.85],
            "molecule_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3"],
            "pchembl_value": [6.5, 7.0, 6.0]
        })

        result = similarity_filter(df, min_similarity=0.8, max_molecules=None)

        assert len(result) == 2
        assert "CHEMBL1" in result["molecule_chembl_id"].values
        assert "CHEMBL3" in result["molecule_chembl_id"].values
        assert "CHEMBL2" not in result["molecule_chembl_id"].values

    def test_limits_number_of_molecules(self):
        """Test that max_molecules limits unique molecules"""
        df = pd.DataFrame({
            "tanimoto_similarity": [0.95, 0.95, 0.90, 0.90, 0.85, 0.85],
            "molecule_chembl_id": ["MOL1", "MOL1", "MOL2", "MOL2", "MOL3", "MOL3"],
            "pchembl_value": [7.0, 6.5, 6.8, 6.2, 6.5, 6.0]
        })

        result = similarity_filter(df, min_similarity=0.8, max_molecules=2)

        unique_molecules = result["molecule_chembl_id"].nunique()
        assert unique_molecules == 2
        assert "MOL1" in result["molecule_chembl_id"].values
        assert "MOL2" in result["molecule_chembl_id"].values
        assert "MOL3" not in result["molecule_chembl_id"].values

    def test_keeps_all_rows_for_selected_molecules(self):
        """Test that all activity rows for selected molecules are kept"""
        df = pd.DataFrame({
            "tanimoto_similarity": [0.95, 0.95, 0.95],
            "molecule_chembl_id": ["MOL1", "MOL1", "MOL1"],
            "pchembl_value": [7.0, 6.5, 6.8]
        })

        result = similarity_filter(df, min_similarity=0.9, max_molecules=1)

        assert len(result) == 3  # All 3 rows for MOL1

    def test_removes_molecules_without_pchembl(self):
        """Test that molecules with all None pchembl values are removed"""
        df = pd.DataFrame({
            "tanimoto_similarity": [0.95, 0.95, 0.90],
            "molecule_chembl_id": ["MOL1", "MOL2", "MOL2"],
            "pchembl_value": [7.0, None, None]
        })

        result = similarity_filter(df, min_similarity=0.85, max_molecules=None)

        assert len(result) == 1
        assert result["molecule_chembl_id"].iloc[0] == "MOL1"

    def test_sorts_by_similarity_descending(self):
        """Test that results are sorted by similarity (highest first)"""
        df = pd.DataFrame({
            "tanimoto_similarity": [0.85, 0.95, 0.90],
            "molecule_chembl_id": ["MOL1", "MOL2", "MOL3"],
            "pchembl_value": [6.0, 7.0, 6.5]
        })

        result = similarity_filter(df, min_similarity=0.8, max_molecules=None)

        similarities = result["tanimoto_similarity"].tolist()
        assert similarities == sorted(similarities, reverse=True)

    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError"""
        df = pd.DataFrame({
            "tanimoto_similarity": [],
            "molecule_chembl_id": [],
            "pchembl_value": []
        })

        with pytest.raises(ValueError, match="Cannot filter empty DataFrame"):
            similarity_filter(df)

    def test_missing_required_columns_raises_error(self):
        """Test that missing required columns raises ValueError"""
        df = pd.DataFrame({
            "tanimoto_similarity": [0.9],
            "molecule_chembl_id": ["MOL1"]
            # Missing pchembl_value
        })

        with pytest.raises(ValueError, match="DataFrame must contain columns"):
            similarity_filter(df)

    def test_invalid_min_similarity_raises_error(self):
        """Test that min_similarity outside [0,1] raises ValueError"""
        df = pd.DataFrame({
            "tanimoto_similarity": [0.9],
            "molecule_chembl_id": ["MOL1"],
            "pchembl_value": [6.0]
        })

        with pytest.raises(ValueError, match="min_similarity must be between 0 and 1"):
            similarity_filter(df, min_similarity=1.5)

    def test_invalid_max_molecules_raises_error(self):
        """Test that max_molecules < 1 raises ValueError"""
        df = pd.DataFrame({
            "tanimoto_similarity": [0.9],
            "molecule_chembl_id": ["MOL1"],
            "pchembl_value": [6.0]
        })

        with pytest.raises(ValueError, match="max_molecules must be positive or None"):
            similarity_filter(df, max_molecules=0)

    def test_no_molecules_pass_threshold(self):
        """Test behavior when no molecules pass similarity threshold"""
        df = pd.DataFrame({
            "tanimoto_similarity": [0.5, 0.6, 0.7],
            "molecule_chembl_id": ["MOL1", "MOL2", "MOL3"],
            "pchembl_value": [6.0, 6.5, 7.0]
        })

        result = similarity_filter(df, min_similarity=0.9, max_molecules=None)

        assert result.empty
        assert list(result.columns) == list(df.columns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
