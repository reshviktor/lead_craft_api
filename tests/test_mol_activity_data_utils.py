import pandas as pd
import pytest
from collections import defaultdict
from unittest.mock import patch, Mock
from src.utils.mol_activity_data_utils import ConversionStatistics, find_targets, get_activities_for_target, \
    combine_activities_for_targets, convert_standard_units_to_pchembl, retrieve_pchembl_value, create_base_dataframe, \
    retrieve_assay_info, determine_assay_type_auxiliary, retrieve_activity_status, attach_smiles, \
    generate_complete_activity_dataframe, add_pchembl_values, save_activities_in_dataframe, \
    create_certain_activity_mapper, generate_exact_assay_type, generate_approx_assay_type_for_row, \
    remove_duplicate_activities


class TestConversionStatistics:
    """Tests for ConversionStatistics tracking dataclass"""

    def test_default_initialization(self):
        """Test default initialization"""
        stats = ConversionStatistics()
        assert isinstance(stats.chembl_targets, dict)
        assert isinstance(stats.unknown_units, defaultdict)
        assert stats.invalid_values == 0
        assert stats.pchembl_negative_values == 0
        assert stats.no_activity == 0

    def test_correct_statistics_accumulation(self):
        """Test that statistics can be accumulated correctly"""
        stats = ConversionStatistics()
        stats.chembl_targets["CHEMBL1000"] = 150
        stats.chembl_targets["CHEMBL2000"] = 200
        stats.unknown_units["mg/L"] += 5
        stats.unknown_units["ug/mL"] += 3
        stats.pchembl_negative_values += 2
        stats.no_activity += 10
        stats.invalid_values += 1

        assert len(stats.chembl_targets) == 2
        assert stats.chembl_targets["CHEMBL1000"] == 150
        assert stats.unknown_units["mg/L"] == 5
        assert stats.unknown_units["ug/mL"] == 3
        assert stats.pchembl_negative_values == 2
        assert stats.no_activity == 10
        assert stats.invalid_values == 1


@pytest.fixture
def sample_targets():
    """Sample target search results from ChEMBL API"""
    return [
        {
            "target_chembl_id": "CHEMBL100",
            "pref_name": "EGFR/PPP1CA",
            "organism": "Homo sapiens",
            "target_type": "SINGLE PROTEIN"
        },
        {
            "target_chembl_id": "CHEMBL200",
            "pref_name": "CCN2-EGFR",
            "organism": "Homo sapiens",
            "target_type": "SINGLE PROTEIN"
        },
        {
            "target_chembl_id": "CHEMBL300",
            "pref_name": "Epidermal growth factor receptor",
            "organism": "Rattus norvegicus",
            "target_type": "SINGLE PROTEIN"
        },
        {
            "target_chembl_id": "CHEMBL400",
            "pref_name": "Protein cereblon/Epidermal growth factor receptor",
            "organism": "Homo sapiens",
            "target_type": "PROTEIN-PROTEIN INTERACTION"
        }
    ]


@pytest.fixture
def sample_activities():
    """Sample activity data from ChEMBL"""
    return [
        {
            'activity_id': 10001,
            'assay_chembl_id': 'CHEMBL3001',
            'assay_type': 'B',
            'molecule_chembl_id': 'CHEMBL7000',
            'pchembl_value': '7.39',
            'relation': '=',
            'standard_type': 'IC50',
            'standard_units': 'nM',
            'standard_value': '41.0',
            'target_chembl_id': 'CHEMBL2000',
            'type': 'IC50',
            'units': 'uM',
            'value': '0.041'
        },
        {
            'activity_id': 10002,
            'assay_chembl_id': 'CHEMBL3001',
            'assay_type': 'F',
            'molecule_chembl_id': 'CHEMBL7001',
            'pchembl_value': None,
            'relation': '=',
            'standard_type': 'IC50',
            'standard_units': 'nM',
            'standard_value': '300.0',
            'target_chembl_id': 'CHEMBL2000',
            'type': 'IC50',
            'units': 'uM',
            'value': '0.3'
        },
        {
            'activity_id': 10003,
            'assay_chembl_id': 'CHEMBL3001',
            'assay_type': 'F',
            'molecule_chembl_id': 'CHEMBL7002',
            'pchembl_value': None,
            'relation': '=',
            'standard_type': None,
            'standard_units': None,
            'standard_value': None,
            'target_chembl_id': 'CHEMBL2000',
            'type': 'IC50',
            'units': 'uM',
            'value': '7.82'
        }
    ]


@pytest.fixture
def sample_activities_ex_2():
    """Sample activity data from ChEMBL"""
    return [
        {
            'activity_id': 10004,
            'assay_chembl_id': 'CHEMBL3002',
            'assay_type': 'B',
            'molecule_chembl_id': 'CHEMBL69960',
            'pchembl_value': None,
            'relation': '=',
            'standard_type': None,
            'standard_units': None,
            'standard_value': None,
            'target_chembl_id': ' CHEMBL2001',
            'type': 'IC50',
            'units': 'not_standard_unit',
            'value': '0.17'
        }]


@pytest.fixture
def sample_molecule_structures():
    """Sample molecule structures data from ChEMBL API"""
    return [
        {
            'molecule_chembl_id': 'CHEMBL7000',
            'molecule_structures': {
                'canonical_smiles': 'O=C(O)c1ccc(O)cc1',
                'molfile': '\n RDKit  2D some large matrix',
                'standard_inchi': 'InChI=1S/C7H6O3/c8-6-3-1-5(2-4-6)7(9)10/h1-4,8H,(H,9,10)',
                'standard_inchi_key': 'AAA-AAA-N'
            }
        },
        {
            'molecule_chembl_id': 'CHEMBL7001',
            'molecule_structures': {
                'canonical_smiles': 'O=C(O)/C=C/c1ccc(O)c(O)c1',
                'molfile': '\n RDKit  2D some large matrix',
                'standard_inchi': 'InChI=1S/C9H8O4/c10-7-3-1-6(5-8(7)11)2-4-9(12)13/h1-5,10-11H,(H,12,13)/b4-2+',
                'standard_inchi_key': 'BBB-BBB-N'
            }
        },
        {
            'molecule_chembl_id': 'CHEMBL7002',
            'molecule_structures': {
                'canonical_smiles': 'O=[N+]([O-])c1ccc(O)c(O)c1',
                'molfile': '\n RDKit  2D some large matrix',
                'standard_inchi': 'InChI=1S/C6H5NO4/c8-5-2-1-4(7(10)11)3-6(5)9/h1-3,8-9H',
                'standard_inchi_key': 'CCC-CCC-N'
            }
        }
    ]


@pytest.fixture
def sample_assay_info():
    """Sample assay information"""
    return [
        {
            "assay_chembl_id": "CHEMBL001",
            "assay_cell_type": None,
            "bao_label": "single protein format"
        },
        {
            "assay_chembl_id": "CHEMBL002",
            "assay_cell_type": "BL2",
            "bao_label": "cell-based format"
        }
    ]


class TestChEMBLAPICallsToFetchTargetsAndActivities:
    """Tests for functions that make ChEMBL API calls (with mocking)"""

    @patch('src.utils.mol_activity_data_utils.new_client')
    def test_find_targets(self, mock_client, sample_targets):
        """Test target search without limit returns all matching targets (3 targets returned and 1 excluded)"""
        mock_target = Mock()
        mock_target.search.return_value = sample_targets
        mock_client.target = mock_target
        result = find_targets("EGFR", organism="Homo sapiens")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert all(result["organism"] == "Homo sapiens")
        expected_ids = ["CHEMBL100", "CHEMBL200", "CHEMBL400"]
        assert list(result["target_chembl_id"]) == expected_ids

    @patch('src.utils.mol_activity_data_utils.new_client')
    def test_find_targets_api_error(self, mock_client):
        """Test that API errors are properly raised"""
        mock_target = Mock()
        mock_target.search.side_effect = Exception("ChEMBL API connection failed")
        mock_client.target = mock_target

        with pytest.raises(Exception, match="ChEMBL API connection failed"):
            find_targets("EGFR", organism="Homo sapiens")

    @patch('src.utils.mol_activity_data_utils.new_client')
    def test_get_activities_for_target(self, mock_client, sample_activities):
        """Test activity retrieval with mocked API"""
        stats = ConversionStatistics()
        mock_activity = Mock()
        mock_only = Mock()
        mock_only.only.return_value = sample_activities
        mock_activity.filter.return_value = mock_only
        mock_client.activity = mock_activity
        result = get_activities_for_target("CHEMBL2000", stats=stats)

        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]['activity_id'] == 10001
        assert result[1]['activity_id'] == 10002
        assert result[2]['activity_id'] == 10003
        assert stats.chembl_targets["CHEMBL2000"] == 3
        mock_activity.filter.assert_called_once()
        mock_only.only.assert_called_once()

    @patch('src.utils.mol_activity_data_utils.new_client')
    def test_get_activities_for_target_empty_result(self, mock_client):
        """Test handling of targets with no activities"""
        stats = ConversionStatistics()
        mock_activity = Mock()
        mock_only = Mock()
        mock_only.only.return_value = []
        mock_activity.filter.return_value = mock_only
        mock_client.activity = mock_activity
        result = get_activities_for_target("CHEMBL9999", stats=stats)

        assert isinstance(result, list)
        assert len(result) == 0
        assert stats.chembl_targets["CHEMBL9999"] == 0

    @patch('src.utils.mol_activity_data_utils.new_client')
    def test_get_activities_for_target_api_error(self, mock_client):
        """Test handling of targets with no activities"""
        mock_activity = Mock()
        mock_only = Mock()
        mock_only.only.side_effect = Exception("ChEMBL API connection failed")
        mock_activity.filter.return_value = mock_only
        mock_client.activity = mock_activity
        with pytest.raises(Exception, match="ChEMBL API connection failed"):
            get_activities_for_target("CHEMBL9999")

    @patch('src.utils.mol_activity_data_utils.new_client')
    def test_combine_activities_for_targets(self, mock_client, sample_activities, sample_activities_ex_2):
        """Test combining activities from multiple targets"""
        stats = ConversionStatistics()
        mock_activity = Mock()
        mock_only = Mock()
        mock_only.only.side_effect = [
            sample_activities,
            sample_activities_ex_2
        ]
        mock_activity.filter.return_value = mock_only
        mock_client.activity = mock_activity
        result = combine_activities_for_targets(["CHEMBL2000", "CHEMBL2001"], stats=stats)

        assert isinstance(result, list)
        assert len(result) == 4
        assert result[0]['activity_id'] == 10001
        assert result[0]['target_chembl_id'] == 'CHEMBL2000'
        assert result[1]['activity_id'] == 10002
        assert result[1]['target_chembl_id'] == 'CHEMBL2000'
        assert result[2]['activity_id'] == 10003
        assert result[2]['target_chembl_id'] == 'CHEMBL2000'
        assert result[3]['activity_id'] == 10004
        assert result[3]['target_chembl_id'].strip() == 'CHEMBL2001'
        assert stats.chembl_targets["CHEMBL2000"] == 3
        assert stats.chembl_targets["CHEMBL2001"] == 1

    @patch('src.utils.mol_activity_data_utils.new_client')
    def test_combine_activities_for_targets_without_activities(self, mock_client):
        """Test combining activities from multiple targets without activities"""
        mock_activity = Mock()
        mock_only = Mock()
        mock_only.only.side_effect = [[], []]
        mock_activity.filter.return_value = mock_only
        mock_client.activity = mock_activity
        with pytest.raises(ValueError, match="No activities found"):
            combine_activities_for_targets(
                ["CHEMBL2000", "CHEMBL2001"],
            )


class TestUnitConversion:
    """Tests for unit conversion to pChEMBL scale"""

    def test_nanomolar_conversion(self):
        """Test conversion from nM to pChEMBL"""
        result = convert_standard_units_to_pchembl("nM", 100)
        expected = 7.0
        assert result == pytest.approx(expected, rel=1e-9)

    def test_micromolar_conversion(self):
        """Test conversion from µM to pChEMBL"""
        result = convert_standard_units_to_pchembl("µM", 10)
        expected = 5.0
        assert result == pytest.approx(expected, rel=1e-9)

    @pytest.mark.parametrize(
        'units_to_test, expected',
        [("uM", 6.0), ("µM", 6.0), ("um", 6.0), ("UM", 6.0)]
    )
    def test_various_unit_formats(self, units_to_test, expected):
        """Test various unit string formats"""
        result = convert_standard_units_to_pchembl(units_to_test, 1)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_negative_value(self):
        """Test that negative values return None"""
        stats = ConversionStatistics()
        result = convert_standard_units_to_pchembl("nM", -100, stats)
        assert result is None
        assert stats.pchembl_negative_values == 1

    def test_zero_value(self):
        """Test that zero values return None"""
        result = convert_standard_units_to_pchembl("nM", 0)
        assert result is None

    def test_unknown_unit(self):
        """Test handling of unknown units"""
        stats = ConversionStatistics()
        result = convert_standard_units_to_pchembl("unknown_unit", 100, stats)
        assert result is None
        assert stats.unknown_units["unknown_unit"] == 1


class TestRetrievePChEMBLValue:
    """Tests for pChEMBL value extraction using sample_activities fixtures"""

    def test_extract_direct_pchembl_value(self, sample_activities):
        stats = ConversionStatistics()
        """Test extraction from direct pchembl_value field (Activity 10001)"""
        activity = sample_activities[0]
        result = retrieve_pchembl_value(activity, stats)

        assert result == 7.39
        assert stats.no_activity == 0
        assert len(stats.unknown_units) == 0

    def test_calculate_from_standard_value(self, sample_activities):
        stats = ConversionStatistics()
        """Test calculation from standard_value/standard_units (Activity 10002)"""
        activity = sample_activities[1]
        result = retrieve_pchembl_value(activity, stats)
        assert result is not None
        assert result == pytest.approx(6.52, rel=0.01)
        assert stats.no_activity == 0

    def test_calculate_from_value_units_fallback(self, sample_activities):
        """Test calculation from value/units as fallback (Activity 10003)"""
        stats = ConversionStatistics()
        activity = sample_activities[2]
        result = retrieve_pchembl_value(activity, stats)
        assert result is not None
        assert result == pytest.approx(5.11, rel=0.01)
        assert stats.no_activity == 0

    def test_unknown_units_returns_none(self, sample_activities_ex_2):
        """Test that unknown units return None and track in stats (Activity 10004)"""
        stats = ConversionStatistics()
        activity = sample_activities_ex_2[0]
        result = retrieve_pchembl_value(activity, stats)
        assert result is None
        assert stats.unknown_units['not_standard_unit'] == 1
        assert stats.no_activity == 1

    def test_invalid_pchembl_value_string(self):
        """Test handling of non-numeric pchembl_value"""
        stats = ConversionStatistics()
        activity = {
            'activity_id': 99999,
            'pchembl_value': 'invalid_string',
            'standard_value': '100',
            'standard_units': 'nM'
        }
        result = retrieve_pchembl_value(activity, stats)
        assert result is not None
        assert result == pytest.approx(7.0, rel=0.01)


class TestAttachSMILES:
    """Tests for SMILES fetching and attachment with batch processing"""

    @patch('src.utils.mol_activity_data_utils.new_client')
    def test_attach_smiles_with_batching(self, mock_client, sample_molecule_structures):
        """Test SMILES fetching with batch_size=2, 3 molecules → 2 API calls."""
        df = pd.DataFrame({
            "molecule_chembl_id": ["CHEMBL7000", "CHEMBL7001", "CHEMBL7002"],
            "activity_id": [1, 2, 3]
        })
        mock_molecule = Mock()
        mock_only = Mock()
        mock_only.only.side_effect = [
            sample_molecule_structures[:2],  # CHEMBL7000, CHEMBL7001
            sample_molecule_structures[2:]  # CHEMBL7002
        ]
        mock_molecule.filter.return_value = mock_only
        mock_client.molecule = mock_molecule
        result = attach_smiles(df, batch_size=2)

        assert mock_molecule.filter.call_count == 2
        first_call_ids = mock_molecule.filter.call_args_list[0][1]['molecule_chembl_id__in']
        second_call_ids = mock_molecule.filter.call_args_list[1][1]['molecule_chembl_id__in']
        assert first_call_ids == ["CHEMBL7000", "CHEMBL7001"]
        assert second_call_ids == ["CHEMBL7002"]
        assert "canonical_smiles" in result.columns
        assert result.loc[0, "canonical_smiles"] == "O=C(O)c1ccc(O)cc1"
        assert result.loc[1, "canonical_smiles"] == "O=C(O)/C=C/c1ccc(O)c(O)c1"
        assert result.loc[2, "canonical_smiles"] == "O=[N+]([O-])c1ccc(O)c(O)c1"
        assert result["canonical_smiles"].notna().sum() == 3

    @patch('src.utils.mol_activity_data_utils.new_client')
    def test_attach_smiles_with_missing_structures(self, mock_client, sample_molecule_structures):
        """Test handling of molecules without structure data."""
        df = pd.DataFrame({
            "molecule_chembl_id": ["CHEMBL7000", "CHEMBL7001"]
        })

        mock_molecule = Mock()
        mock_only = Mock()
        mock_only.only.return_value = [
            sample_molecule_structures[0],
            {
                "molecule_chembl_id": "CHEMBL7001",
                "molecule_structures": None
            }
        ]
        mock_molecule.filter.return_value = mock_only
        mock_client.molecule = mock_molecule
        result = attach_smiles(df, batch_size=100)

        assert result.loc[0, "canonical_smiles"] == "O=C(O)c1ccc(O)cc1"
        assert pd.isna(result.loc[1, "canonical_smiles"])
        assert result["canonical_smiles"].notna().sum() == 1

    def test_attach_smiles_invalid_batch_size(self):
        """Test that batch_size <= 0 raises ValueError."""
        df = pd.DataFrame({
            "molecule_chembl_id": ["CHEMBL7000", "CHEMBL7001"]
        })
        with pytest.raises(ValueError, match="batch_size 0 cannot be less than 0"):
            attach_smiles(df, batch_size=0)

        with pytest.raises(ValueError, match="batch_size -5 cannot be less than 0"):
            attach_smiles(df, batch_size=-5)

    def test_attach_smiles_empty_dataframe(self):
        """Test that empty DataFrame raises ValueError."""
        df = pd.DataFrame({"molecule_chembl_id": []})

        with pytest.raises(ValueError, match="Empty DataFrame received from ChEMBL"):
            attach_smiles(df, batch_size=100)


def test_create_base_dataframe(sample_activities):
    """Test base DataFrame creation"""
    df = create_base_dataframe(sample_activities)
    expected_cols = [
        "molecule_chembl_id", "activity_id", "assay_chembl_id",
        "assay_type", "standard_type", "relation", "target_chembl_id"
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == 3


class TestAddPchEMBL:
    def test_add_pchembl_values_with_stats_tracking(self, sample_activities, sample_activities_ex_2):
        """Test that statistics are correctly tracked"""
        stats = ConversionStatistics()
        df = pd.DataFrame({
            'activity_id': [10001, 10002, 10003, 10004],
            'molecule_chembl_id': ['CHEMBL7000', 'CHEMBL7001', 'CHEMBL7002', 'CHEMBL69960']
        })
        all_activities = sample_activities + sample_activities_ex_2
        stats = ConversionStatistics()
        result = add_pchembl_values(df, all_activities, stats)

        assert stats.unknown_units['not_standard_unit'] == 1
        assert result['pchembl_value'].notna().sum() == 3
        assert result.loc[3, 'pchembl_value'] is None or pd.isna(result.loc[3, 'pchembl_value'])
        assert pd.api.types.is_numeric_dtype(result['pchembl_value'])
        assert isinstance(result.loc[0, 'pchembl_value'], (float, int)) or pd.isna(result.loc[0, 'pchembl_value'])

    def test_add_pchembl_values_without_stats(self, sample_activities):
        """Test that function works without stats parameter"""
        df = pd.DataFrame({
            'activity_id': [10001, 10002, 10003],
            'molecule_chembl_id': ['CHEMBL7000', 'CHEMBL7001', 'CHEMBL7002']
        })
        result = add_pchembl_values(df, sample_activities, stats=None)

        assert 'pchembl_value' in result.columns
        assert result['pchembl_value'].notna().sum() == 3
        assert pd.api.types.is_numeric_dtype(result['pchembl_value'])

    def test_add_pchembl_values_preserves_other_columns(self, sample_activities):
        """Test that other DataFrame columns are preserved"""
        df = pd.DataFrame({
            'activity_id': [10001, 10002, 10003],
            'molecule_chembl_id': ['CHEMBL7000', 'CHEMBL7001', 'CHEMBL7002'],
            'assay_chembl_id': ['ASSAY1', 'ASSAY2', 'ASSAY3'],
            'standard_type': ['IC50', 'IC50', 'IC50']
        })
        stats = ConversionStatistics()
        result = add_pchembl_values(df, sample_activities, stats)

        assert 'activity_id' in result.columns
        assert 'molecule_chembl_id' in result.columns
        assert 'assay_chembl_id' in result.columns
        assert 'standard_type' in result.columns
        assert list(result['activity_id']) == [10001, 10002, 10003]
        assert list(result['assay_chembl_id']) == ['ASSAY1', 'ASSAY2', 'ASSAY3']
        assert pd.api.types.is_numeric_dtype(result['pchembl_value'])
        assert result.loc[0, 'pchembl_value'] == pytest.approx(7.39, rel=0.01)

    def test_add_pchembl_values_empty_activities(self):
        """Test handling of empty activities list"""
        df = pd.DataFrame({
            'activity_id': [],
            'molecule_chembl_id': []
        })
        activities = list()
        with pytest.raises(ValueError, match="DataFrame with activities should not be empty"):
            add_pchembl_values(df, activities)

    def test_add_pchembl_values_mismatched_lengths(self, sample_activities):
        """Test that DataFrame and activities list can have different lengths"""
        df = pd.DataFrame({
            'activity_id': [10001, 10002],
            'molecule_chembl_id': ['CHEMBL7000', 'CHEMBL7001']
        })
        with pytest.raises(ValueError,
                           match=(r"Number of activities provided \(3\) does not match "
                                  r"number of activities DataFrame \(2\)")):
            add_pchembl_values(df, sample_activities)


class TestSaveActivitiesInDataframe:
    """Tests for converting activities list to DataFrame with SMILES and pChEMBL values"""

    @staticmethod
    def mock_smiles(df):
        df = df.copy()
        df['canonical_smiles'] = ['O=C(O)c1ccc(O)cc1', 'O=C(O)/C=C/c1ccc(O)c(O)c1',
                                  'O=[N+]([O-])c1ccc(O)c(O)c1', None]
        return df

    @patch('src.utils.mol_activity_data_utils.attach_smiles')
    def test_save_activities_with_stats_tracking(self, mock_attach_smiles, sample_activities, sample_activities_ex_2):
        """Test that statistics are correctly tracked through the pipeline"""

        mock_attach_smiles.side_effect = TestSaveActivitiesInDataframe.mock_smiles
        all_activities = sample_activities + sample_activities_ex_2
        stats = ConversionStatistics()
        result = save_activities_in_dataframe(all_activities, stats)
        assert stats.unknown_units['not_standard_unit'] == 1
        assert len(result) == 4
        assert result['pchembl_value'].notna().sum() == 3
        mock_attach_smiles.assert_called_once()

    def test_save_activities_empty_list(self):
        """Test that empty activities list raises ValueError"""
        with pytest.raises(ValueError, match="No activities found"):
            save_activities_in_dataframe([])


class TestRetrieveAssayInfo:
    """Tests for retrieving assay information from ChEMBL"""

    @patch('src.utils.mol_activity_data_utils.new_client')
    def test_retrieve_assay_info(self, mock_client, sample_assay_info):
        """Test retrieve_assay_info with mocking, NaN assay IDs are filtered out"""
        df = pd.DataFrame({
            'assay_chembl_id': ['CHEMBL001', None, 'CHEMBL002', None, 'CHEMBL002'],
            'activity_id': [1, 2, 3, 4, 5]
        })
        mock_assay = Mock()
        mock_only = Mock()
        mock_only.only.return_value = sample_assay_info[:2]
        mock_assay.filter.return_value = mock_only
        mock_client.assay = mock_assay
        result = retrieve_assay_info(df)
        call_args = mock_assay.filter.call_args[1]
        assay_ids = call_args['assay_chembl_id__in']
        assert len(assay_ids) == 2
        assert set(assay_ids) == {'CHEMBL001', 'CHEMBL002'}
        assert None not in assay_ids
        assert set(call_args['assay_chembl_id__in']) == {'CHEMBL001', 'CHEMBL002'}
        mock_only.only.assert_called_once_with(['assay_chembl_id', 'assay_cell_type', 'bao_label'])
        assert isinstance(result, list)
        assert len(result) == 2
        assert result == sample_assay_info[:2]

    @patch('src.utils.mol_activity_data_utils.new_client')
    def test_retrieve_assay_info_returns_correct_structure(self, mock_client, sample_assay_info):
        """Test that returned data has correct structure"""
        df = pd.DataFrame({
            'assay_chembl_id': ['CHEMBL001', 'CHEMBL002'],
            'activity_id': [1, 2]
        })
        mock_assay = Mock()
        mock_only = Mock()
        mock_only.only.return_value = sample_assay_info
        mock_assay.filter.return_value = mock_only
        mock_client.assay = mock_assay
        result = retrieve_assay_info(df)
        for assay_info in result:
            assert 'assay_chembl_id' in assay_info
            assert 'assay_cell_type' in assay_info or assay_info.get('assay_cell_type') is None
            assert 'bao_label' in assay_info

    def test_retrieve_assay_info_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        df = pd.DataFrame({'assay_chembl_id': [],
                           'activity_id': []})

        with pytest.raises(ValueError, match="Cannot retrieve assay information from empty activities dataframe"):
            retrieve_assay_info(df)


@pytest.mark.parametrize("bao_label,cell_type,expected", [
    ("single protein format", None, "biochemical"),
    ("cell-based format", "HEK293", "cellular"),
    ("organism-based format", None, "organism"),
    ("protein format", None, "biochemical"),
    ("assay format", "CHO", "cellular"),
    ("assay format", None, None),
])
def test_assay_type_auxiliary(bao_label, cell_type, expected):
    """Test assay type determination from BAO label"""
    assay_info = {
        "assay_chembl_id": "CHEMBL001",
        "bao_label": bao_label,
        "assay_cell_type": cell_type
    }
    result = determine_assay_type_auxiliary(assay_info)
    assert result == expected


class TestCreateCertainActivityMapper:
    def test_create_mapper_empty_list_raises_error(self):
        """Test that empty assay info list raises ValueError"""
        with pytest.raises(ValueError, match="No assay information provided"):
            create_certain_activity_mapper([])

    def test_create_mapper_basic(self, sample_assay_info):
        """Test basic mapping of assay IDs to context types"""
        result = create_certain_activity_mapper(sample_assay_info)

        assert isinstance(result, dict)
        assert len(result) > 0
        assert result.get('CHEMBL001') == 'biochemical'
        assert result.get('CHEMBL002') == 'cellular'

    def test_create_mapper_optimization_avoids_reprocessing(self):
        """Test that assay with same IDs are processed only one time (in general just in case condition as
         duplicates are to be removed before API call)"""
        assay_info = [
            {
                'assay_chembl_id': 'CHEMBL001',
                'bao_label': 'single protein format',
                'assay_cell_type': None
            },
            {
                'assay_chembl_id': 'CHEMBL001',
                'bao_label': 'single protein format',
                'assay_cell_type': None
            },
        ]
        result = create_certain_activity_mapper(assay_info)

        assert len(result) == 1
        assert result['CHEMBL001'] == 'biochemical'


class TestGenerateExactAssayType:
    """Tests for adding context column based on exact assay metadata"""

    @patch('src.utils.mol_activity_data_utils.create_certain_activity_mapper')
    @patch('src.utils.mol_activity_data_utils.retrieve_assay_info')
    def test_generate_exact_assay_type_basic(self, mock_retrieve, mock_mapper, sample_assay_info):
        """Test basic context assignment from assay metadata"""
        df = pd.DataFrame({
            'activity_id': [1, 2, 3],
            'assay_chembl_id': ['CHEMBL001', 'CHEMBL002', 'CHEMBL001'],
            'molecule_chembl_id': ['MOL1', 'MOL2', 'MOL3']
        })

        mock_retrieve.return_value = sample_assay_info
        mock_mapper.return_value = {
            'CHEMBL001': 'biochemical',
            'CHEMBL002': 'cellular'
        }

        result = generate_exact_assay_type(df)
        assert 'context' in result.columns
        assert result.loc[0, 'context'] == 'biochemical'
        assert result.loc[1, 'context'] == 'cellular'
        assert result.loc[2, 'context'] == 'biochemical'
        assert result['context'].notna().sum() == 3
        mock_retrieve.assert_called_once()
        mock_mapper.assert_called_once_with(sample_assay_info)
        assert 'context' not in df.columns

    def test_generate_exact_assay_type_empty_dataframe(self):
        """Test that empty DataFrame raises ValueError"""
        df = pd.DataFrame({
            'activity_id': [],
            'assay_chembl_id': []
        })
        with pytest.raises(ValueError, match="Cannot retrieve assay information from empty activities dataframe"):
            generate_exact_assay_type(df)

    @patch('src.utils.mol_activity_data_utils.create_certain_activity_mapper')
    @patch('src.utils.mol_activity_data_utils.retrieve_assay_info')
    def test_generate_exact_assay_type_partial_mapping(self, mock_retrieve, mock_mapper):
        """Test handling when some assays don't have context mapping"""

        df = pd.DataFrame({
            'activity_id': [1, 2, 3, 4],
            'assay_chembl_id': ['CHEMBL001', 'CHEMBL002', 'CHEMBL003', 'CHEMBL001'],
            'molecule_chembl_id': ['MOL1', 'MOL2', 'MOL3', 'MOL4']
        })
        mock_retrieve.return_value = [
            {'assay_chembl_id': 'CHEMBL001', 'bao_label': 'single protein format'},
            {'assay_chembl_id': 'CHEMBL002', 'bao_label': None},
            {'assay_chembl_id': 'CHEMBL003', 'bao_label': 'assay format'}
        ]
        mock_mapper.return_value = {
            'CHEMBL001': 'biochemical',
            'CHEMBL002': None,
        }
        result = generate_exact_assay_type(df)
        assert 'context' in result.columns
        assert result.loc[0, 'context'] == 'biochemical'
        assert pd.isna(result.loc[1, 'context'])
        assert pd.isna(result.loc[2, 'context'])
        assert result.loc[3, 'context'] == 'biochemical'
        assert result['context'].notna().sum() == 2


class TestGenerateApproxAssayType:
    """Tests for inferring missing assay contexts using heuristics"""

    def test_generate_approx_assay_type_empty_dataframe(self):
        """Test that empty DataFrame raises ValueError"""
        df = pd.DataFrame({
            'activity_id': [],
            'standard_type': [],
            'assay_type': [],
            'context': []
        })
        with pytest.raises(ValueError, match="Cannot retrieve assay information from empty activities dataframe"):
            generate_approx_assay_type_for_row(df)

    def test_generate_approx_assay_type_basic_heuristics(self):
        """Test all three heuristic rules for context inference"""
        df = pd.DataFrame({
            'activity_id': [1, 2, 3, 4, 5],
            'standard_type': ['Ki', 'Kd', 'EC50', 'IC50', 'IC50'],
            'assay_type': ['B', 'B', 'F', 'B', 'F'],
            'context': [None, None, None, None, None]
        })
        result = generate_approx_assay_type_for_row(df)

        assert result.loc[0, 'context'] == 'biochemical'
        assert result.loc[1, 'context'] == 'biochemical'
        assert result.loc[2, 'context'] == 'cellular'
        assert result.loc[3, 'context'] == 'biochemical'
        assert pd.isna(result.loc[4, 'context'])
        assert result['context'].notna().sum() == 4
        assert 'context' in df.columns
        assert df['context'].isna().sum() == 5

    def test_generate_approx_assay_type_preserves_existing_contexts(self):
        """Test that existing contexts are NOT overwritten by heuristics"""
        df = pd.DataFrame({
            'activity_id': [1, 2, 3, 4],
            'standard_type': ['Ki', 'Ki', 'EC50', 'IC50'],
            'assay_type': ['B', 'B', 'F', 'B'],
            'context': ['organism', None, 'biochemical', None]
        })
        result = generate_approx_assay_type_for_row(df)

        assert result.loc[0, 'context'] == 'organism'
        assert result.loc[1, 'context'] == 'biochemical'
        assert result.loc[2, 'context'] == 'biochemical'
        assert result.loc[3, 'context'] == 'biochemical'
        assert result['context'].notna().sum() == 4


class TestActivityClassification:
    """Tests for active/inactive classification"""

    def test_activity_status_biochemical(self):
        """Test classification for biochemical assays"""
        df = pd.DataFrame({
            "pchembl_value": [7.0, 5.5, 8.0],
            "context": ["biochemical", "biochemical", "biochemical"],
            "relation": ["=", "=", "="]
        })
        result = retrieve_activity_status(df)

        assert result.loc[0, "is_active"] == True
        assert result.loc[1, "is_active"] == False
        assert result.loc[2, "is_active"] == True

    def test_activity_status_cellular(self):
        """Test classification for cellular assays"""
        df = pd.DataFrame({
            "pchembl_value": [5.5, 4.5, 6.0],
            "context": ["cellular", "cellular", "cellular"],
            "relation": ["=", "=", "="]
        })
        result = retrieve_activity_status(df)

        assert result.loc[0, "is_active"] == True
        assert result.loc[1, "is_active"] == False
        assert result.loc[2, "is_active"] == True

    def test_activity_status_relation_filter(self):
        """Test that only certain relations are considered active"""
        df = pd.DataFrame({
            "pchembl_value": [7.0, 7.0, 7.0],
            "context": ["biochemical", "biochemical", "biochemical"],
            "relation": ["=", ">", "~"]
        })
        result = retrieve_activity_status(df)

        assert result.loc[0, "is_active"] == True
        assert result.loc[1, "is_active"] == False
        assert result.loc[2, "is_active"] == True


class TestRemoveDuplicateActivities:
    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError"""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="Activities dataframe should not be empty"):
            remove_duplicate_activities(df)

    def test_missing_columns_raises_error(self):
        """Test that missing required columns raises ValueError"""
        df = pd.DataFrame({
            'molecule_chembl_id': ['CHEMBL1'],
            'target_chembl_id': ['TARGET1']
        })
        with pytest.raises(ValueError, match="Column context not in dataframe columns during deduplication process"):
            remove_duplicate_activities(df)

    def test_keeps_highest_pchembl_value(self):
        """Test that for duplicate entries, highest pChEMBL value is kept"""
        df = pd.DataFrame({
            'molecule_chembl_id': ['CHEMBL1', 'CHEMBL1', 'CHEMBL2', 'CHEMBL2'],
            'target_chembl_id': ['TARGET1', 'TARGET1', 'TARGET2', 'TARGET2'],
            'context': ['biochemical', 'biochemical', 'cellular', 'cellular'],
            'pchembl_value': [6.5, 7.2, 5.0, 4.5],
            'activity_id': [1, 2, 3, 4]
        })

        result = remove_duplicate_activities(df)

        assert len(result) == 2
        chembl1_row = result[result['molecule_chembl_id'] == 'CHEMBL1'].iloc[0]
        assert chembl1_row['pchembl_value'] == 7.2
        assert chembl1_row['activity_id'] == 2
        chembl2_row = result[result['molecule_chembl_id'] == 'CHEMBL2'].iloc[0]
        assert chembl2_row['pchembl_value'] == 5.0
        assert chembl2_row['activity_id'] == 3

    def test_handles_nan_contexts(self):
        """Test that NaN contexts are properly grouped and deduplicated"""
        df = pd.DataFrame({
            'molecule_chembl_id': ['CHEMBL1', 'CHEMBL1', 'CHEMBL1'],
            'target_chembl_id': ['TARGET1', 'TARGET1', 'TARGET1'],
            'context': [None, None, 'biochemical'],
            'pchembl_value': [6.0, 7.0, 5.5],
            'activity_id': [1, 2, 3]
        })

        result = remove_duplicate_activities(df)
        assert len(result) == 2
        nan_rows = result[result['context'].isna()]
        assert len(nan_rows) == 1
        assert nan_rows.iloc[0]['pchembl_value'] == 7.0
        assert nan_rows.iloc[0]['activity_id'] == 2
        bio_rows = result[result['context'] == 'biochemical']
        assert len(bio_rows) == 1
        assert bio_rows.iloc[0]['pchembl_value'] == 5.5


class TestIntegration:
    """Integration tests for complete workflows"""

    @patch("src.utils.mol_activity_data_utils.new_client")
    def test_complete_pipeline_mock(self, mock_client, sample_targets,
                                    sample_activities, sample_assay_info,
                                    sample_molecule_structures):
        """Test complete pipeline with mocked ChEMBL calls"""
        mock_target = Mock()
        mock_target.search.return_value = sample_targets

        mock_activity = Mock()
        mock_filter = Mock()
        mock_filter.only.side_effect = [
            [sample_activities[0]],
            [sample_activities[1]],
            [sample_activities[2]]
        ]
        mock_activity.filter.return_value = mock_filter

        mock_molecule = Mock()
        mock_mol_filter = Mock()
        mock_mol_filter.only.return_value = sample_molecule_structures
        mock_molecule.filter.return_value = mock_mol_filter

        mock_assay = Mock()
        mock_assay_filter = Mock()
        mock_assay_filter.only.return_value = sample_assay_info
        mock_assay.filter.return_value = mock_assay_filter

        mock_client.target = mock_target
        mock_client.activity = mock_activity
        mock_client.molecule = mock_molecule
        mock_client.assay = mock_assay

        activities_df, targets_df = generate_complete_activity_dataframe("CDK2")

        assert isinstance(activities_df, pd.DataFrame)
        assert len(activities_df) == 3
        expected_columns = [
            "molecule_chembl_id", "activity_id", "assay_chembl_id",
            "pchembl_value", "context", "canonical_smiles", "is_active", "target_chembl_id"
        ]
        assert len(activities_df.columns) == 8
        assert all(col in activities_df.columns for col in expected_columns)
        assert activities_df["canonical_smiles"].notna().any()
        assert "O=C(O)c1ccc(O)cc1" in activities_df["canonical_smiles"].values
        assert "O=C(O)/C=C/c1ccc(O)c(O)c1" in activities_df["canonical_smiles"].values
        assert "O=[N+]([O-])c1ccc(O)c(O)c1" in activities_df["canonical_smiles"].values
        assert "CHEMBL7000" in activities_df["molecule_chembl_id"].values
        assert "CHEMBL7001" in activities_df["molecule_chembl_id"].values
        assert "CHEMBL7002" in activities_df["molecule_chembl_id"].values
        assert isinstance(targets_df, pd.DataFrame)
        assert "target_chembl_id" in targets_df.columns
        assert "pref_name" in targets_df.columns
        assert "organism" in targets_df.columns
        assert "target_type" in targets_df.columns

    @patch("src.utils.mol_activity_data_utils.new_client")
    def test_complete_pipeline_no_targets_found(self, mock_client):
        """Test pipeline raises ValueError when no targets found"""
        mock_target = Mock()
        mock_target.search.return_value = []
        mock_client.target = mock_target

        with pytest.raises(ValueError, match="No targets found for query: 'NONEXISTENT'"):
            generate_complete_activity_dataframe("NONEXISTENT")

    @patch("src.utils.mol_activity_data_utils.new_client")
    def test_complete_pipeline_no_activities_found(self, mock_client, sample_targets):
        """Test pipeline raises ValueError when targets exist but no activities found"""
        mock_target = Mock()
        mock_target.search.return_value = sample_targets

        mock_activity = Mock()
        mock_filter = Mock()
        mock_filter.only.return_value = []
        mock_activity.filter.return_value = mock_filter

        mock_client.target = mock_target
        mock_client.activity = mock_activity

        with pytest.raises(ValueError, match="No activities found"):
            generate_complete_activity_dataframe("CDK2")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
