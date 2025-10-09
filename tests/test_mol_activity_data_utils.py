import pandas as pd
import pytest
from collections import defaultdict
from unittest.mock import patch, Mock
from src.utils.mol_activity_data_utils import ConversionStatistics, find_targets, get_activities_for_target, \
    combine_activities_for_targets, convert_standard_units_to_pchembl


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
            'molecule_chembl_id': 'CHEMBL68920',
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
            'molecule_chembl_id': 'CHEMBL68920',
            'pchembl_value': '6.52',
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
            'molecule_chembl_id': 'CHEMBL68920',
            'pchembl_value': '5.11',
            'relation': '=',
            'standard_type': 'IC50',
            'standard_units': 'nM',
            'standard_value': '7820.0',
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
            'pchembl_value': '6.77',
            'relation': '=',
            'standard_type': 'IC50',
            'standard_units': 'nM',
            'standard_value': '170.0',
            'target_chembl_id': ' CHEMBL2001',
            'type': 'IC50',
            'units': 'uM',
            'value': '0.17'
        }]


class TestChEMBLAPICallsToFetchTargetsAndActivities:
    """Tests for functions that make ChEMBL API calls (with mocking)"""

    @patch('src.utils.mol_activity_data_utils.new_client')
    def test_find_targets_with_limit(self, mock_client, sample_targets):
        """Test target search with mocked API"""
        mock_target = Mock()
        mock_target.search.return_value = sample_targets
        mock_client.target = mock_target
        result = find_targets("EGFR", organism="Homo sapiens", limit=2)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "target_chembl_id" in result.columns
        assert "pref_name" in result.columns
        assert "organism" in result.columns
        assert "target_type" in result.columns
        assert all(result["organism"] == "Homo sapiens")
        assert result.iloc[0]["target_chembl_id"] == "CHEMBL100"
        assert result.iloc[1]["target_chembl_id"] == "CHEMBL200"

        mock_target.search.assert_called_once_with("EGFR")

    @patch('src.utils.mol_activity_data_utils.new_client')
    def test_find_targets_no_limit(self, mock_client, sample_targets):
        """Test target search without limit returns all matching targets (3 targets returned and 1 excluded)"""
        mock_target = Mock()
        mock_target.search.return_value = sample_targets
        mock_client.target = mock_target
        result = find_targets("EGFR", organism="Homo sapiens", limit=None)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])