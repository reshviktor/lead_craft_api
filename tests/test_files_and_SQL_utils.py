"""
Comprehensive test suite for SQLAlchemy Database Models and Operations

Tests cover:
- Database initialization
- CRUD operations for targets, subtargets, and activities
- Query workflow with similarity filtering
- Edge cases and error handling
"""

import pytest
import pandas as pd
import tempfile
import os
import time
from unittest.mock import patch
from datetime import datetime
from sqlalchemy import select, func

from src.mol_activity.utils.files_and_SQL_utils import (
    Base,
    Target,
    MolecularActivityDatabase
)


@pytest.fixture
def temp_db_path():
    """Create a temporary database file for testing"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    for attempt in range(5):
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
            break
        except PermissionError:
            time.sleep(0.1)
            if attempt == 4:
                pass


@pytest.fixture
def db(temp_db_path):
    """Create a test database instance"""
    database = MolecularActivityDatabase(db_path=temp_db_path)
    yield database
    database.engine.dispose()


@pytest.fixture
def sample_targets_df():
    """Sample ChEMBL targets DataFrame"""
    return pd.DataFrame([
        {
            'target_chembl_id': 'CHEMBL123',
            'pref_name': 'Cyclin-dependent kinase 2',
            'organism': 'Homo sapiens',
            'target_type': 'SINGLE PROTEIN'
        },
        {
            'target_chembl_id': 'CHEMBL456',
            'pref_name': 'Cyclin-dependent kinase 2/cyclin A',
            'organism': 'Homo sapiens',
            'target_type': 'PROTEIN COMPLEX'
        }
    ])


@pytest.fixture
def sample_activities_df():
    """Sample ChEMBL activities DataFrame"""
    return pd.DataFrame([
        {
            'activity_id': 'ACT001',
            'molecule_chembl_id': 'CHEMBL1',
            'target_chembl_id': 'CHEMBL123',
            'assay_chembl_id': 'ASSAY001',
            'pchembl_value': 7.5,
            'context': 'biochemical',
            'canonical_smiles': 'CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5',
            'is_active': True
        },
        {
            'activity_id': 'ACT002',
            'molecule_chembl_id': 'CHEMBL2',
            'target_chembl_id': 'CHEMBL123',
            'assay_chembl_id': 'ASSAY002',
            'pchembl_value': 6.2,
            'context': 'cellular',
            'canonical_smiles': 'CC(OC1=C(O[C@@H]2[C@]34CCN(C)[C@@H]([C@@H]4C=C[C@@H]2OC(C)=O)C5)C3=C5C=C1)=O',
            'is_active': False
        },
        {
            'activity_id': 'ACT003',
            'molecule_chembl_id': 'CHEMBL3',
            'target_chembl_id': 'CHEMBL456',
            'assay_chembl_id': 'ASSAY003',
            'pchembl_value': 8.1,
            'context': 'biochemical',
            'canonical_smiles': 'CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O',
            'is_active': True
        }
    ])


class TestDatabaseInitialization:
    """Test database initialization and setup"""

    def test_database_creation(self, temp_db_path):
        """Test that database file is created"""
        db = MolecularActivityDatabase(db_path=temp_db_path)

        assert os.path.exists(temp_db_path)
        assert db.db_path == temp_db_path
        db.engine.dispose()

    def test_database_with_nested_path(self):
        """Test database creation with nested directory path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, "subdir", "test.db")
            db = MolecularActivityDatabase(db_path=nested_path)

            assert os.path.exists(nested_path)
            db.engine.dispose()

    def test_tables_created(self, db):
        """Test that all tables are created"""
        table_names = Base.metadata.tables.keys()

        assert 'targets' in table_names
        assert 'subtargets' in table_names
        assert 'activities' in table_names

    def test_session_creation(self, db):
        """Test that sessions can be created"""
        with db.get_session() as session:
            result = session.execute(select(func.count()).select_from(Target))

            assert session is not None
            assert result.scalar() == 0


class TestTargetOperations:
    """Test target-related database operations"""

    def test_target_exists_false_on_empty_db(self, db):
        """Test target_exists returns False when target doesn't exist"""
        assert db.target_exists("CDK2") is False

    def test_save_and_check_target_exists(self, db, sample_targets_df, sample_activities_df):
        """Test saving a target and checking it exists"""
        db.save_target_data("CDK2", sample_activities_df, sample_targets_df)

        assert db.target_exists("CDK2") is True

    def test_get_nonexistent_target(self, db):
        """Test getting a target that doesn't exist"""
        target = db.get_target("NONEXISTENT")

        assert target is None

    def test_get_existing_target(self, db, sample_targets_df, sample_activities_df):
        """Test retrieving an existing target"""
        db.save_target_data("CDK2", sample_activities_df, sample_targets_df)
        target = db.get_target("CDK2")

        assert target is not None
        assert target.target_name == "CDK2"
        assert target.target_inner_id is not None

    def test_target_created_timestamp(self, db, sample_targets_df, sample_activities_df):
        """Test that timestamp is set on target creation"""
        db.save_target_data("CDK2", sample_activities_df, sample_targets_df)
        target = db.get_target("CDK2")

        assert target.created_or_updated_at is not None
        assert isinstance(target.created_or_updated_at, datetime)

    def test_delete_existing_target(self, db, sample_targets_df, sample_activities_df):
        """Test deleting an existing target"""
        db.save_target_data("CDK2", sample_activities_df, sample_targets_df)
        result = db.delete_target("CDK2")

        assert result is True
        assert db.target_exists("CDK2") is False

    def test_delete_nonexistent_target(self, db):
        """Test deleting a target that doesn't exist"""
        result = db.delete_target("NONEXISTENT")

        assert result is False

    def test_cascade_delete_removes_subtargets_and_activities(
            self, db, sample_targets_df, sample_activities_df
    ):
        """Test that deleting target also deletes subtargets and activities"""
        db.save_target_data("CDK2", sample_activities_df, sample_targets_df)
        stats_before = db.get_database_stats()

        assert stats_before['subtargets'] > 0
        assert stats_before['activities'] > 0

        db.delete_target("CDK2")
        stats_after = db.get_database_stats()

        assert stats_after['targets'] == 0
        assert stats_after['subtargets'] == 0
        assert stats_after['activities'] == 0


class TestSubtargetOperations:
    """Test subtarget-related database operations"""

    def test_save_subtargets(self, db, sample_targets_df, sample_activities_df):
        """Test that subtargets are saved correctly"""
        db.save_target_data("CDK2", sample_activities_df, sample_targets_df)
        stats = db.get_database_stats()

        assert stats['subtargets'] == len(sample_targets_df)

    def test_subtarget_foreign_key(self, db, sample_targets_df, sample_activities_df):
        """Test that subtargets are linked to correct target"""
        db.save_target_data("CDK2", sample_activities_df, sample_targets_df)
        with db.get_session() as session:
            target = session.execute(
                select(Target).where(Target.target_name == "CDK2")
            ).scalar_one()

            assert len(target.subtargets) == len(sample_targets_df)
            for subtarget in target.subtargets:
                assert subtarget.target_inner_id == target.target_inner_id

    def test_duplicate_subtargets_not_added(self, db, sample_targets_df, sample_activities_df):
        """Test that duplicate subtargets are not added"""
        db.save_target_data("CDK2", sample_activities_df, sample_targets_df)
        db.save_target_data("CDK2", sample_activities_df, sample_targets_df)
        stats = db.get_database_stats()

        assert stats['subtargets'] == len(sample_targets_df)


class TestActivityOperations:
    """Test activity-related database operations"""

    def test_save_activities(self, db, sample_targets_df, sample_activities_df):
        """Test that activities are saved correctly"""
        db.save_target_data("CDK2", sample_activities_df, sample_targets_df)
        stats = db.get_database_stats()

        assert stats['activities'] == len(sample_activities_df)

    def test_get_target_activities(self, db, sample_targets_df, sample_activities_df):
        """Test retrieving activities for a target"""
        db.save_target_data("CDK2", sample_activities_df, sample_targets_df)
        activities_df = db.get_target_activities("CDK2")

        assert activities_df is not None
        assert len(activities_df) == len(sample_activities_df)
        assert 'activity_id' in activities_df.columns
        assert 'pchembl_value' in activities_df.columns

    def test_get_activities_nonexistent_target(self, db):
        """Test getting activities for nonexistent target"""
        activities_df = db.get_target_activities("NONEXISTENT")

        assert activities_df is None

    def test_duplicate_activities_not_added(self, db, sample_targets_df, sample_activities_df):
        """Test that duplicate activities are not added"""
        db.save_target_data("CDK2", sample_activities_df, sample_targets_df)
        db.save_target_data("CDK2", sample_activities_df, sample_targets_df)
        stats = db.get_database_stats()

        assert stats['activities'] == len(sample_activities_df)

    def test_activity_foreign_key(self, db, sample_targets_df, sample_activities_df):
        """Test that activities are linked to correct subtargets"""
        db.save_target_data("CDK2", sample_activities_df, sample_targets_df)
        activities_df = db.get_target_activities("CDK2")

        for _, activity in activities_df.iterrows():
            assert activity['target_chembl_id'] in sample_targets_df['target_chembl_id'].values

    def test_activities_with_optional_fields(self, db, sample_targets_df):
        """Test saving activities with None/null values"""
        activities_df = pd.DataFrame([
            {
                'activity_id': 'ACT_NULL',
                'molecule_chembl_id': 'CHEMBL999',
                'target_chembl_id': 'CHEMBL123',
                'assay_chembl_id': None,
                'pchembl_value': None,
                'context': None,
                'canonical_smiles': None,
                'is_active': None
            }
        ])
        db.save_target_data("TEST", activities_df, sample_targets_df)
        retrieved_df = db.get_target_activities("TEST")

        assert len(retrieved_df) == 1
        assert pd.isna(retrieved_df.iloc[0]['pchembl_value'])


class TestDatabaseStats:
    """Test database statistics functionality"""

    def test_empty_database_stats(self, db):
        """Test stats on empty database"""
        stats = db.get_database_stats()

        assert stats['targets'] == 0
        assert stats['subtargets'] == 0
        assert stats['activities'] == 0

    def test_stats_after_adding_data(self, db, sample_targets_df, sample_activities_df):
        """Test stats after adding data"""
        db.save_target_data("CDK2", sample_activities_df, sample_targets_df)
        stats = db.get_database_stats()

        assert stats['targets'] == 1
        assert stats['subtargets'] == len(sample_targets_df)
        assert stats['activities'] == len(sample_activities_df)

    def test_stats_multiple_targets(self, db, sample_targets_df, sample_activities_df):
        """Test stats with multiple targets"""
        db.save_target_data("CDK2", sample_activities_df, sample_targets_df)
        egfr_activities = sample_activities_df.copy()
        egfr_activities['activity_id'] = egfr_activities['activity_id'].str.replace('ACT', 'EGFR_ACT')
        db.save_target_data("EGFR", egfr_activities, sample_targets_df)
        stats = db.get_database_stats()

        assert stats['targets'] == 2
        assert stats['activities'] == len(sample_activities_df) + len(egfr_activities)


class TestListTargets:
    """Test listing all targets"""

    def test_list_empty_targets(self, db):
        """Test listing targets on empty database"""
        targets = db.list_all_targets()

        assert targets == []

    def test_list_single_target(self, db, sample_targets_df, sample_activities_df):
        """Test listing a single target"""
        db.save_target_data("CDK2", sample_activities_df, sample_targets_df)
        targets = db.list_all_targets()

        assert len(targets) == 1
        assert targets[0]['target_name'] == "CDK2"
        assert 'target_inner_id' in targets[0]
        assert 'created_or_updated_at' in targets[0]
        assert 'num_subtargets' in targets[0]

    def test_list_multiple_targets(self, db, sample_targets_df, sample_activities_df):
        """Test listing multiple targets"""
        db.save_target_data("CDK2", sample_activities_df, sample_targets_df)
        egfr_activities = sample_activities_df.copy()
        egfr_activities['activity_id'] = egfr_activities['activity_id'].str.replace('ACT', 'EGFR_ACT')
        db.save_target_data("EGFR", egfr_activities, sample_targets_df)
        targets = db.list_all_targets()
        target_names = [t['target_name'] for t in targets]

        assert len(targets) == 2
        assert "CDK2" in target_names
        assert "EGFR" in target_names


class TestProcessQuery:
    """Test the main process_query workflow"""

    @patch('src.mol_activity.utils.files_and_SQL_utils.generate_complete_activity_dataframe')
    @patch('src.mol_activity.utils.files_and_SQL_utils.generate_similarity_column')
    @patch('src.mol_activity.utils.files_and_SQL_utils.similarity_filter')
    def test_process_query_fetch_from_chembl(
            self, mock_filter, mock_similarity, mock_generate,
            db, sample_targets_df, sample_activities_df
    ):
        """Test process_query when target not in cache"""
        mock_generate.return_value = (sample_activities_df, sample_targets_df)
        mock_similarity.return_value = sample_activities_df.copy()
        mock_filter.return_value = sample_activities_df.head(1)
        result = db.process_query(
            target_name="CDK2",
            query_smiles="OC[C@H](O)[C@H]1OC(=O)C(O)=C1O",
            min_similarity=0.8,
            max_molecules=10
        )

        assert mock_generate.called
        assert mock_similarity.called
        assert mock_filter.called
        assert len(result) == 1
        assert db.target_exists("CDK2")

    @patch('src.mol_activity.utils.files_and_SQL_utils.generate_complete_activity_dataframe')
    @patch('src.mol_activity.utils.files_and_SQL_utils.generate_similarity_column')
    @patch('src.mol_activity.utils.files_and_SQL_utils.similarity_filter')
    def test_process_query_use_cache(
            self, mock_filter, mock_similarity, mock_generate,
            db, sample_targets_df, sample_activities_df
    ):
        """Test process_query when target is cached"""
        db.save_target_data("CDK2", sample_activities_df, sample_targets_df)
        mock_similarity.return_value = sample_activities_df.copy()
        mock_filter.return_value = sample_activities_df.head(1)
        result = db.process_query(
            target_name="CDK2",
            query_smiles="CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
            min_similarity=0.8,
            max_molecules=10
        )

        assert not mock_generate.called
        assert mock_similarity.called
        assert mock_filter.called
        assert len(result) == 1

    @patch('src.mol_activity.utils.files_and_SQL_utils.generate_complete_activity_dataframe')
    @patch('src.mol_activity.utils.files_and_SQL_utils.generate_similarity_column')
    @patch('src.mol_activity.utils.files_and_SQL_utils.similarity_filter')
    def test_process_query_force_refresh(
            self, mock_filter, mock_similarity, mock_generate,
            db, sample_targets_df, sample_activities_df
    ):
        """Test process_query with force_refresh=True"""
        db.save_target_data("CDK2", sample_activities_df, sample_targets_df)
        mock_generate.return_value = (sample_activities_df, sample_targets_df)
        mock_similarity.return_value = sample_activities_df.copy()
        mock_filter.return_value = sample_activities_df.head(1)
        result = db.process_query(
            target_name="CDK2",
            query_smiles="CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
            min_similarity=0.8,
            max_molecules=10,
            force_refresh=True
        )

        assert mock_generate.called
        assert mock_similarity.called
        assert mock_filter.called

    @patch('src.mol_activity.utils.files_and_SQL_utils.generate_complete_activity_dataframe')
    @patch('src.mol_activity.utils.files_and_SQL_utils.generate_similarity_column')
    @patch('src.mol_activity.utils.files_and_SQL_utils.similarity_filter')
    def test_process_query_empty_cache_refetch(
            self, mock_filter, mock_similarity, mock_generate,
            db, sample_targets_df, sample_activities_df
    ):
        """Test that empty cached activities triggers refetch"""
        empty_activities = sample_activities_df.iloc[0:0]
        db.save_target_data("CDK2", empty_activities, sample_targets_df)
        mock_generate.return_value = (sample_activities_df, sample_targets_df)
        mock_similarity.return_value = sample_activities_df.copy()
        mock_filter.return_value = sample_activities_df.head(1)
        result = db.process_query(
            target_name="CDK2",
            query_smiles="CC(OC1=C(O[C@@H]2[C@]34CCN(C)[C@@H]([C@@H]4C=C[C@@H]2OC(C)=O)C5)C3=C5C=C1)=O"
        )

        assert mock_generate.called


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_session_rollback_on_error(self, db):
        """Test that session rolls back on error"""
        with pytest.raises(Exception):
            with db.get_session() as session:
                target = Target(target_name="TEST")
                session.add(target)
                session.flush()
                duplicate = Target(target_name="TEST")
                session.add(duplicate)
                session.flush()

        assert not db.target_exists("TEST")

    def test_save_with_missing_dataframe_columns(self, db):
        """Test behavior with DataFrames missing expected columns"""
        incomplete_targets = pd.DataFrame([
            {'target_chembl_id': 'CHEMBL123'}
        ])
        incomplete_activities = pd.DataFrame([
            {
                'activity_id': 'ACT001',
                'molecule_chembl_id': 'CHEMBL1',
                'target_chembl_id': 'CHEMBL123'
            }
        ])
        db.save_target_data("TEST", incomplete_activities, incomplete_targets)

        assert db.target_exists("TEST")


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_full_workflow_single_target(self, db, sample_targets_df, sample_activities_df):
        """Test complete workflow: save, retrieve, query, delete"""
        target_name = "CDK2"
        db.save_target_data(target_name, sample_activities_df, sample_targets_df)

        assert db.target_exists(target_name)

        stats = db.get_database_stats()

        assert stats['targets'] == 1
        assert stats['activities'] > 0

        targets = db.list_all_targets()

        assert len(targets) == 1
        assert targets[0]['target_name'] == target_name

        activities = db.get_target_activities(target_name)

        assert len(activities) == len(sample_activities_df)
        assert db.delete_target(target_name)
        assert not db.target_exists(target_name)

    def test_multiple_targets_isolation(self, db, sample_targets_df, sample_activities_df):
        """Test that multiple targets are properly isolated"""
        db.save_target_data("CDK2", sample_activities_df, sample_targets_df)
        egfr_activities = sample_activities_df.copy()
        egfr_activities['activity_id'] = egfr_activities['activity_id'].str.replace('ACT', 'EGFR_ACT')
        egfr_targets = sample_targets_df.copy()
        egfr_targets['target_chembl_id'] = (egfr_targets['target_chembl_id']
                                            .str.replace('CHEMBL', 'EGFR_CHEMBL'))
        egfr_activities['target_chembl_id'] = (egfr_activities['target_chembl_id']
                                               .str.replace('CHEMBL', 'EGFR_CHEMBL'))
        db.save_target_data("EGFR", egfr_activities, egfr_targets)

        assert db.target_exists("CDK2")
        assert db.target_exists("EGFR")

        db.delete_target("CDK2")

        assert not db.target_exists("CDK2")
        assert db.target_exists("EGFR")

        activities = db.get_target_activities("EGFR")

        assert activities is not None
        assert len(activities) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
