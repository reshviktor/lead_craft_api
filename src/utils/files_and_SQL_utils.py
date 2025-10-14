"""
SQLAlchemy Database Models and operations for Molecular Activity Data

This module defines the database schema using SQLAlchemy ORM for storing
molecular activity data from ChEMBL with similarity search capabilities
as well as perform  high-level database operations.
"""

import logging
import pandas as pd
from pathlib import Path
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from datetime import datetime
from typing import Optional
from sqlalchemy import String, Integer, Float, Boolean, DateTime, ForeignKey, Index, func, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from src.utils.mol_activity_data_utils import generate_complete_activity_dataframe
from src.utils.similarity_data_utils import generate_similarity_column, similarity_filter

logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

class Target(Base):
    """
    Main targets table storing user-facing target names.
    One target can map to multiple ChEMBL subtargets.
    """
    __tablename__ = "targets"

    target_inner_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    target_name: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    created_or_updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    subtargets: Mapped[list["Subtarget"]] = relationship(
        "Subtarget",
        back_populates="target",
        cascade="all, delete-orphan",
    )


class Subtarget(Base):
    """
    ChEMBL subtargets table storing individual ChEMBL target metadata.
    """
    __tablename__ = "subtargets"

    target_chembl_id: Mapped[str] = mapped_column(String, primary_key=True)
    target_inner_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("targets.target_inner_id", ondelete="CASCADE"),
        nullable=False
    )
    pref_name: Mapped[Optional[str]] = mapped_column(String)
    organism: Mapped[Optional[str]] = mapped_column(String)
    target_type: Mapped[Optional[str]] = mapped_column(String)

    target: Mapped["Target"] = relationship("Target", back_populates="subtargets")
    activities: Mapped[list["Activity"]] = relationship(
        "Activity",
        back_populates="subtarget",
        cascade="all, delete-orphan",
    )
    __table_args__ = (Index("idx_subtargets_target_inner_id", "target_inner_id"),)


class Activity(Base):
    """
    Activities table storing all molecular activity data retrieved/calculated from ChEMBL.
    """
    __tablename__ = "activities"

    activity_id: Mapped[str] = mapped_column(String, primary_key=True)
    molecule_chembl_id: Mapped[str] = mapped_column(String, nullable=False)
    target_chembl_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("subtargets.target_chembl_id", ondelete="CASCADE"),
        nullable=False
    )
    assay_chembl_id: Mapped[Optional[str]] = mapped_column(String)
    pchembl_value: Mapped[Optional[float]] = mapped_column(Float)
    context: Mapped[Optional[str]] = mapped_column(String)
    canonical_smiles: Mapped[Optional[str]] = mapped_column(String)
    is_active: Mapped[Optional[bool]] = mapped_column(Boolean)
    subtarget: Mapped["Subtarget"] = relationship("Subtarget", back_populates="activities")

    __table_args__ = (
        Index('idx_activities_target', 'target_chembl_id'),
        Index('idx_activities_molecule', 'molecule_chembl_id'),
        Index('idx_activities_target_molecule', 'target_chembl_id', 'molecule_chembl_id'),
    )

class MolecularActivityDatabase:
    """
    Database service for managing molecular activity data with SQLAlchemy ORM.
    Handles caching of ChEMBL data and similarity-based searches.
    """

    def __init__(self, db_path: str = "molecular_activities.db"):
        """
        Initialize database connection and create tables.
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            connect_args={"check_same_thread": False}
        )
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autoflush=False,
            expire_on_commit=False)

        Base.metadata.create_all(bind=self.engine)
        logger.info(f"Database initialized at {self.db_path}")

    @contextmanager
    def get_session(self):
        """Context manager for database sessions with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session error: {e}")
            raise
        finally:
            session.close()

    def target_exists(self, target_name: str) -> bool:
        """
        Check if target already exists in database.
        Args:
            target_name: User-defined target name
        Returns:
            True if target exists, False otherwise
        """
        with self.get_session() as session:
            stmt = select(Target.target_inner_id).where(Target.target_name == target_name).limit(1)
            return session.execute(stmt).first() is not None

    def get_target(self, target_name: str) -> Optional[Target]:
        """
        Retrieve target object by name.
        Args:
            target_name: User-defined target name
        Returns:
            Target object or None if not found
        """
        with self.get_session() as session:
            stmt = select(Target).where(Target.target_name == target_name)
            return session.execute(stmt).scalar_one_or_none()

    def get_target_activities(self, target_name: str) -> Optional[pd.DataFrame]:
        """
        Retrieve all activities for a target as a pandas DataFrame.
        Args:
            target_name: User-defined target name
        Returns:
            DataFrame with activity data or None if target not found
        """
        with self.get_session() as session:
            stmt = (
                select(Activity)
                .join(Subtarget, Activity.target_chembl_id == Subtarget.target_chembl_id)
                .join(Target, Subtarget.target_inner_id == Target.target_inner_id)
                .where(Target.target_name == target_name)
            )

            results = session.execute(stmt).scalars().unique().all()

            if not results:
                logger.warning(f"No activities found for target '{target_name}'")
                return None
            data = [
                {
                    "activity_id": activity.activity_id,
                    "molecule_chembl_id": activity.molecule_chembl_id,
                    "target_chembl_id": activity.target_chembl_id,
                    "assay_chembl_id": activity.assay_chembl_id,
                    "pchembl_value": activity.pchembl_value,
                    "context": activity.context,
                    "canonical_smiles": activity.canonical_smiles,
                    "is_active": activity.is_active,
                }
                for activity in results
            ]
            df = pd.DataFrame.from_records(data)
            logger.info(f"Retrieved {len(df)} activities for target '{target_name}'")
            return df

    def save_target_data(
            self,
            target_name: str,
            activities_df: pd.DataFrame,
            targets_df: pd.DataFrame
    ) -> None:
        """
        Save new target and its activities to database.
        Args:
            target_name: User-defined target name
            activities_df: DataFrame with activity data
            targets_df: DataFrame with ChEMBL target metadata
        """
        with self.get_session() as session:
            # Get or create target
            target = session.execute(
                select(Target).where(Target.target_name == target_name)
            ).scalar_one_or_none()

            if not target:
                target = Target(target_name=target_name)
                session.add(target)
                session.flush()
                logger.info(f"Created new target '{target_name}' with ID {target.target_inner_id}")
            else:
                logger.info(f"Target '{target_name}' already exists, updating data")

            for _, row in targets_df.iterrows():
                subtarget = session.execute(
                    select(Subtarget).where(
                        Subtarget.target_chembl_id == row['target_chembl_id']
                    )
                ).scalar_one_or_none()

                if not subtarget:
                    subtarget = Subtarget(
                        target_chembl_id=row['target_chembl_id'],
                        target_inner_id=target.target_inner_id,
                        pref_name=row.get('pref_name'),
                        organism=row.get('organism'),
                        target_type=row.get('target_type')
                    )
                    session.add(subtarget)

            session.flush()

            existing_activity_ids = set(
                session.execute(
                    select(Activity.activity_id)
                    .join(Subtarget)
                    .where(Subtarget.target_inner_id == target.target_inner_id)
                ).scalars().all()
            )

            new_activities = 0
            for _, row in activities_df.iterrows():
                if row['activity_id'] not in existing_activity_ids:
                    session.add(Activity(
                        activity_id=row["activity_id"],
                        molecule_chembl_id=row["molecule_chembl_id"],
                        target_chembl_id=row["target_chembl_id"],
                        assay_chembl_id=row.get("assay_chembl_id"),
                        pchembl_value=row.get("pchembl_value"),
                        context=row.get("context"),
                        canonical_smiles=row.get("canonical_smiles"),
                        is_active=row.get("is_active"),
                    ))
                    new_activities += 1

            session.flush()

            logger.info(
                f"Saved {len(targets_df)} subtargets and "
                f"{new_activities} new activities for target '{target_name}'"
            )

    def get_database_stats(self) -> dict:
        """
        Get statistics about the database contents.

        Returns:
            Dictionary with counts of targets, subtargets, and activities
        """
        with self.get_session() as session:
            stats = {
                'targets': session.execute(select(func.count(Target.target_inner_id))).scalar(),
                'subtargets': session.execute(select(func.count(Subtarget.target_chembl_id))).scalar(),
                'activities': session.execute(select(func.count(Activity.activity_id))).scalar()
            }
            return stats

    def delete_target(self, target_name: str) -> bool:
        """
        Delete a target and all associated data (cascades to subtargets and activities).

        Args:
            target_name: User-defined target name

        Returns:
            True if deleted, False if target not found
        """
        with self.get_session() as session:
            target = session.execute(
                select(Target).where(Target.target_name == target_name)
            ).scalar_one_or_none()

            if not target:
                logger.warning(f"Target '{target_name}' not found for deletion")
                return False

            session.delete(target)
            logger.info(f"Deleted target '{target_name}' and all associated data")
            return True

    def process_query(
            self,
            target_name: str,
            query_smiles: str,
            min_similarity: float = 0.8,
            max_molecules: int = 10,
            organism: Optional[str] = "Homo sapiens",
            force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Main workflow: check database or fetch from ChEMBL, then filter by similarity.

        Args:
            target_name: Target name to search
            query_smiles: SMILES string for similarity search
            min_similarity: Minimum Tanimoto similarity threshold (default: 0.8)
            max_molecules: Maximum number of similar molecules to return (default: 10)
            organism: Organism filter for ChEMBL search (default: "Homo sapiens")
            force_refresh: If True, fetch fresh data from ChEMBL even if cached

        Returns:
            DataFrame with filtered activities similar to query molecule
        """

        if self.target_exists(target_name) and not force_refresh:
            logger.info(f"Target '{target_name}' found in database, using cached data")
            activities_df = self.get_target_activities(target_name)
            if activities_df is None or activities_df.empty:
                logger.info("No cached activities; fetching fresh from ChEMBL")
                activities_df, targets_df = generate_complete_activity_dataframe(
                    query=target_name, organism=organism
                )
                self.save_target_data(target_name, activities_df, targets_df)
        else:
            if force_refresh:
                logger.info(f"Force refresh requested for target '{target_name}'")
            else:
                logger.info(f"Target '{target_name}' not found in database")

            logger.info(f"Fetching fresh data from ChEMBL...")
            activities_df, targets_df = generate_complete_activity_dataframe(
                query=target_name,
                organism=organism
            )
            self.save_target_data(target_name, activities_df, targets_df)

        logger.info(f"Calculating similarity to query molecule: {query_smiles}")
        activities_with_sim = generate_similarity_column(
            activities_df,
            query_smiles=query_smiles,
            smiles_col="canonical_smiles"
        )

        filtered_df = similarity_filter(
            activities_with_sim,
            min_similarity=min_similarity,
            max_molecules=max_molecules
        )

        logger.info(
            f"Found {len(filtered_df)} activities from "
            f"{filtered_df['molecule_chembl_id'].nunique()} molecules "
            f"with similarity >= {min_similarity}"
        )

        return filtered_df

    def list_all_targets(self) -> list[dict]:
        """
        List all targets in the database with their metadata.
        Uses SQL aggregation to avoid N+1 query problem.

        Returns:
            List of dictionaries containing target information
        """
        with self.get_session() as session:
            rows = session.execute(
                select(
                    Target.target_inner_id,
                    Target.target_name,
                    Target.created_or_updated_at,
                    func.count(Subtarget.target_chembl_id).label("num_subtargets"),
                )
                .outerjoin(Subtarget, Target.target_inner_id == Subtarget.target_inner_id)
                .group_by(Target.target_inner_id, Target.target_name, Target.created_or_updated_at)
                .order_by(Target.target_inner_id)
            ).all()

            return [
                {
                    "target_inner_id": row.target_inner_id,
                    "target_name": row.target_name,
                    "created_or_updated_at": row.created_or_updated_at,
                    "num_subtargets": row.num_subtargets or 0,
                }
                for row in rows
            ]
