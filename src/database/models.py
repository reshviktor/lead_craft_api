"""
SQLAlchemy Database Models

This module defines the database schema using SQLAlchemy ORM for storing
molecular activity data from ChEMBL with similarity search capabilities.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Target(Base):
    """
    Main targets table storing user-facing target names.
    One target can map to multiple ChEMBL subtargets.
    """

    __tablename__ = "targets"

    target_inner_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    target_name: Mapped[str] = mapped_column(
        String, unique=True, nullable=False, index=True
    )
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
        nullable=False,
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
        nullable=False,
    )
    assay_chembl_id: Mapped[Optional[str]] = mapped_column(String)
    pchembl_value: Mapped[Optional[float]] = mapped_column(Float)
    context: Mapped[Optional[str]] = mapped_column(String)
    canonical_smiles: Mapped[Optional[str]] = mapped_column(String)
    is_active: Mapped[Optional[bool]] = mapped_column(Boolean)
    subtarget: Mapped["Subtarget"] = relationship(
        "Subtarget", back_populates="activities"
    )

    __table_args__ = (
        Index("idx_activities_target", "target_chembl_id"),
        Index("idx_activities_molecule", "molecule_chembl_id"),
        Index(
            "idx_activities_target_molecule", "target_chembl_id", "molecule_chembl_id"
        ),
    )