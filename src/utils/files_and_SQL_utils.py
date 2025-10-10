import logging

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd


Base = declarative_base()
class User(Base):
    __tablename__ = 'activities'
    activity_id = Column(Integer, primary_key=True)
    molecule_chembl_id = Column(String(100), nullable=False)
    assay_chembl_id = Column(String(100), nullable=True)
    pchembl_value = Column(Float, nullable=False)
    context = Column(String(50), nullable=True)
    canonical_smiles = Column(String(500), nullable=False)
    is_active = Boolean()


def save_activities_to_db(df: pd.DataFrame, target: str):
    engine = create_engine(f"sqlite:///{target}_database.db", echo=False)
    dtype_mapping = {
        'activity_id': Integer,
        'molecule_chembl_id': String(100),
        'assay_chembl_id': String(100),
        'pchembl_value': Float,
        'context': String(50),
        'canonical_smiles': String(500),
        'is_active': Boolean,
    }
    df.to_sql(
        name="activities",
        dtype=dtype_mapping,
        con=engine,
        if_exists="replace",
        index=False,
        method='multi',
        chunksize=1000
    )

