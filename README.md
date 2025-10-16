# LeadCraft API

### v0.1: Activity Retrieval, Caching & Similarity Search

**Version 0.1 provides a tool for bioactivity retrieval, local caching, and similarity-based compound discovery.**

“LeadCraft currently queries ChEMBL, caches results locally, and identifies structurally similar active compounds using
molecular fingerprints, forming the basis for drug optimization in drug discovery projects. See the roadmap below for
the planned next steps

---

## Key Features/Operations

### Data Retrieval & Processing

1. **Target Search**: Searches ChEMBL database for biological targets if it not cached
2. **Activity Retrieval**: Fetches/calculates bioactivity data, assay content, molecular structures
3. **Data Processing**: Calculates pChEMBL values and classifies activity, deduplicate measurements (activity-aware
   operation)
4. **Caching**: Stores data in SQLite for fast subsequent queries/reuse.
5. **Similarity Search**: Finds molecules similar to your query using Tanimoto similarity (MACCS keys)

---

## Installation

### Prerequisites

- Conda or Miniconda
- Python ≥3.10

### Setup

```
# Clone the repository
git clone https://github.com/reshviktor/lead_craft_api.git
cd lead_craft_api

# Create conda environment
conda env create -f environment.yml
conda activate lead_craft_api
```

---

## Example to run

### Python API

```python
from src.mol_activity.main import basic_query
from src.mol_activity.utils.logging_utils import setup_logging

# Configure logging (optional)
setup_logging(level="INFO")

# Query for CDK2 inhibitors similar to Roscovitine
results = basic_query(
    target="CDK2",
    smiles="CC[C@H](CO)Nc1nc(Nc2ccccc2)nc(NC(C)(C)C)n1",
    min_similarity=0.7,
    max_molecules=15,
    organism="Homo sapiens"
)

print(f"Found {len(results)} activities from {results['molecule_chembl_id'].nunique()} molecules")
print(f"Similarity range: {results['tanimoto_similarity'].min():.2f} - {results['tanimoto_similarity'].max():.2f}")
print(f"pChEMBL range: {results['pchembl_value'].min():.2f} - {results['pchembl_value'].max():.2f}")
```

**First run**: 5-60 minutes (fetches from ChEMBL + processes data)
**Subsequent runs**: <1-2 minutes (uses cached data)

### Jupyter Notebook Demo

You can find a Jupyter notebook demo with example to use at lead_craft_api/notebooks/LeadCraft_demo.ipynb

## Result DataFrame Columns

| Column                | Description                                                    |
|-----------------------|----------------------------------------------------------------|
| `activity_id`         | Unique ChEMBL activity identifier                              |
| `molecule_chembl_id`  | ChEMBL molecule identifier                                     |
| `target_chembl_id`    | ChEMBL target identifier                                       |
| `assay_chembl_id`     | ChEMBL assay identifier                                        |
| `pchembl_value`       | Negative log10 of molar IC50/Ki/Kd/EC50 (higher = more potent) |
| `context`             | Assay type: biochemical, cellular, or organism                 |
| `canonical_smiles`    | Standardized SMILES representation                             |
| `is_active`           | Boolean classification based on context-aware thresholds       |
| `tanimoto_similarity` | Similarity to query molecule (0-1)                             |

### Activity Classification Thresholds

| Context     | Active Threshold | Interpretation |
|-------------|------------------|----------------|
| Biochemical | pChEMBL ≥ 6.0    | ≤ 1 µM         |
| Cellular    | pChEMBL ≥ 5.0    | ≤ 10 µM        |
| Organism    | pChEMBL ≥ 4.5    | ≤ ~33 µM       |

---

## Project Structure

```
lead_craft_api/
├── src/
│   └── mol_activity/
│       ├── main.py                          # Main entry point: basic_query()
│       └── utils/
│           ├── config.py                    # Configuration (DB paths, constants)
│           ├── files_and_SQL_utils.py       # SQLAlchemy models & database
│           ├── logging_utils.py             # Logging configuration
│           ├── mol_activity_data_utils.py   # ChEMBL API interaction & processing
│           └── similarity_data_utils.py     # RDKit fingerprints & similarity
├── tests/
│   ├── test_files_and_SQL_utils.py
│   ├── test_mol_activity_data_utils.py
│   └── test_similarity_data_utils.py
├── notebooks/
│   └── LeadCraft_example.ipynb               # Interactive demo
├── data/                                     # Auto-created folder for SQLite cache
├── environment.yml                           # Conda environment specification
└── README.md
```

## Known Limitations

### ChEMBL API Reliability

- **Issue**: ChEMBL public API occasionally returns HTTP 500 errors or experiences downtime
- **How to solve**: Local caching ensures data persists; retry after 10-30 minutes (or few hours)

### Import-Time API Calls

- **Issue**: `chembl_webresource_client` fetches API schema at import time
- **Current Status**: Lazy imports implemented in data utils to minimize impact
- **Impact**: First import may fail if ChEMBL is down; cached queries unaffected


---

## Roadmap

### (v0.2)

- [ ] FastAPI REST endpoints for queries and cache management
- [ ] Docker containerization with pre-populated dev database
- [ ] Automatic retry logic for ChEMBL calls

### (v0.3)

- [ ] PostgreSQL support for production deployments
- [ ] Export formats (SDF, CSV with structures)
- [ ] Molecular docking integration (AutoDock Vina)

### (v1.0)

- [ ] QSAR model predictions
- [ ] Advanced filtering (molecular weight, LogP, rule-of-five)
- [ ] Multi-objective scoring and ranking

---

## Citation

If you use LeadCraft in your research, please cite:

- **ChEMBL**: Mendez et al. (2019). ChEMBL: towards direct deposition of bioassay data. *Nucleic Acids
  Research*. [DOI:10.1093/nar/gky1075]
- **RDKit**: [https://www.rdkit.org](https://www.rdkit.org)

---
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open a GitHub issue or contact: reshviktor1@gmail.com.
