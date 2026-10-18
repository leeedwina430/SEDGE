"""Reference-free validity matching LDMol's original utils.get_validity helper."""
from __future__ import annotations

from rdkit import Chem, rdBase

LDMOL_VALIDITY_RULE = "ldmol_utils_get_validity_v1"


def ldmol_validity_definition():
    """Make the exact upstream path and the denominator convention inspectable."""
    return {
        "rule_id": LDMOL_VALIDITY_RULE,
        "display_name": "LDMol-compatible validity",
        "upstream_function": "utils.get_validity",
        "upstream_source": "https://github.com/jinhojsk515/LDMol/blob/main/utils.py",
        "rule": "raw nonempty string; default MolFromSmiles then MolToSmiles(isomericSmiles=False) succeeds",
        "raw_smiles_preprocessing": "none",
        "extra_fragment_carbon_metal_filters": False,
        "stereoisomer_enumeration": False,
        "batch_denominator": "all_planned_requests_including_missing_and_invalid",
        "all_invalid_rate": 0.0,
        "note": "Not metrics.mol_evaluate or molfinger_evaluate; their reference-dependent paths and empty handling differ",
    }


def ldmol_compatible_valid(raw_smiles) -> bool:
    """Test the upstream per-output acceptance rule, independently of task validity.

    The temporary non-isomeric serialization only checks success. It never
    replaces the generated SMILES or the identity used by other metrics.
    Missing/non-string outputs are failures in our JSON/CSV output contract.
    """
    if not isinstance(raw_smiles, str) or raw_smiles == "":
        return False
    with rdBase.BlockLogs():
        try:
            Chem.MolToSmiles(Chem.MolFromSmiles(raw_smiles), isomericSmiles=False)
        except Exception:
            return False
    return True
