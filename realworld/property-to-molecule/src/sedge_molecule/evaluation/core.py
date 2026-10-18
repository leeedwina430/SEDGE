"""Single-molecule evaluation. No dataset, sampling, or file-layout dependencies."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors

from .validity import LDMOL_VALIDITY_RULE, ldmol_compatible_valid, ldmol_validity_definition

EXPECTED_RDKIT = "2023.09.6"
SCHEMA = "continuous_generation_evaluation_v1"
# Atomic numbers excluded by the evaluation validity rules.
METALS = frozenset([
    3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    *range(55, 85), *range(87, 117),
])


@dataclass(frozen=True)
class PropertyDefinition:
    """A named scalar property; definition and units are included in provenance."""

    unit: str
    definition: str
    calculator: Callable


DEFAULT_PROPERTIES = MappingProxyType({
    "mol_logp": PropertyDefinition("dimensionless", "rdkit.Chem.Crippen.MolLogP", Crippen.MolLogP),
    "tpsa": PropertyDefinition("angstrom^2", "rdkit.Chem.rdMolDescriptors.CalcTPSA; includeSandP=False", rdMolDescriptors.CalcTPSA),
    "qed": PropertyDefinition("dimensionless", "rdkit.Chem.QED.qed; default weights", QED.qed),
    "hbd": PropertyDefinition("count", "rdkit.Chem.rdMolDescriptors.CalcNumHBD", rdMolDescriptors.CalcNumHBD),
    "hba": PropertyDefinition("count", "rdkit.Chem.rdMolDescriptors.CalcNumHBA", rdMolDescriptors.CalcNumHBA),
    "mol_wt": PropertyDefinition("g/mol", "rdkit.Chem.Descriptors.MolWt", Descriptors.MolWt),
    "aromatic_ring_count": PropertyDefinition("count", "rdkit.Chem.rdMolDescriptors.CalcNumAromaticRings", rdMolDescriptors.CalcNumAromaticRings),
})


def finite_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def load_profile(variant: str) -> dict:
    if variant not in {"v0-1", "v0-2", "three_properties_example"}:
        raise ValueError(f"Unknown bundled profile: {variant}")
    return json.loads((Path(__file__).parent / "profiles" / f"{variant}.json").read_text())


def _validate_expression(node, properties: set[str], boolean: bool, depth=0) -> None:
    if depth > 24:
        raise ValueError("Region expression exceeds maximum depth")
    if not boolean and isinstance(node, (int, float)) and not isinstance(node, bool):
        finite_number(node, "region constant")
        return
    if not isinstance(node, dict) or len(node) != 1:
        raise ValueError("Each region expression must contain one allowed operator")
    op, args = next(iter(node.items()))
    if not boolean and op == "property":
        if not isinstance(args, str) or args not in properties:
            raise ValueError(f"Region property is not a target property: {args!r}")
        return
    if boolean and op == "not":
        _validate_expression(args, properties, True, depth + 1)
        return
    if not isinstance(args, list) or not args or len(args) > 64:
        raise ValueError(f"Invalid argument list for region operator {op}")
    if boolean and op in {"all", "any"}:
        for arg in args:
            _validate_expression(arg, properties, True, depth + 1)
        return
    if boolean and op in {"ge", "gt", "le", "lt", "eq"} and len(args) == 2:
        for arg in args:
            _validate_expression(arg, properties, False, depth + 1)
        return
    if not boolean and op in {"add", "sub", "mul", "div"}:
        if (op == "add" and len(args) < 2) or (op != "add" and len(args) != 2):
            raise ValueError(f"Wrong arity for region operator {op}")
        for arg in args:
            _validate_expression(arg, properties, False, depth + 1)
        if op == "div" and finite_number(args[1], "constant region divisor") == 0:
            raise ValueError("Region divisor cannot be zero")
        return
    raise ValueError(f"Unsupported region operator or expression type: {op}")


def _expression(node, values: Mapping[str, float]):
    if isinstance(node, (int, float)):
        return float(node)
    op, args = next(iter(node.items()))
    if op == "property":
        return values[args]
    if op == "not":
        return not _expression(args, values)
    if op in {"all", "any"}:
        evaluated = (_expression(arg, values) for arg in args)
        return all(evaluated) if op == "all" else any(evaluated)
    evaluated = [_expression(arg, values) for arg in args]
    a, b = evaluated[:2]
    if op in {"ge", "gt", "le", "lt", "eq"}:
        return {"ge": a >= b, "gt": a > b, "le": a <= b,
                "lt": a < b, "eq": a == b}[op]
    if op == "add":
        # Ordered binary addition, preserving LogP + (TPSA / 20).
        result = evaluated[0]
        for value in evaluated[1:]:
            result += value
    elif op == "sub":
        result = a - b
    elif op == "mul":
        result = a * b
    else:
        result = a / b
    if not math.isfinite(result):
        raise ValueError("Nonfinite region arithmetic")
    return result


def _parse_molecule(raw_smiles):
    """Return the canonical molecule and structured chemistry diagnostics."""
    result = {"molecule_valid": False, "failure_reason": None,
              "canonical_smiles": None, "formal_charge": None, "fragment_count": None}

    def fail(reason):
        result["failure_reason"] = reason
        return None, result

    if not isinstance(raw_smiles, str):
        return fail("smiles_not_string")
    text = raw_smiles.strip()
    if not text:
        return fail("empty_smiles")
    if any(char.isspace() for char in text):
        return fail("internal_whitespace_or_trailing_text")
    if any(ord(char) < 32 or ord(char) == 127 for char in text):
        return fail("control_character")
    params = Chem.SmilesParserParams()
    params.sanitize = False
    params.parseName = False
    params.allowCXSMILES = False
    with rdBase.BlockLogs():
        try:
            molecule = Chem.MolFromSmiles(text, params)
        except Exception:
            molecule = None
    if molecule is None:
        return fail("parse_failure")
    with rdBase.BlockLogs():
        try:
            Chem.SanitizeMol(molecule)
        except Exception:
            return fail("sanitize_failure")
    result["fragment_count"] = len(Chem.GetMolFrags(molecule))
    if result["fragment_count"] != 1:
        return fail("multiple_fragments")
    elements = {atom.GetAtomicNum() for atom in molecule.GetAtoms()}
    if 6 not in elements:
        return fail("no_carbon")
    if elements & METALS:
        return fail("contains_metal")
    try:
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        params.sanitize = True
        roundtrip = Chem.MolFromSmiles(canonical, params)
        if roundtrip is None or Chem.MolToSmiles(roundtrip, canonical=True, isomericSmiles=True) != canonical:
            return fail("canonical_roundtrip_failure")
    except Exception:
        return fail("canonical_roundtrip_failure")
    result.update(molecule_valid=True, canonical_smiles=canonical,
                  formal_charge=sum(atom.GetFormalCharge() for atom in roundtrip.GetAtoms()))
    # Compute on the canonical round trip so equivalent SMILES atom order is irrelevant.
    return roundtrip, result


class Evaluator:
    """Freeze a profile once; evaluate any number of named, continuous targets."""

    def __init__(self, profile: Mapping, properties: Mapping[str, PropertyDefinition] | None = None):
        self._profile = deepcopy(dict(profile))
        if self._profile.get("schema_version") != SCHEMA:
            raise ValueError("Unsupported evaluation profile schema")
        if self._profile.get("rdkit_version") != EXPECTED_RDKIT or rdBase.rdkitVersion != EXPECTED_RDKIT:
            raise ValueError(f"This protocol requires RDKit {EXPECTED_RDKIT}; running {rdBase.rdkitVersion}")
        if not isinstance(self._profile.get("profile_id"), str) or not self._profile["profile_id"]:
            raise ValueError("profile_id must be a nonempty string")
        registry = dict(DEFAULT_PROPERTIES if properties is None else properties)
        tolerances = self._profile.get("tolerances")
        if not isinstance(tolerances, dict) or not tolerances:
            raise ValueError("At least one named tolerance profile is required")
        self.main_tolerance = self._profile.get("main_tolerance")
        if self.main_tolerance not in tolerances:
            raise ValueError("main_tolerance must name a declared tolerance profile")
        first = tolerances[self.main_tolerance]
        if not isinstance(first, dict) or not first:
            raise ValueError("Tolerance profiles must map property names to absolute tolerances")
        self.names = tuple(sorted(first))
        for name in self.names:
            definition = registry.get(name)
            if not isinstance(definition, PropertyDefinition) or not callable(definition.calculator):
                raise ValueError(f"Unknown or invalid property: {name}")
        self.properties = {name: registry[name] for name in self.names}
        for label, settings in tolerances.items():
            if not isinstance(label, str) or not label or not isinstance(settings, dict) or set(settings) != set(self.names):
                raise ValueError("Every tolerance profile must specify the same named properties")
            for name, value in settings.items():
                if finite_number(value, f"tolerance {label}.{name}") < 0:
                    raise ValueError("Absolute tolerances must be nonnegative")
        self.tolerances = deepcopy(tolerances)
        self.guard_ulps = self._profile.get("comparison_guard_ulps", 8)
        if type(self.guard_ulps) is not int or not 0 <= self.guard_ulps <= 64:
            raise ValueError("comparison_guard_ulps must be an integer in [0,64]")
        self.region = deepcopy(self._profile.get("region"))
        _validate_expression(self.region, set(self.names), True)

    @property
    def profile(self) -> dict:
        """Return a copy so callers cannot change the validated profile in place."""
        return deepcopy(self._profile)

    def validate_targets(self, targets: Mapping) -> dict[str, float]:
        if not isinstance(targets, Mapping) or set(targets) != set(self.names):
            raise ValueError(f"Targets must specify exactly: {', '.join(self.names)}")
        return {name: finite_number(targets[name], f"target {name}") for name in self.names}

    def region_contains(self, values: Mapping) -> bool:
        return bool(_expression(self.region, self.validate_targets(values)))

    def metadata(self) -> dict:
        return {"rdkit_version": rdBase.rdkitVersion, "profile_id": self._profile["profile_id"],
                "chemistry_policy": "strict_single_fragment_carbon_no_frozen_metals_v1",
                "ldmol_compatible_validity": ldmol_validity_definition(),
                "metal_atomic_numbers": sorted(METALS),
                "properties": {name: {"unit": prop.unit, "definition": prop.definition}
                               for name, prop in self.properties.items()}}

    def evaluate(self, raw_smiles, targets: Mapping, *, train_bounds: Mapping | None = None) -> dict:
        """Evaluate an OOD target; configuration/request errors raise ValueError.

        Invalid model outputs return failed results. Targets outside the declared
        OOD region are request errors, not failed model generations. A reference
        structure, training table, or scaffold lookup is never required here.
        """
        targets = self.validate_targets(targets)
        if not self.region_contains(targets):
            raise ValueError("Requested target is outside the declared OOD region")
        bounds = None
        if train_bounds is not None:
            if not isinstance(train_bounds, Mapping) or set(train_bounds) != set(self.names):
                raise ValueError("train_bounds must cover exactly the target properties")
            bounds = {}
            for name, pair in train_bounds.items():
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError(f"Invalid train bounds for {name}")
                lo, hi = (finite_number(x, f"train bound {name}") for x in pair)
                if lo > hi:
                    raise ValueError(f"Reversed train bounds for {name}")
                bounds[name] = (lo, hi)
        molecule, chemistry = _parse_molecule(raw_smiles)
        result = {**chemistry, "profile_id": self._profile["profile_id"],
                  "ldmol_compatible_valid": ldmol_compatible_valid(raw_smiles),
                  "ldmol_compatible_validity_rule": LDMOL_VALIDITY_RULE,
                  "targets": targets, "properties_valid": False,
                  "measured": {name: None for name in self.names},
                  "errors": {name: {"signed": None, "absolute": None} for name in self.names},
                  "region_hit": False, "axis_covered": False if bounds is not None else None,
                  "profiles": {}}
        if molecule is not None:
            try:
                for name, definition in self.properties.items():
                    result["measured"][name] = finite_number(float(definition.calculator(molecule)), name)
                result["region_hit"] = self.region_contains(result["measured"])
                result["properties_valid"] = True
            except Exception as error:
                result["failure_reason"] = f"property_computation_failure:{type(error).__name__}"
            if result["properties_valid"]:
                for name in self.names:
                    signed = result["measured"][name] - targets[name]
                    result["errors"][name] = {"signed": signed, "absolute": abs(signed)}
                if bounds is not None:
                    result["axis_covered"] = all(lo <= result["measured"][name] <= hi
                                                 for name, (lo, hi) in bounds.items())
        for label, tolerances in self.tolerances.items():
            matches = {}
            for name, epsilon in tolerances.items():
                if not result["properties_valid"]:
                    matches[name] = False
                    continue
                # A tiny representation guard, NOT extra scientific tolerance.
                guard = (self.guard_ulps * max(math.ulp(result["measured"][name]),
                         math.ulp(targets[name]), math.ulp(float(epsilon)))) if epsilon else 0.0
                absolute = result["errors"][name]["absolute"]
                matches[name] = math.isfinite(absolute) and (absolute <= epsilon or absolute - epsilon <= guard)
            target_match = all(matches.values())
            success = target_match and result["region_hit"]
            result["profiles"][label] = {"property_match": matches, "target_match": target_match,
                "extrapolation_success": success,
                "axis_covered_extrapolation_success": (success and result["axis_covered"]) if bounds is not None else None}
        result.update({key: result["profiles"][self.main_tolerance][key]
                       for key in ("target_match", "extrapolation_success", "axis_covered_extrapolation_success")})
        return result
