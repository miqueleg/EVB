from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

KJ_PER_KCAL = 4.184
KJ_PER_EV = 96.48533212331002
KJ_PER_HARTREE = 2625.4996394799

ALIASES = {
    "rc": "reactant",
    "r": "reactant",
    "reactant_complex": "reactant",
    "reactants": "reactant",
    "bound": "bound",
    "ts": "transition_state",
    "transition-state": "transition_state",
    "transitionstate": "transition_state",
    "prod": "product",
    "products": "product",
}


def canonical_label(label: str) -> str:
    key = str(label).strip().replace(" ", "_").lower()
    return ALIASES.get(key, key)


def energy_to_kj(value: float, unit: str) -> float:
    u = unit.strip().lower().replace(" ", "")
    if u in {"kj/mol", "kjmol", "kilojoule/mol", "kilojoules/mol"}:
        return float(value)
    if u in {"kcal/mol", "kcalmol"}:
        return float(value) * KJ_PER_KCAL
    if u in {"ev", "electronvolt"}:
        return float(value) * KJ_PER_EV
    if u in {"hartree", "eh"}:
        return float(value) * KJ_PER_HARTREE
    raise ValueError(f"Unsupported reference energy unit: {unit!r}")


@dataclass(slots=True)
class ReferencePoint:
    label: str
    relative_kj_mol: float | None = None
    absolute_kj_mol: float | None = None
    uncertainty_kj_mol: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReferenceProfile:
    method_label: str
    zero_label: str
    points: dict[str, ReferencePoint]
    calibration_target: list[str]
    primary_barrier: tuple[str, str] | None = None
    source_path: str | None = None

    @property
    def target_kj_mol(self) -> dict[str, float]:
        return {label: point.relative_kj_mol for label, point in self.points.items() if point.relative_kj_mol is not None}

    @property
    def target_kcal_mol(self) -> dict[str, float]:
        return {label: value / KJ_PER_KCAL for label, value in self.target_kj_mol.items()}

    def barrier_kj_mol(self, start: str = "reactant", end: str = "transition_state") -> float | None:
        a = self.points.get(canonical_label(start))
        b = self.points.get(canonical_label(end))
        if a is None or b is None or a.relative_kj_mol is None or b.relative_kj_mol is None:
            return None
        return b.relative_kj_mol - a.relative_kj_mol

    def reaction_free_energy_kj_mol(self) -> float | None:
        a = self.points.get("reactant")
        b = self.points.get("product")
        if a is None or b is None or a.relative_kj_mol is None or b.relative_kj_mol is None:
            return None
        return b.relative_kj_mol - a.relative_kj_mol


def _read_payload(path: str | Path) -> dict[str, Any]:
    if yaml is None:  # pragma: no cover
        raise ImportError("PyYAML is required to load reference profiles.")
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Reference profile file must contain a mapping.")
    return payload


def load_reference_profile(path: str | Path, profile_name: str | None = None) -> ReferenceProfile:
    payload = _read_payload(path)
    source = Path(path)
    if "reference_profile" in payload:
        return _load_schema(payload["reference_profile"], source)
    if "points" in payload:
        return _load_schema(payload, source)
    if "states" in payload:
        return _load_state_profile(payload, source, profile_name)
    raise ValueError("Reference profile must contain 'reference_profile', 'points', or 'states'.")


def _load_schema(data: dict[str, Any], source: Path) -> ReferenceProfile:
    unit = str(data.get("energy_unit") or data.get("units", {}).get("relative_energy") or "kJ/mol")
    method = str(data.get("method_label") or data.get("method") or "user_reference")
    zero = canonical_label(str(data.get("zero") or "reactant"))
    raw_points = data.get("points") or {}
    points: dict[str, ReferencePoint] = {}
    absolutes: dict[str, float] = {}
    for label, row in raw_points.items():
        canon = canonical_label(label)
        row = row or {}
        rel = row.get("relative_energy")
        abs_val = row.get("absolute_energy")
        if rel is not None:
            rel_kj = energy_to_kj(float(rel), unit)
        else:
            rel_kj = None
        abs_kj = energy_to_kj(float(abs_val), unit) if abs_val is not None else None
        if abs_kj is not None:
            absolutes[canon] = abs_kj
        unc = row.get("uncertainty")
        points[canon] = ReferencePoint(canon, rel_kj, abs_kj, energy_to_kj(float(unc), unit) if unc is not None else None, dict(row))
    if any(p.relative_kj_mol is None for p in points.values()) and absolutes:
        if zero not in absolutes:
            raise ValueError(f"Zero reference {zero!r} has no absolute energy.")
        base = absolutes[zero]
        for label, point in points.items():
            if point.relative_kj_mol is None and point.absolute_kj_mol is not None:
                point.relative_kj_mol = point.absolute_kj_mol - base
    target = data.get("calibration_target", {}).get("profile") if isinstance(data.get("calibration_target"), dict) else None
    if not target:
        target = [label for label in ["reactant", "transition_state", "product"] if label in points] or list(points)
    primary = data.get("calibration_target", {}).get("primary_barrier") if isinstance(data.get("calibration_target"), dict) else None
    primary_tuple = tuple(canonical_label(x) for x in primary) if primary and len(primary) == 2 else None
    return ReferenceProfile(method, zero, points, [canonical_label(x) for x in target], primary_tuple, str(source))


def _load_state_profile(payload: dict[str, Any], source: Path, profile_name: str | None) -> ReferenceProfile:
    method = str(payload.get("method") or payload.get("method_label") or "user_reference")
    profiles = payload.get("reaction_profiles") or {}
    recommended = payload.get("recommended_calibration_profile")
    chosen = profile_name or recommended or next(iter(profiles), None)
    if chosen and chosen in profiles:
        row = profiles[chosen]
        unit = str(row.get("units") or "kcal/mol")
        points = {
            canonical_label(label): ReferencePoint(canonical_label(label), energy_to_kj(float(value), unit), None, None, {})
            for label, value in row.items()
            if label != "units"
        }
        target = [label for label in ["reactant", "transition_state", "product"] if label in points] or list(points)
        return ReferenceProfile(method, target[0] if target else "reactant", points, target, (target[0], "transition_state") if "transition_state" in points and target else None, str(source))
    unit = str(payload.get("units", {}).get("relative_energy") or "kcal/mol")
    points = {}
    for label, row in (payload.get("states") or {}).items():
        canon = canonical_label(label)
        rel = row.get("relative_to_unbound_kcal_mol", row.get("relative_energy"))
        if rel is not None:
            points[canon] = ReferencePoint(canon, energy_to_kj(float(rel), unit), None, None, dict(row))
    target = [label for label in ["reactant", "transition_state", "product"] if label in points] or list(points)
    return ReferenceProfile(method, target[0] if target else "reactant", points, target, None, str(source))
