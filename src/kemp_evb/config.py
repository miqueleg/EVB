from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass(slots=True)
class StateFiles:
    prmtop: str
    inpcrd: str
    pdb: str | None = None
    format: str = "amber"


@dataclass(slots=True)
class CalibrationGeometryFiles:
    min1: str | None = None
    min2: str | None = None
    ts: str | None = None


@dataclass(slots=True)
class CalibrationData:
    e_qmmm_min1: float
    e_qmmm_min2: float
    e_qmmm_ts: float
    e_mm_min1_state1: float | None = None
    e_mm_min1_state2: float | None = None
    e_mm_min2_state1: float | None = None
    e_mm_min2_state2: float | None = None
    e_mm_ts_state1: float | None = None
    e_mm_ts_state2: float | None = None
    coordinates: CalibrationGeometryFiles = field(default_factory=CalibrationGeometryFiles)


@dataclass(slots=True)
class IntegratorSettings:
    name: str = "LangevinMiddle"
    timestep_fs: float = 1.0
    friction_per_ps: float = 1.0
    seed: int = 2026


@dataclass(slots=True)
class MDRunSettings:
    equilibration_steps: int = 0
    production_steps: int = 100
    report_stride: int = 10
    save_stride: int | None = None
    platform: str | None = None
    temperature_k: float = 300.0
    pressure_bar: float | None = None
    nonbonded_method: str = "PME"
    constraints: str = "HBonds"
    minimize_steps: int = 200
    minimize_tolerance: float = 10.0


@dataclass(slots=True)
class EquilibrationRestraintSettings:
    enabled: bool = False
    atom1: int | None = None
    atom2: int | None = None
    target_distance_nm: float | None = None
    force_constant_kj_mol_nm2: float = 2092.0


@dataclass(slots=True)
class ProductionRestraintSettings:
    enabled: bool = False
    substrate_com_atoms: list[int] = field(default_factory=list)
    substrate_com_force_constant_kj_mol_nm2: float | None = None


@dataclass(slots=True)
class FarFieldRestraintSettings:
    enabled: bool = False
    active_atoms: list[int] = field(default_factory=list)
    restrained_atoms: list[int] = field(default_factory=list)
    radius_nm: float = 1.2
    force_constant_kj_mol_nm2: float = 25.0


@dataclass(slots=True)
class UmbrellaRampSettings:
    enabled: bool = True
    fractions: list[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 1.0])


@dataclass(slots=True)
class SeedRelaxationSettings:
    enabled: bool = True
    minimization_steps: int = 500
    equilibration_steps: int = 0
    timestep_fs: float | None = None
    temperature_k: float | None = None
    restraint_force_constant_kj_mol_nm2: float = 250.0
    restraint_decay: list[float] = field(default_factory=lambda: [1.0, 0.5, 0.1, 0.0])


@dataclass(slots=True)
class SimulationSettings:
    timestep_fs: float = 1.0
    temperature_k: float = 300.0
    friction_per_ps: float = 1.0
    steps: int = 100
    report_interval: int = 10
    minimize_steps: int = 200
    minimize_tolerance: float = 10.0
    seed: int = 2026
    platform: str | None = None
    nonbonded_method: str = "PME"
    constraints: str = "HBonds"
    integrator: str = "LangevinMiddle"


@dataclass(slots=True)
class EVBParameterConfig:
    delta_alpha: float | None = None
    h12: float | None = None


@dataclass(slots=True)
class CouplingModelConfig:
    model: str = "constant"
    delta_alpha_kj_mol: float | None = None
    h12_kj_mol: float | None = None
    parameters: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class CVDefinition:
    donor: int
    proton: int
    acceptor: int


@dataclass(slots=True)
class DistanceCVDefinition:
    name: str
    atom1: int
    atom2: int


@dataclass(slots=True)
class ReactionCoordinateDefinition:
    name: str
    kind: str
    atom1: int
    atom2: int
    atom3: int | None = None
    event_threshold_nm: float | None = None


@dataclass(slots=True)
class GapDefinition:
    shifted: bool = True


@dataclass(slots=True)
class ObservableSettings:
    gap: GapDefinition = field(default_factory=GapDefinition)
    distances: list[DistanceCVDefinition] = field(default_factory=list)
    reaction_coordinates: list[ReactionCoordinateDefinition] = field(default_factory=list)


@dataclass(slots=True)
class ReactionMetadata:
    name: str = "unnamed_reaction"
    phase: str = "solution"
    temperature_k: float = 300.0
    pressure_bar: float | None = None
    notes: str | None = None


@dataclass(slots=True)
class ReactionSettings:
    metadata: ReactionMetadata = field(default_factory=ReactionMetadata)
    atoms: CVDefinition | None = None
    substrate_atoms: list[int] = field(default_factory=list)
    environment_atoms: list[int] = field(default_factory=list)


@dataclass(slots=True)
class ProjectSettings:
    name: str = "kemp-evb"
    output_dir: str = "outputs"


@dataclass(slots=True)
class MappingWindows:
    lambda_values: list[float] = field(default_factory=list)


@dataclass(slots=True)
class GapUmbrellaWindows:
    centers_kj_mol: list[float] = field(default_factory=list)
    force_constant_kj_mol2: float | None = None


@dataclass(slots=True)
class ProtonTransferUmbrellaWindows:
    centers_nm: list[float] = field(default_factory=list)
    force_constant_kj_mol_nm2: float | None = None


@dataclass(slots=True)
class SamplingWindows:
    mapping: MappingWindows = field(default_factory=MappingWindows)
    gap_umbrella: GapUmbrellaWindows = field(default_factory=GapUmbrellaWindows)
    proton_transfer_umbrella: ProtonTransferUmbrellaWindows = field(default_factory=ProtonTransferUmbrellaWindows)


@dataclass(slots=True)
class WindowSeedDefinition:
    window_id: str
    coordinates: str
    branch: str | None = None


@dataclass(slots=True)
class SamplingSettings:
    mode: str = "mapping"
    bidirectional: bool = False
    integrator: IntegratorSettings = field(default_factory=IntegratorSettings)
    md: MDRunSettings = field(default_factory=MDRunSettings)
    equilibration_restraint: EquilibrationRestraintSettings = field(default_factory=EquilibrationRestraintSettings)
    production_restraint: ProductionRestraintSettings = field(default_factory=ProductionRestraintSettings)
    far_field_restraint: FarFieldRestraintSettings = field(default_factory=FarFieldRestraintSettings)
    umbrella_ramp: UmbrellaRampSettings = field(default_factory=UmbrellaRampSettings)
    seed_relaxation: SeedRelaxationSettings = field(default_factory=SeedRelaxationSettings)
    windows: SamplingWindows = field(default_factory=SamplingWindows)
    seed_windows: list[WindowSeedDefinition] = field(default_factory=list)


@dataclass(slots=True)
class HistogramSettings:
    bin_min_kj_mol: float = -200.0
    bin_max_kj_mol: float = 200.0
    n_bins: int = 200


@dataclass(slots=True)
class PMFSettings:
    temperature_k: float = 300.0
    zero_mode: str = "reactant_min"


@dataclass(slots=True)
class BarrierSettings:
    reactant_region: tuple[float, float] | None = None
    product_region: tuple[float, float] | None = None


@dataclass(slots=True)
class UncertaintySettings:
    blocks: int = 5
    bootstrap_samples: int = 200


@dataclass(slots=True)
class AnalysisSettings:
    histogram: HistogramSettings = field(default_factory=HistogramSettings)
    pmf: PMFSettings = field(default_factory=PMFSettings)
    barrier: BarrierSettings = field(default_factory=BarrierSettings)
    uncertainty: UncertaintySettings = field(default_factory=UncertaintySettings)


@dataclass(slots=True)
class FitTargets:
    reaction_free_energy_kj_mol: float | None = None
    barrier_kj_mol: float | None = None
    source_label: str | None = None


@dataclass(slots=True)
class FitScanSettings:
    delta_alpha_min_kj_mol: float = -50.0
    delta_alpha_max_kj_mol: float = 50.0
    delta_alpha_samples: int = 21
    h12_min_kj_mol: float = 0.0
    h12_max_kj_mol: float = 50.0
    h12_samples: int = 11


@dataclass(slots=True)
class FitSettings:
    bootstrap_targets: FitTargets = field(default_factory=FitTargets)
    ensemble_targets: FitTargets = field(default_factory=FitTargets)
    scan: FitScanSettings = field(default_factory=FitScanSettings)


@dataclass(slots=True)
class PlumedSettings:
    enabled: bool = False
    script: str | None = None
    script_file: str | None = None
    output_colvar: str = "COLVAR"
    restart: bool = False


@dataclass(slots=True)
class EVBConfig:
    state1: StateFiles
    state2: StateFiles
    calibration: CalibrationData | None = None
    evb_parameters: EVBParameterConfig = field(default_factory=EVBParameterConfig)
    cv: CVDefinition | None = None
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    output_dir: str = "outputs"
    start_state: str = "state1"
    project: ProjectSettings = field(default_factory=ProjectSettings)
    reaction: ReactionSettings = field(default_factory=ReactionSettings)
    observables: ObservableSettings = field(default_factory=ObservableSettings)
    sampling: SamplingSettings = field(default_factory=SamplingSettings)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    fit: FitSettings = field(default_factory=FitSettings)
    plumed: PlumedSettings = field(default_factory=PlumedSettings)


def _coerce_state_files(data: dict[str, Any]) -> StateFiles:
    topology = data.get("topology", data.get("prmtop"))
    coordinates = data.get("coordinates", data.get("inpcrd"))
    if topology is None or coordinates is None:
        raise ValueError("Each state requires topology/prmtop and coordinates/inpcrd fields.")
    return StateFiles(
        prmtop=topology,
        inpcrd=coordinates,
        pdb=data.get("pdb"),
        format=data.get("format", "amber"),
    )


def _coerce_cv(data: dict[str, Any] | None) -> CVDefinition | None:
    if data is None:
        return None
    return CVDefinition(donor=data["donor"], proton=data["proton"], acceptor=data["acceptor"])


def _coerce_distance_cvs(data: list[dict[str, Any]] | None) -> list[DistanceCVDefinition]:
    if not data:
        return []
    return [DistanceCVDefinition(**entry) for entry in data]


def _coerce_reaction_coordinates(data: list[dict[str, Any]] | None) -> list[ReactionCoordinateDefinition]:
    if not data:
        return []
    return [ReactionCoordinateDefinition(**entry) for entry in data]


def _coerce_window_seeds(data: list[dict[str, Any]] | None) -> list[WindowSeedDefinition]:
    if not data:
        return []
    return [WindowSeedDefinition(**entry) for entry in data]


def _coerce_fit_targets(data: dict[str, Any] | None) -> FitTargets:
    if not data:
        return FitTargets()
    return FitTargets(**data)


def _coerce_barrier_regions(data: dict[str, Any] | None) -> BarrierSettings:
    if not data:
        return BarrierSettings()
    reactant_region = tuple(data["reactant_region"]) if data.get("reactant_region") else None
    product_region = tuple(data["product_region"]) if data.get("product_region") else None
    return BarrierSettings(reactant_region=reactant_region, product_region=product_region)


def _from_legacy_mapping(data: dict[str, Any]) -> EVBConfig:
    state1 = StateFiles(**data["state1"])
    state2 = StateFiles(**data["state2"])
    calibration = None
    if data.get("calibration"):
        calibration_payload = dict(data["calibration"])
        coordinates = CalibrationGeometryFiles(**calibration_payload.pop("coordinates", {}))
        calibration = CalibrationData(coordinates=coordinates, **calibration_payload)
    evb_parameters = EVBParameterConfig(**data.get("evb_parameters", {}))
    cv = CVDefinition(**data["cv"]) if data.get("cv") else None
    simulation = SimulationSettings(**data.get("simulation", {}))
    project = ProjectSettings(
        name=data.get("project", {}).get("name", "kemp-evb"),
        output_dir=data.get("output_dir", data.get("project", {}).get("output_dir", "outputs")),
    )
    observables = ObservableSettings(
        gap=GapDefinition(),
        distances=[] if cv is None else [
            DistanceCVDefinition(name="donor_h", atom1=cv.donor, atom2=cv.proton),
            DistanceCVDefinition(name="h_acceptor", atom1=cv.proton, atom2=cv.acceptor),
        ],
        reaction_coordinates=[] if cv is None else [
            ReactionCoordinateDefinition(
                name="proton_transfer_rc",
                kind="difference_of_distances",
                atom1=cv.donor,
                atom2=cv.proton,
                atom3=cv.acceptor,
                event_threshold_nm=0.0,
            )
        ],
    )
    reaction = ReactionSettings(
        metadata=ReactionMetadata(temperature_k=simulation.temperature_k),
        atoms=cv,
    )
    sampling = SamplingSettings(
        bidirectional=False,
        integrator=IntegratorSettings(
            name=simulation.integrator,
            timestep_fs=simulation.timestep_fs,
            friction_per_ps=simulation.friction_per_ps,
            seed=simulation.seed,
        ),
        md=MDRunSettings(
            equilibration_steps=0,
            production_steps=simulation.steps,
            report_stride=simulation.report_interval,
            platform=simulation.platform,
            temperature_k=simulation.temperature_k,
            nonbonded_method=simulation.nonbonded_method,
            constraints=simulation.constraints,
            minimize_steps=simulation.minimize_steps,
            minimize_tolerance=simulation.minimize_tolerance,
        ),
        equilibration_restraint=EquilibrationRestraintSettings(enabled=False),
        production_restraint=ProductionRestraintSettings(enabled=False),
        far_field_restraint=FarFieldRestraintSettings(enabled=False),
        umbrella_ramp=UmbrellaRampSettings(enabled=False),
        seed_relaxation=SeedRelaxationSettings(enabled=False),
        seed_windows=[],
    )
    return EVBConfig(
        state1=state1,
        state2=state2,
        calibration=calibration,
        evb_parameters=evb_parameters,
        cv=cv,
        simulation=simulation,
        output_dir=project.output_dir,
        start_state=data.get("start_state", "state1"),
        project=project,
        reaction=reaction,
        observables=observables,
        sampling=sampling,
    )


def _from_modern_mapping(data: dict[str, Any]) -> EVBConfig:
    project_payload = data.get("project", {})
    project = ProjectSettings(
        name=project_payload.get("name", "kemp-evb"),
        output_dir=project_payload.get("output_dir", "outputs"),
    )
    states_payload = data.get("states")
    if not states_payload:
        raise ValueError("Modern config requires a top-level 'states' section.")
    state1 = _coerce_state_files(states_payload["state1"])
    state2 = _coerce_state_files(states_payload["state2"])

    reaction_payload = data.get("reaction", {})
    metadata_payload = reaction_payload.get("metadata", {})
    reaction = ReactionSettings(
        metadata=ReactionMetadata(**metadata_payload) if metadata_payload else ReactionMetadata(),
        atoms=_coerce_cv(reaction_payload.get("atoms")),
        substrate_atoms=list(reaction_payload.get("substrate_atoms", [])),
        environment_atoms=list(reaction_payload.get("environment_atoms", [])),
    )
    cv = reaction.atoms

    evb_payload = data.get("evb", {})
    coupling_payload = evb_payload.get("coupling_model", {})
    parameter_block = dict(coupling_payload.get("parameters", {}))
    delta_alpha = parameter_block.get("delta_alpha_kj_mol", coupling_payload.get("delta_alpha_kj_mol"))
    h12 = parameter_block.get("h12_kj_mol", coupling_payload.get("h12_kj_mol"))
    evb_parameters = EVBParameterConfig(delta_alpha=delta_alpha, h12=h12)

    sampling_payload = data.get("sampling", {})
    integrator_payload = sampling_payload.get("integrator", {})
    md_payload = sampling_payload.get("md", {})
    windows_payload = sampling_payload.get("windows", {})
    sampling = SamplingSettings(
        mode=sampling_payload.get("mode", "mapping"),
        bidirectional=sampling_payload.get("bidirectional", False),
        integrator=IntegratorSettings(**integrator_payload) if integrator_payload else IntegratorSettings(),
        md=MDRunSettings(**md_payload) if md_payload else MDRunSettings(),
        equilibration_restraint=EquilibrationRestraintSettings(**sampling_payload.get("equilibration_restraint", {})),
        production_restraint=ProductionRestraintSettings(**sampling_payload.get("production_restraint", {})),
        far_field_restraint=FarFieldRestraintSettings(**sampling_payload.get("far_field_restraint", {})),
        umbrella_ramp=UmbrellaRampSettings(**sampling_payload.get("umbrella_ramp", {})),
        seed_relaxation=SeedRelaxationSettings(**sampling_payload.get("seed_relaxation", {})),
        windows=SamplingWindows(
            mapping=MappingWindows(**windows_payload.get("mapping", {})),
            gap_umbrella=GapUmbrellaWindows(**windows_payload.get("gap_umbrella", {})),
            proton_transfer_umbrella=ProtonTransferUmbrellaWindows(**windows_payload.get("proton_transfer_umbrella", {})),
        ),
        seed_windows=_coerce_window_seeds(sampling_payload.get("seed_windows")),
    )
    simulation = SimulationSettings(
        timestep_fs=sampling.integrator.timestep_fs,
        temperature_k=sampling.md.temperature_k,
        friction_per_ps=sampling.integrator.friction_per_ps,
        steps=sampling.md.production_steps,
        report_interval=sampling.md.report_stride,
        minimize_steps=sampling.md.minimize_steps,
        minimize_tolerance=sampling.md.minimize_tolerance,
        seed=sampling.integrator.seed,
        platform=sampling.md.platform,
        nonbonded_method=sampling.md.nonbonded_method,
        constraints=sampling.md.constraints,
        integrator=sampling.integrator.name,
    )

    observables_payload = data.get("observables", {})
    reaction_coordinates = _coerce_reaction_coordinates(observables_payload.get("reaction_coordinates"))
    if not reaction_coordinates and cv is not None:
        reaction_coordinates = [
            ReactionCoordinateDefinition(
                name="proton_transfer_rc",
                kind="difference_of_distances",
                atom1=cv.donor,
                atom2=cv.proton,
                atom3=cv.acceptor,
                event_threshold_nm=0.0,
            )
        ]
    observables = ObservableSettings(
        gap=GapDefinition(**observables_payload.get("gap", {})),
        distances=_coerce_distance_cvs(observables_payload.get("distances")),
        reaction_coordinates=reaction_coordinates,
    )

    analysis_payload = data.get("analysis", {})
    analysis = AnalysisSettings(
        histogram=HistogramSettings(**analysis_payload.get("histogram", {})),
        pmf=PMFSettings(**analysis_payload.get("pmf", {})),
        barrier=_coerce_barrier_regions(analysis_payload.get("barrier")),
        uncertainty=UncertaintySettings(**analysis_payload.get("uncertainty", {})),
    )

    fit_payload = data.get("fit", {})
    fit = FitSettings(
        bootstrap_targets=_coerce_fit_targets(fit_payload.get("bootstrap_targets")),
        ensemble_targets=_coerce_fit_targets(fit_payload.get("ensemble_targets")),
        scan=FitScanSettings(**fit_payload.get("scan", {})),
    )
    plumed = PlumedSettings(**data.get("plumed", {}))
    _validate_modern_payload(data, evb_payload, coupling_payload)

    return EVBConfig(
        state1=state1,
        state2=state2,
        calibration=None,
        evb_parameters=evb_parameters,
        cv=cv,
        simulation=simulation,
        output_dir=project.output_dir,
        start_state=data.get("start_state", "state1"),
        project=project,
        reaction=reaction,
        observables=observables,
        sampling=sampling,
        analysis=analysis,
        fit=fit,
        plumed=plumed,
    )


def _from_mapping(data: dict[str, Any]) -> EVBConfig:
    if "states" in data:
        return _from_modern_mapping(data)
    return _from_legacy_mapping(data)


def load_config(path: str | Path) -> EVBConfig:
    path = Path(path)
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as handle:
        if suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise ImportError("PyYAML is required to read YAML config files.")
            payload = yaml.safe_load(handle)
        else:
            payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Configuration file must contain a mapping/object at the top level.")
    return _from_mapping(payload)


def validate_config(config: EVBConfig) -> list[str]:
    errors: list[str] = []
    if config.evb_parameters.delta_alpha is None:
        errors.append("evb.coupling_model.parameters.delta_alpha_kj_mol is required.")
    if config.evb_parameters.h12 is None:
        errors.append("evb.coupling_model.parameters.h12_kj_mol is required.")
    if config.sampling.mode not in {"mapping", "gap_umbrella", "proton_transfer_umbrella"}:
        errors.append(f"Unsupported sampling.mode {config.sampling.mode!r}.")
    if config.sampling.mode == "mapping" and not config.sampling.windows.mapping.lambda_values:
        errors.append("mapping mode requires sampling.windows.mapping.lambda_values.")
    if config.sampling.mode == "gap_umbrella":
        if not config.sampling.windows.gap_umbrella.centers_kj_mol:
            errors.append("gap_umbrella mode requires sampling.windows.gap_umbrella.centers_kj_mol.")
        if config.sampling.windows.gap_umbrella.force_constant_kj_mol2 is None:
            errors.append("gap_umbrella mode requires sampling.windows.gap_umbrella.force_constant_kj_mol2.")
    if config.plumed.enabled and not (config.plumed.script or config.plumed.script_file):
        errors.append("plumed.enabled requires plumed.script or plumed.script_file.")
    return errors


def _validate_modern_payload(data: dict[str, Any], evb_payload: dict[str, Any], coupling_payload: dict[str, Any]) -> None:
    allowed_top_level = {
        "project",
        "reaction",
        "states",
        "evb",
        "sampling",
        "observables",
        "analysis",
        "fit",
        "plumed",
        "start_state",
    }
    extra = sorted(set(data) - allowed_top_level)
    if extra:
        raise ValueError(f"Unknown top-level config section(s): {', '.join(extra)}")
    model = coupling_payload.get("model", "constant")
    if model != "constant":
        raise ValueError(
            "Only constant EVB coupling is currently supported. Geometry-dependent H12 is a planned extension."
        )
    if "geometry_dependent" in evb_payload:
        raise ValueError("Geometry-dependent H12 is not implemented in this validated API.")
