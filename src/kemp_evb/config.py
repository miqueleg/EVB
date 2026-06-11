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
class EnergyDecompositionSettings:
    enabled: bool = False
    mode: str = "exact"
    fallback_to_legacy_for_unsupported_terms: bool = True
    report: bool = True
    common_force_placement: str = "outer_system"


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
    from_irc_scan: bool = False
    n_windows: int | None = None
    center_strategy: str = "evb_mixing"
    mixing_gap_kj_mol: float = 0.0
    basin_extension_kj_mol: float = 0.0
    allow_pathological_irc_windows: bool = False


@dataclass(slots=True)
class ProtonTransferUmbrellaWindows:
    centers_nm: list[float] = field(default_factory=list)
    force_constant_kj_mol_nm2: float | None = None


@dataclass(slots=True)
class MetadynamicsSettings:
    cv: str = "gap"
    min_value: float | None = None
    max_value: float | None = None
    bias_width: float | None = None
    height_kj_mol: float = 1.2
    bias_factor: float = 10.0
    frequency: int = 500
    save_frequency: int | None = 5000
    bias_dir: str = "bias"
    grid_width: int | None = None
    wall_force_constant_kj_mol2: float | None = None
    convergence_check_interval: int | None = None
    convergence_tolerance_kj_mol: float | None = None
    convergence_consecutive_checks: int = 3
    convergence_min_steps: int = 0
    convergence_ts_window_kj_mol: float | None = None
    convergence_min_ts_samples: int = 0
    convergence_min_weight2: float | None = None
    convergence_min_mixing_samples: int = 0
    convergence_mixing_weight2_min: float = 0.2
    convergence_mixing_weight2_max: float = 0.8


@dataclass(slots=True)
class NativeGapBiasSettings:
    method: str = "well_tempered_metadynamics"
    cv: str = "gap"
    min_value: float | None = None
    max_value: float | None = None
    grid_width: int | None = None
    bias_width: float | None = None
    height_kj_mol: float | None = None
    bias_factor: float | None = None
    temperature_k: float | None = None
    frequency: int | None = None
    save_frequency: int | None = None
    bias_dir: str | None = None
    restart: bool = True
    wall_force_constant_kj_mol2: float | None = None
    update_scheme: str = "table_in_context"
    out_of_grid: str = "clamp"


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
    metadynamics: MetadynamicsSettings = field(default_factory=MetadynamicsSettings)
    native_gap_bias: NativeGapBiasSettings = field(default_factory=NativeGapBiasSettings)
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
    derive_regions_from_irc: bool = False
    allow_sign_fallback: bool = False


@dataclass(slots=True)
class IRCRelaxationSettings:
    enabled: bool = False
    mode: str = "mapped"
    platform: str | None = None
    require_platform: bool = True
    pre_relaxation_enabled: bool = True
    solvent_residue_names: list[str] = field(default_factory=lambda: ["HOH", "WAT", "SOL", "TIP3", "TIP3P"])
    solvent_minimization_steps: int = 500
    protein_minimization_steps: int = 500
    pre_relax_mobile_radius_nm: float = 0.8
    pre_relax_nonmobile_restraint_kj_mol_nm2: float = 5000.0
    fix_alpha_carbons: bool = True
    alpha_carbon_restraint_kj_mol_nm2: float = 10000.0
    frame_stride: int = 1
    frame_indices: list[int] = field(default_factory=list)
    minimization_steps: int = 500
    minimization_tolerance_kj_mol_nm: float = 10.0
    mobile_atoms: list[int] = field(default_factory=list)
    mobile_radius_nm: float = 0.45
    restrain_nonmobile: bool = True
    nonmobile_restraint_kj_mol_nm2: float = 2500.0
    irc_atom_restraint_kj_mol_nm2: float = 250.0
    use_relaxed_for_scan: bool = True
    output_subdir: str = "irc_relaxed_seeds"


@dataclass(slots=True)
class IRCSettings:
    path: str | None = None
    order: str = "auto"
    rc_frame: int | str | None = "auto"
    ts_frame: int | str | None = "auto"
    product_frame: int | str | None = "auto"
    embedding: dict[str, Any] = field(default_factory=dict)
    relaxation: IRCRelaxationSettings = field(default_factory=IRCRelaxationSettings)


@dataclass(slots=True)
class ReferenceProfileSettings:
    units: str = "kJ/mol"
    rc: float | None = None
    ts: float | None = None
    product: float | None = None
    source_label: str | None = None


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
    mode: str = "plain"
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
    energy_decomposition: EnergyDecompositionSettings = field(default_factory=EnergyDecompositionSettings)
    evb_representation: str = "full_state"
    q_region: dict[str, Any] = field(default_factory=dict)
    cv: CVDefinition | None = None
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    output_dir: str = "outputs"
    start_state: str = "state1"
    start_coordinates: str | None = None
    project: ProjectSettings = field(default_factory=ProjectSettings)
    reaction: ReactionSettings = field(default_factory=ReactionSettings)
    observables: ObservableSettings = field(default_factory=ObservableSettings)
    sampling: SamplingSettings = field(default_factory=SamplingSettings)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    fit: FitSettings = field(default_factory=FitSettings)
    plumed: PlumedSettings = field(default_factory=PlumedSettings)
    irc: IRCSettings = field(default_factory=IRCSettings)
    reference_profile: ReferenceProfileSettings | None = None


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
    return BarrierSettings(
        reactant_region=reactant_region,
        product_region=product_region,
        derive_regions_from_irc=bool(data.get("derive_regions_from_irc", False)),
        allow_sign_fallback=bool(data.get("allow_sign_fallback", False)),
    )


def _coerce_irc_settings(data: dict[str, Any] | None) -> IRCSettings:
    if not data:
        return IRCSettings()
    relaxation_payload = data.get("relaxation", {})
    return IRCSettings(
        path=data.get("path"),
        order=data.get("order", "auto"),
        rc_frame=data.get("rc_frame", "auto"),
        ts_frame=data.get("ts_frame", "auto"),
        product_frame=data.get("product_frame", "auto"),
        embedding=dict(data.get("embedding", {})),
        relaxation=IRCRelaxationSettings(**relaxation_payload) if relaxation_payload else IRCRelaxationSettings(),
    )


def _coerce_reference_profile(data: dict[str, Any] | None) -> ReferenceProfileSettings | None:
    if not data:
        return None
    return ReferenceProfileSettings(
        units=data.get("units", "kJ/mol"),
        rc=data.get("rc"),
        ts=data.get("ts"),
        product=data.get("product"),
        source_label=data.get("source_label"),
    )


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
        metadynamics=MetadynamicsSettings(),
        seed_windows=[],
    )
    return EVBConfig(
        state1=state1,
        state2=state2,
        calibration=calibration,
        evb_parameters=evb_parameters,
        energy_decomposition=EnergyDecompositionSettings(),
        evb_representation=data.get("evb", {}).get("representation", "full_state"),
        q_region=dict(data.get("evb", {}).get("q_region", {})),
        cv=cv,
        simulation=simulation,
        output_dir=project.output_dir,
        start_state=data.get("start_state", "state1"),
        start_coordinates=data.get("start_coordinates"),
        project=project,
        reaction=reaction,
        observables=observables,
        sampling=sampling,
        irc=IRCSettings(),
        reference_profile=None,
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
    energy_decomposition = EnergyDecompositionSettings(**evb_payload.get("energy_decomposition", {}))
    evb_representation = evb_payload.get("representation", "q_region" if evb_payload.get("q_region", {}).get("enabled") else "full_state")
    q_region = dict(evb_payload.get("q_region", {}))

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
        metadynamics=MetadynamicsSettings(**sampling_payload.get("metadynamics", {})),
        native_gap_bias=NativeGapBiasSettings(**sampling_payload.get("native_gap_bias", {})),
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
    irc = _coerce_irc_settings(data.get("irc"))
    reference_profile = _coerce_reference_profile(data.get("reference_profile"))
    _validate_modern_payload(data, evb_payload, coupling_payload)

    return EVBConfig(
        state1=state1,
        state2=state2,
        calibration=None,
        evb_parameters=evb_parameters,
        energy_decomposition=energy_decomposition,
        evb_representation=evb_representation,
        q_region=q_region,
        cv=cv,
        simulation=simulation,
        output_dir=project.output_dir,
        start_state=data.get("start_state", "state1"),
        start_coordinates=data.get("start_coordinates"),
        project=project,
        reaction=reaction,
        observables=observables,
        sampling=sampling,
        analysis=analysis,
        fit=fit,
        plumed=plumed,
        irc=irc,
        reference_profile=reference_profile,
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
    if config.energy_decomposition.enabled and config.energy_decomposition.mode not in {"exact", "legacy"}:
        errors.append("evb.energy_decomposition.mode must be 'exact' or 'legacy'.")
    if config.energy_decomposition.common_force_placement not in {"outer_system", "cv_compatible"}:
        errors.append("evb.energy_decomposition.common_force_placement must be 'outer_system' or 'cv_compatible'.")
    if config.evb_representation not in {"full_state", "q_region"}:
        errors.append("evb.representation must be 'full_state' or 'q_region'.")
    if config.evb_representation == "q_region":
        q_atoms = config.q_region.get("q_atoms") or []
        if not q_atoms and not config.q_region.get("q_atoms_from_reaction", False):
            errors.append("q_region representation requires evb.q_region.q_atoms or q_atoms_from_reaction.")
    if config.sampling.mode not in {"mapping", "gap_umbrella", "proton_transfer_umbrella", "gap_metadynamics", "gap_table_metadynamics"}:
        errors.append(f"Unsupported sampling.mode {config.sampling.mode!r}.")
    if config.sampling.mode == "mapping" and not config.sampling.windows.mapping.lambda_values:
        errors.append("mapping mode requires sampling.windows.mapping.lambda_values.")
    if config.sampling.mode == "gap_umbrella":
        if not config.sampling.windows.gap_umbrella.centers_kj_mol and not config.sampling.windows.gap_umbrella.from_irc_scan:
            errors.append("gap_umbrella mode requires sampling.windows.gap_umbrella.centers_kj_mol.")
        if config.sampling.windows.gap_umbrella.force_constant_kj_mol2 is None:
            errors.append("gap_umbrella mode requires sampling.windows.gap_umbrella.force_constant_kj_mol2.")
    if config.plumed.enabled and not (config.plumed.script or config.plumed.script_file):
        errors.append("plumed.enabled requires plumed.script or plumed.script_file.")
    if config.plumed.mode not in {"plain", "metad", "opes", "opes_metad"}:
        errors.append("plumed.mode must be one of: plain, metad, opes, opes_metad.")
    if config.sampling.mode == "gap_metadynamics":
        meta = config.sampling.metadynamics
        if meta.cv != "gap":
            errors.append("gap_metadynamics currently supports sampling.metadynamics.cv: gap only.")
        if meta.min_value is None or meta.max_value is None or meta.bias_width is None:
            errors.append("gap_metadynamics requires min_value, max_value, and bias_width in kJ/mol.")
    if config.sampling.mode == "gap_table_metadynamics":
        native = resolve_native_gap_bias_settings(config)
        if native.method != "well_tempered_metadynamics":
            errors.append("gap_table_metadynamics currently supports native_gap_bias.method: well_tempered_metadynamics only.")
        if native.cv != "gap":
            errors.append("gap_table_metadynamics currently supports native_gap_bias.cv: gap only.")
        if native.min_value is None or native.max_value is None or native.bias_width is None:
            errors.append("gap_table_metadynamics requires min_value, max_value, and bias_width in kJ/mol.")
        if native.grid_width is None:
            errors.append("gap_table_metadynamics requires native_gap_bias.grid_width or metadynamics.grid_width.")
        if native.update_scheme != "table_in_context":
            errors.append("native_gap_bias.update_scheme must be 'table_in_context'.")
        if native.out_of_grid not in {"clamp", "reject"}:
            errors.append("native_gap_bias.out_of_grid must be 'clamp' or 'reject'.")
    return errors


def resolve_native_gap_bias_settings(config: EVBConfig) -> NativeGapBiasSettings:
    native = config.sampling.native_gap_bias
    meta = config.sampling.metadynamics
    return NativeGapBiasSettings(
        method=native.method,
        cv=native.cv or meta.cv,
        min_value=native.min_value if native.min_value is not None else meta.min_value,
        max_value=native.max_value if native.max_value is not None else meta.max_value,
        grid_width=native.grid_width if native.grid_width is not None else meta.grid_width,
        bias_width=native.bias_width if native.bias_width is not None else meta.bias_width,
        height_kj_mol=native.height_kj_mol if native.height_kj_mol is not None else meta.height_kj_mol,
        bias_factor=native.bias_factor if native.bias_factor is not None else meta.bias_factor,
        temperature_k=native.temperature_k if native.temperature_k is not None else config.simulation.temperature_k,
        frequency=native.frequency if native.frequency is not None else meta.frequency,
        save_frequency=native.save_frequency if native.save_frequency is not None else meta.save_frequency,
        bias_dir=native.bias_dir if native.bias_dir is not None else meta.bias_dir,
        restart=native.restart,
        wall_force_constant_kj_mol2=(
            native.wall_force_constant_kj_mol2
            if native.wall_force_constant_kj_mol2 is not None
            else meta.wall_force_constant_kj_mol2
        ),
        update_scheme=native.update_scheme,
        out_of_grid=native.out_of_grid,
    )


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
        "irc",
        "reference_profile",
        "start_state",
        "start_coordinates",
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
