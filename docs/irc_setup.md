# EVB setup from cluster-model IRC

The cluster-model IRC multi-XYZ is a geometry path reference. It is used to assign chemically meaningful RC, TS, and product geometries, diagnose the diabatic energy gap, fit `delta_alpha` and constant `H12`, and seed later EVB sampling windows.

It is not the EVB free-energy profile. Do not report barriers directly from IRC XYZ comment energies.

## Required inputs

The IRC XYZ may have no energies in the comment lines. That is valid. Reference thermodynamic values must be supplied separately:

```yaml
irc:
  path: inputs/hg317_irc.xyz
  order: prod_ts_rc
  rc_frame: auto
  ts_frame: auto
  product_frame: auto

reference_profile:
  units: kcal/mol
  rc: 0.0
  ts: 18.5
  product: 7.2
  source_label: "CM g-xTB + frequency correction"
```

Supported `irc.order` values are `rc_ts_prod`, `prod_ts_rc`, and `auto`. Use `prod_ts_rc` when the input file is ordered product to TS to reactant. Downstream code always sees the canonical order RC to TS to product.

Explicit `rc_frame`, `ts_frame`, and `product_frame` values are original 0-based XYZ frame indices. Explicit indices override comment labels and automatic defaults.

## Cluster-model embedding

If the IRC is a cluster model rather than a full-system coordinate path, enable embedding. The fixed alpha carbons are used as anchors to place the cluster path into the full OpenMM coordinate frame. The code then maps same-element cluster atoms onto nearby OpenMM atoms and inserts only those mapped atoms into the full-system coordinate array before evaluating E1/E2.

```yaml
irc:
  path: examples/HD3.17_IRC.xyz
  order: prod_ts_rc
  embedding:
    enabled: true
    alpha_carbon_mapping: outputs/hg317_cm_irc/hg317_5rge_irc_ca_mapping.yaml
    max_anchor_error_angstrom: 0.5
    max_match_angstrom: 1.25
    include_hydrogens: true
    auto_match: false
    irc_to_openmm:
      6: 4540
      7: 4541
      62: 4542
      63: 4543
      64: 4544
      65: 4545
      66: 4546
      67: 4547
      68: 4548
      81: 4549
      82: 4550
      83: 4551
      194: 4552
      195: 4553
      196: 4554
      197: 4555
```

Explicit `irc_to_openmm` entries are original cluster atom indices to OpenMM atom indices, both 0-based. Use them for chemically important atoms, especially the matched ligand/proton atoms where nearest-neighbor matching can be ambiguous.

For an active-site cluster cut from a protein, `auto_match: false` is usually safer. The fixed alpha carbons then define the coordinate transform, but protein side-chain/backbone fragment atoms are not transplanted into the intact OpenMM protein. Moving those fragments without moving all bonded neighbors can create artificial bond/angle strain and enormous diabatic energies.

## IRC-seeded relaxation

Raw cluster-model coordinates should normally be relaxed in the full OpenMM environment before using them for EVB fitting or window generation. Enable `irc.relaxation` to convert each embedded IRC frame into a restrained full-system seed. The default protocol is designed for general enzyme active-site clusters with fixed alpha carbons:

1. Minimize solvent while restraining the solute.
2. Minimize the local protein/active-site environment while restraining far-field atoms and alpha carbons.
3. Use that pre-relaxed full-system structure as the base for all IRC-frame insertions.
4. Run short per-frame mapped-Hamiltonian minimizations for the embedded IRC geometries.

```yaml
irc:
  relaxation:
    enabled: true
    mode: mapped
    platform: CUDA
    require_platform: true
    pre_relaxation_enabled: true
    solvent_minimization_steps: 500
    protein_minimization_steps: 500
    pre_relax_mobile_radius_nm: 0.8
    fix_alpha_carbons: true
    alpha_carbon_restraint_kj_mol_nm2: 10000.0
    frame_stride: 1
    minimization_steps: 75
    minimization_tolerance_kj_mol_nm: 10.0
    mobile_radius_nm: 0.55
    restrain_nonmobile: true
    nonmobile_restraint_kj_mol_nm2: 2500.0
    irc_atom_restraint_kj_mol_nm2: 500.0
    use_relaxed_for_scan: true
    output_subdir: irc_relaxed_seeds
```

The pre-relaxation stage is done once and written under `analysis/irc_relaxed_seeds/pre_relaxation/`. Per-frame relaxation then uses a mapped Hamiltonian with `lambda = canonical_frame/(n_frames-1)`, so reactant-side seeds relax mostly under state 1 and product-side seeds relax mostly under state 2. The explicit IRC-mapped atoms and nearby atoms are mobile; the far field is restrained to the pre-relaxed OpenMM coordinates. Relaxed full-system PDB files and a relaxation CSV are written under `analysis/irc_relaxed_seeds/`.

Set `platform: CUDA` and keep `require_platform: true` for production setup. If CUDA is not usable, setup fails instead of silently running the relaxations on CPU.

This is a seed-preparation step only. After relaxation, inspect the shifted-gap scan again. If the gap is still discontinuous or enormous, the diabatic topology/atom mapping is still not suitable for production EVB sampling.

## Setup command

```bash
evb setup-from-irc --config config.yaml --write-window-config
```

The command:

- reads and validates the multi-XYZ path;
- canonicalizes the order to RC to TS to product;
- evaluates every canonical frame under both diabatic OpenMM states;
- writes `analysis/irc_diabatic_scan_prefit.csv`;
- fits `delta_alpha` and constant `H12` from labeled RC/TS/PROD frames;
- writes `analysis/irc_diabatic_scan.csv`;
- writes `analysis/evb_reference_fit_from_irc.json`;
- optionally writes an IRC-derived gap umbrella window proposal.

It exits before expensive MD sampling.

## EVB-ready inputs from AMBER prmtops

Ordinary endpoint `.prmtop` files are not always EVB-ready. Around a bond-breaking/bond-forming coordinate, one state can evaluate the other state's endpoint geometry as a severe nonbonded overlap. For proton transfer, the transferred H may be bonded and nonbonded-excluded in the product state, but an ordinary reactant prmtop may treat that same short product contact as a full Lennard-Jones pair.

Prepare EVB-ready OpenMM bundles before IRC fitting:

```bash
evb prepare-evb-inputs \
  --config config.yaml \
  --output prep/my_system/evb_ready
```

The command loads the AMBER states, infers reactive atoms from `reaction.atoms`, mirrors zero nonbonded exclusions around those atoms across both states, and writes:

- `state1_evb_ready/system.xml`
- `state1_evb_ready/coordinates.pdb`
- `state2_evb_ready/system.xml`
- `state2_evb_ready/coordinates.pdb`
- `evb_ready_input_report.json`
- a derived YAML config that points to the generated `format: openmm` bundles

The original `.prmtop` files are not modified. Re-run `setup-from-irc` with the derived YAML and refit `delta_alpha`/`H12`, because changing reactive exclusions changes the diabatic reference energies.

## Checks before sampling

Inspect the post-fit scan and JSON warnings before launching production:

- shifted gap should vary smoothly along the canonical path;
- the TS should be near a crossing/mixing region;
- TS EVB weights should not be endpoint-like;
- adjacent gap jumps should not be enormous;
- RC and product should not be swapped;
- fitted barrier should not simply equal the reaction free energy.

Only after these checks should you run EVB sampling. The final activation free energy must come from EVB sampling and WHAM/MBAR-style reconstruction over the energy-gap coordinate with good window overlap.
