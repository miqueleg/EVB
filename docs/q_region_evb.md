# Q-Region EVB

Full-state EVB builds two complete endpoint systems and mixes their full energies. That is exact, but it duplicates expensive work when both states contain full protein/solvent nonbonded forces. Exact decomposition separates common terms when possible, but if the two PME `NonbondedForce` objects differ it still keeps both full PME forces and reports `duplicated_full_nonbonded: true`.

Q-region EVB follows the Q6-style idea: evaluate a shared environment/baseline once, then mix only state-specific Q-region residual energies. The implemented force form is:

```text
E_total = E_common + lower_EVB(e1_Q, e2_Q, delta_alpha, h12)
gap = e1_Q - e2_Q - delta_alpha
```

With native table bias:

```text
E_total = E_common + lower_EVB(e1_Q, e2_Q, delta_alpha, h12) + gap_bias(gap)
```

`E_common` is added to the outer OpenMM `System`; the EVB `CustomCVForce` contains only `e1_Q`, `e2_Q`, and optional `gap_bias(gap)`. This keeps the architecture compatible with `evb-gap-table-metad` and with future native OPES/OPES-Vk.

## Modes

| Mode | Full-state duplicated PME? | Exact? | Intended use |
| --- | --- | --- | --- |
| legacy full-state | yes when both states have full PME | yes | reference |
| exact decomposition | maybe | yes | audit/reference |
| q_region exact direct-space | no | yes for supported direct-space systems | production where validated |
| q_region local_pme_approx | no | no | experimental, validated only |
| q_region table-metad | depends on q_region policy | depends on q_region policy | enhanced sampling |

## Nonbonded Policies

`exact_identical_nonbonded`: identical `NonbondedForce` objects are placed once in `E_common`.

`exact_direct_nonbonded`: for direct-space `NoCutoff`, `CutoffNonPeriodic`, or `CutoffPeriodic`, one baseline `NonbondedForce` is evaluated once and Q-region pairwise corrections are added to the state residual. This is tested against legacy on toy systems.

`local_pme_approx`: disabled by default. Differing PME is not decomposed exactly because reciprocal-space terms are global. If explicitly enabled, Q-region uses a baseline PME force plus local direct-space corrections and reports `exactness_status: approximate`, `pme_approximation: true`, and `reciprocal_pme_difference_ignored_or_approximated: true`. Production use requires validation against legacy.

## Commands

Derive a proposal:

```bash
evb derive-q-region \
  --config examples/hg317_evb_gap_metad.yaml \
  --output outputs/hg317_q_region_derivation \
  --include-reaction-atoms
```

Validate a Q-region config:

```bash
evb q-region-singlepoint \
  --config outputs/hg317_q_region_derivation/hg317_q_region_config.yaml \
  --output outputs/hg317_q_region_validation
```

Run native table metadynamics with Q-region representation:

```bash
evb q-region-gap-table-metad --config q_region_config.yaml
```

`evb-gap-table-metad` also uses Q-region automatically when the config says `evb.representation: q_region`. Neither path uses OpenMM `app.Metadynamics` or `BiasVariable`.

## Config Sketch

```yaml
evb:
  representation: q_region
  q_region:
    enabled: true
    q_atoms: [0, 1]
    baseline_state: state1
    changed_atom_policy: require_subset
    common_force_placement: outer_system
    nonbonded:
      local_approx_enabled: false
```

For HG3.17, do not silently guess production Q atoms. `derive-q-region` writes a report and a config proposal that require review.

## Limitations

Exact Q-region PME decomposition is not implemented. Local PME correction is approximate and disabled by default. Unsupported changed bonded term topologies fail rather than falling back to duplicated full-state systems.
