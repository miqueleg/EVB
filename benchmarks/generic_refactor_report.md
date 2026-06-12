# Generic EVB workflow refactor report

## Summary

The workflow layer has been generalized from system-specific prototypes into method- and system-agnostic modules. Source code no longer contains hardcoded system names, reference-method names, or selected candidate names.

## Generic modules

- `reference_profile.py`
- `profile_calibration.py`
- `frame_mapping.py`
- `qregion_candidates.py`
- `sampling_workflow.py`
- `umbrella_analysis.py`
- `metad_analysis.py`
- `reproduction_workflow.py`

## Generic commands

- `validate-workflow-config`
- `make-qregion-candidates`
- `calibrate-profile`
- `derive-windows`
- `run-umbrella`
- `analyze-umbrella`
- `run-metad`
- `analyze-metad`
- `run-workflow`
- `compare-profile`
- `write-run-scripts`

## Validation

The hardcoding scan target is:

```bash
rg -n "HG3|hg317|g-xtb|gxtb|local_pme_q_atoms_cutoff_0\.8" src/kemp_evb
```

Expected result: no matches.

## Next branch

After long PMF/metadynamics reproduction data are available through the generic workflow, native OPES should be implemented generically on top of `sampling_workflow` and native table-bias infrastructure.
