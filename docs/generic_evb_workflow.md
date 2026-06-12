# Generic EVB workflow

The package is organized around user-supplied diabatic states, an arbitrary reference profile, optional frame/IRC mapping, an EVB representation, and a sampling workflow.

Typical sequence:

```bash
evb validate-workflow-config --config config.yaml
evb calibrate-profile --reference reference.yaml --coords frame_energies.csv --output outputs/calibration
evb derive-windows --config config.yaml --output outputs/windows
evb run-workflow --config config.yaml --output outputs/workflow --workflow all --mode pilot
evb compare-profile --reference reference.yaml --output outputs/comparison
```

`full_state` remains the exact endpoint-state reference. `q_region` is the scalable representation and can use exact direct-space, shared nonbonded, or explicitly approximate local PME nonbonded policies.
