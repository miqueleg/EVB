# HG3.17 example dataset

This directory is an example/tutorial dataset for the generic EVB workflow. The reference method label in `reference_profile.yaml` is dataset metadata only; no package source code depends on this system or method.

Generic workflow sketch:

```bash
.venv/bin/evb validate-workflow-config --config examples/hg317/qregion_config.yaml
.venv/bin/evb make-qregion-candidates --config examples/hg317/qregion_config.yaml --output outputs/example_qregion_candidates
.venv/bin/evb run-workflow --config examples/hg317/qregion_config.yaml --reference examples/hg317/reference_profile.yaml --output outputs/example_workflow --mode pilot
```
