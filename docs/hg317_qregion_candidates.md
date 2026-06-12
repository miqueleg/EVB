# HG3.17 Q-Region Candidate Hamiltonians

HG3.17 cannot use exact Q-region decomposition of differing endpoint PME `NonbondedForce` objects because PME reciprocal-space electrostatics are global. This workflow makes the system computable by generating explicit candidate Hamiltonians and validating them against the slow full-state EVB reference.

## Candidate Modes

`full_state_reference` is the legacy two-endpoint EVB model. It remains the validation reference and may keep duplicated full PME.

`shared_nonbonded_model` uses one baseline PME `NonbondedForce` in `E_common` and ignores endpoint-specific nonbonded differences in `e1_Q/e2_Q`. It is exact for a changed Hamiltonian, not equivalent to the legacy endpoint-charge model. Its reports set `exactness_status: exact_for_shared_nonbonded_model`, `legacy_equivalence: false`, and `nonbonded_model_changed: true`. Use it only after Q-region EVB parameters have been refit and the resulting barriers/scans are physically sane.

`local_pme_approx` uses one baseline PME force plus local direct-space residual corrections for selected Q/correction atom interactions. It is approximate relative to full-state PME because reciprocal-space differences are not decomposed. Its reports set `exactness_status: approximate`, `pme_approximation: true`, and `legacy_equivalence: approximate_only`.

## One-Command Workflow

```bash
python scripts/benchmark_hg317_qregion_candidates.py \
  --config examples/hg317_evb_gap_metad.yaml \
  --output outputs/hg317_qregion_candidates \
  --platform CUDA \
  --run-smoke \
  --smoke-steps 2000
```

The equivalent CLI steps are:

```bash
evb make-hg317-qregion-candidates \
  --config examples/hg317_evb_gap_metad.yaml \
  --output outputs/hg317_qregion_candidates \
  --include-reaction-atoms

evb q-region-fit-irc \
  --config outputs/hg317_qregion_candidates/candidates/shared_nonbonded_state1.yaml \
  --output outputs/hg317_qregion_candidates/fits/shared_nonbonded_state1

evb validate-hg317-qregion-candidates \
  --candidate-dir outputs/hg317_qregion_candidates \
  --output outputs/hg317_qregion_candidates/validation \
  --platform CUDA \
  --run-smoke \
  --smoke-steps 2000
```

## Outputs

Candidate generation writes `q_region_derivation_report.json` and YAMLs under `outputs/hg317_qregion_candidates/candidates/`. Validation writes `candidate_validation_summary.json`, `.csv`, `.md`, and full per-candidate JSON files under `validation/per_candidate/`.

The smoke benchmark runs only if a candidate passes the validation/refit criteria, unless `--force-smoke` is given. Forced smoke reports `forced_despite_failed_validation: true`.

## Interpreting Results

A computable candidate is not automatically production-ready. Shared-nonbonded candidates must be interpreted as changed Hamiltonians and require refit. Local-PME candidates must satisfy energy, gap, and force thresholds versus the full-state reference. If no candidate passes, the workflow reports the best failed candidate and skips MD.
