# Migration from prototype workflows

Earlier development branches contained system-specific prototype modules. This mainline architecture moves those prototypes out of `src/kemp_evb` and exposes generic modules: `reference_profile`, `profile_calibration`, `frame_mapping`, `qregion_candidates`, `sampling_workflow`, `umbrella_analysis`, `metad_analysis`, and `reproduction_workflow`. Future development should target these generic modules.
