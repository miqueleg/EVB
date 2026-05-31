from __future__ import annotations

from pathlib import Path

from kemp_evb.config import load_config


def test_loads_modern_yaml_config(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project:
  name: kemp-solution-baseline
  output_dir: outputs/test
reaction:
  metadata:
    name: acetate_kemp
    phase: solution
    temperature_k: 300.0
  atoms:
    donor: 9
    proton: 22
    acceptor: 15
states:
  state1:
    topology: data/state1.prmtop
    coordinates: data/state1.inpcrd
  state2:
    topology: data/state2.prmtop
    coordinates: data/state2.inpcrd
evb:
  coupling_model:
    parameters:
      delta_alpha_kj_mol: 1.5
      h12_kj_mol: 3.0
sampling:
  mode: mapping
  seed_windows:
    - window_id: p024
      coordinates: outputs/seeds/ts_like.pdb
      branch: forward
  md:
    production_steps: 500
    report_stride: 25
observables:
  reaction_coordinates:
    - name: proton_transfer_rc
      kind: difference_of_distances
      atom1: 9
      atom2: 22
      atom3: 15
      event_threshold_nm: 0.0
    - name: ring_opening_rc
      kind: distance
      atom1: 1
      atom2: 11
  distances:
    - name: donor_h
      atom1: 9
      atom2: 22
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.project.name == "kemp-solution-baseline"
    assert config.output_dir == "outputs/test"
    assert config.state1.prmtop == "data/state1.prmtop"
    assert config.evb_parameters.delta_alpha == 1.5
    assert config.evb_parameters.h12 == 3.0
    assert config.simulation.steps == 500
    assert config.observables.distances[0].name == "donor_h"
    assert [item.name for item in config.observables.reaction_coordinates] == ["proton_transfer_rc", "ring_opening_rc"]
    assert config.sampling.seed_windows[0].window_id == "p024"
    assert config.sampling.seed_windows[0].branch == "forward"
