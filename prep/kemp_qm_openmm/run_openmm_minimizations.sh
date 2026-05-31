#!/usr/bin/env bash
source "$HOME/.bashrc"
conda activate OpenMM
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p 06_minimized_states/RC 06_minimized_states/PROD
rm -f 06_minimized_states/RC/* 06_minimized_states/PROD/*
python ../../scripts/equilibrate_solvated_state_openmm.py   --prmtop 03_solvated_rc/RC_solvated_initial.prmtop   --inpcrd 03_solvated_rc/RC_solvated_initial.inpcrd   --coordinates-pdb 03_solvated_rc/RC_solvated_initial.pdb   --output-dir 06_minimized_states/RC   --prefix RC   --minimize-only   --restraint-k-kcal-a2 250.0
python ../../scripts/equilibrate_solvated_state_openmm.py   --prmtop 03_solvated_prod/PROD_solvated_initial.prmtop   --inpcrd 03_solvated_prod/PROD_solvated_initial.inpcrd   --coordinates-pdb 03_solvated_prod/PROD_solvated_initial.pdb   --output-dir 06_minimized_states/PROD   --prefix PROD   --minimize-only   --restraint-k-kcal-a2 250.0

PYTHONPATH=../../src python ../../scripts/build_openmm_bundle_from_amber.py   --prmtop 03_solvated_rc/RC_solvated_initial.prmtop   --coordinates 06_minimized_states/RC/RC_minimized.pdb   --output-dir 07_openmm_bundles/RC   --nonbonded-method PME   --constraints None

PYTHONPATH=../../src python ../../scripts/build_openmm_bundle_from_amber.py   --prmtop 03_solvated_prod/PROD_solvated_initial.prmtop   --coordinates 06_minimized_states/PROD/PROD_minimized.pdb   --output-dir 07_openmm_bundles/PROD   --nonbonded-method PME   --constraints None
