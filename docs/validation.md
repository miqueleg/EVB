# EVB Validation Protocol

Minimum checks before trusting a barrier:

1. Validate both diabatic states with `evb validate --config config.yaml`.
2. Confirm single-point OpenMM EVB energies match the analytical Python Hamiltonian.
3. Confirm finite-difference forces on a small representative system.
4. Verify `gap = E1 - E2 - delta_alpha` is logged and has overlapping distributions across adjacent windows.
5. Run conventional mapped or gap-umbrella sampling before adding PLUMED.
6. Inspect overlap matrices, blocking/statistical inefficiency estimates, and independent replicates.
7. Fit `delta_alpha` and constant `H12` against documented reference targets.
8. Report uncertainty on reaction free energy and barrier. Treat diagnostic histogram PMFs as qualitative until MBAR/WHAM convergence is demonstrated.

PLUMED/OPES can help sample geometrical CVs, but it does not automatically validate the EVB Hamiltonian and currently cannot see the internal EVB energy gap without an explicit bridge.
