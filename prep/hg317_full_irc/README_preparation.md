# HG3.17 Full-System IRC Preparation

This directory contains cleaned 5RGE-derived model inputs and IRC-derived ligand seeds.

Important: these are not final EVB E1/E2 systems yet. The reactant and product states differ by a proton transfer between the ligand and the catalytic protein fragment. To make a valid two-state EVB system, both states must contain the same atoms in the same order and masses, with different bonding/charges only in the diabatic topologies.

The current generated model is enough to start AmberTools preparation, but the transferred proton H197 needs a custom topology decision:

- state 1/reactant: H197 is on the substrate-like ligand.
- state 2/product: H197 is on the catalytic-base/protein fragment.

Standard ff14SB + GAFF2 cannot guarantee identical atom ordering across that proton-transfer topology unless we create matched custom residues or manually reorder/validate the prmtops.
