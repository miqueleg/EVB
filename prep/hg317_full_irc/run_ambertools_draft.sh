#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}"
source /home/mestevez/Programs/Amber24_AT25/ambertools25/amber.sh

cd "$(dirname "$0")"

# Draft ligand parameterization. Charges must be reviewed for the actual Kemp states.
antechamber -i irc_reactant_ligand_state1.pdb -fi pdb -o state1_reactant_ligand.mol2 -fo mol2 -at gaff2 -c bcc -s 2 -nc 0
parmchk2 -i state1_reactant_ligand.mol2 -f mol2 -o state1_reactant_ligand.frcmod -s gaff2

antechamber -i irc_product_ligand_state2.pdb -fi pdb -o state2_product_ligand.mol2 -fo mol2 -at gaff2 -c bcc -s 2 -nc -1
parmchk2 -i state2_product_ligand.mol2 -f mol2 -o state2_product_ligand.frcmod -s gaff2

antechamber -i irc_product_ligand_state2_matched16.pdb -fi pdb -o state2_product_ligand_matched16.mol2 -fo mol2 -at gaff2 -c bcc -s 2 -nc 0
parmchk2 -i state2_product_ligand_matched16.mol2 -f mol2 -o state2_product_ligand_matched16.frcmod -s gaff2

echo "Draft GAFF2/BCC ligand files generated. Do not run EVB until transferred-proton topology is resolved and state compatibility is validated."
