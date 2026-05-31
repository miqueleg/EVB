#!/usr/bin/env bash
source "$HOME/.bashrc"
conda activate ambertraj
set -euo pipefail
export PATH="$CONDA_PREFIX/bin:$PATH"

cd "$(dirname "$0")"
mkdir -p 02_ambertools 03_solvated_rc 03_solvated_prod 05_templates 06_minimized_states

antechamber -i 01_qm_inputs/rc_substrate_simple.pdb -fi pdb -o 02_ambertools/rc_substrate_raw.mol2 -fo mol2 -c bcc -s 2 -nc 0
python ../../scripts/rename_kemp_qm_mol2.py --input 02_ambertools/rc_substrate_raw.mol2 --output 02_ambertools/rc_substrate_named.mol2 --resname SBR --names-file 01_qm_inputs/rc_substrate.names
parmchk2 -i 02_ambertools/rc_substrate_named.mol2 -f mol2 -o 02_ambertools/rc_substrate.frcmod

antechamber -i 01_qm_inputs/rc_partner_simple.pdb -fi pdb -o 02_ambertools/rc_partner_raw.mol2 -fo mol2 -c bcc -s 2 -nc -1
python ../../scripts/rename_kemp_qm_mol2.py --input 02_ambertools/rc_partner_raw.mol2 --output 02_ambertools/rc_partner_named.mol2 --resname BAR --names-file 01_qm_inputs/rc_partner.names
parmchk2 -i 02_ambertools/rc_partner_named.mol2 -f mol2 -o 02_ambertools/rc_partner.frcmod

antechamber -i 01_qm_inputs/prod_substrate_simple.pdb -fi pdb -o 02_ambertools/prod_substrate_raw.mol2 -fo mol2 -c bcc -s 2 -nc -1
python ../../scripts/rename_kemp_qm_mol2.py --input 02_ambertools/prod_substrate_raw.mol2 --output 02_ambertools/prod_substrate_named.mol2 --resname SDP --names-file 01_qm_inputs/prod_substrate.names
parmchk2 -i 02_ambertools/prod_substrate_named.mol2 -f mol2 -o 02_ambertools/prod_substrate.frcmod

antechamber -i 01_qm_inputs/prod_partner_simple.pdb -fi pdb -o 02_ambertools/prod_partner_raw.mol2 -fo mol2 -c bcc -s 2 -nc 0
python ../../scripts/rename_kemp_qm_mol2.py --input 02_ambertools/prod_partner_raw.mol2 --output 02_ambertools/prod_partner_named.mol2 --resname ACP --names-file 01_qm_inputs/prod_partner.names
parmchk2 -i 02_ambertools/prod_partner_named.mol2 -f mol2 -o 02_ambertools/prod_partner.frcmod

python ../../scripts/extract_solvent_template_from_pdb.py   --input-pdb ../../systems/KEMP-solvent/RC.pdb   --output-pdb 05_templates/solvent_only_from_old_rc.pdb   --reactive-atom-count 23

python ../../scripts/build_kemp_prod_from_rc_template.py   --rc-master-pdb ../../systems/KEMP-solvent/RC.pdb   --rc-reactive-pdb 01_qm_inputs/rc_complex_fragmented.pdb   --prod-reactive-pdb 01_qm_inputs/prod_complex_fragmented.pdb   --rc-output-pdb 05_templates/rc_in_old_rc_frame_fragmented.pdb   --prod-output-pdb 05_templates/prod_in_old_rc_frame_fragmented.pdb   --reactive-atom-count 23

tleap -f tleap_rc_from_previous_solvent.in
tleap -f tleap_prod_from_previous_solvent.in

echo 'Initial solvated QM-guided states are under 03_solvated_rc/ and 03_solvated_prod/.'
