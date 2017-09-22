#!/usr/bin/env python

"""
Adds SEQRES records to a PDB file.

Useful together with `pdbfixer` to add termini to structures.
"""

import argparse
from collections import OrderedDict

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('structure', help='PDB file')
ap.add_argument('--output', default=None, help='Output file name')
ap.add_argument('--termini', action='store_true', help='Adds ACE/NME residues at chain termini')
cmd = ap.parse_args()


structure = OrderedDict()
lines = []
with open(cmd.structure) as handle:
    for line in handle:
        if not line.startswith(('ATOM', 'HETATM', 'MODEL', 'ENDMDL', 'END', 'TER')):
            continue

        lines.append(line)
        if line.startswith('ATOM'):
            # Parse res ids
            chain_id = line[21]
            resn = line[17:20]
            resi = line[22:26]
            res_uid = (resn, resi)

            if chain_id not in structure:
                structure[chain_id] = [res_uid]
            else:
                if res_uid != structure[chain_id][-1]:
                    structure[chain_id].append(res_uid)

# SEQRES Records
# 13 residues per line
_fmt = "SEQRES {:3d} {:1s} {:4d}  {}          "
seqres = []
for chain in structure:
    if cmd.termini:
        structure[chain].insert(0, ('ACE', -1))
        structure[chain].append(('NME', -1))

    n_seqres = 1
    
    res_list = structure[chain]
    n_res = len(res_list)
    for idx in range(0, n_res, 13):
        aa_list = [aa for aa, num in res_list[idx:idx+13]]
        aa_str = ' '.join(aa_list)
        ln = _fmt.format(n_seqres, chain, n_res, aa_str)
        n_seqres += 1
        seqres.append(ln)

seqres = '\n'.join(seqres) + '\n'

if cmd.output is not None:
    fname = cmd.output
else:
    fname = cmd.structure[:-4] + 'seqRes.pdb'

with open(fname, 'w') as handle:
    handle.write(seqres)
    handle.write(''.join(lines))
