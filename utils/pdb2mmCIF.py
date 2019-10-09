#!/usr/bin/env python

"""
Converts a PDB file to mmCIF format, 
adding missing heavy atoms in the process.
"""

import argparse

from pdbfixer import PDBFixer
from simtk.openmm.app import PDBxFile

ap = argparse.ArgumentParser()
ap.add_argument('structure')
ap.add_argument('--output', default=None, help='Output name for fixed structure')
cmd = ap.parse_args()

print('Reading PDB structure: {}'.format(cmd.structure))
fixer = PDBFixer(cmd.structure)

print('Finding and adding missing heavy atoms')
fixer.findMissingResidues()
fixer.findMissingAtoms()
fixer.addMissingAtoms()

if cmd.output is None:
    ofname = cmd.structure[:-4] + '.cif'
else:
    ofname = cmd.output

print('Writing completed structure in CIF format: {}'.format(ofname))
with open('{}'.format(ofname), 'w') as handle:
    PDBxFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
