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

fixer = PDBFixer(cmd.structure)
fixer.findMissingResidues()
fixer.findMissingAtoms()
fixer.addMissingAtoms()
#fixer.addMissingHydrogens(7.0)

if cmd.output is None:
    ofname = cmd.structure[:-4]
else:
    ofname = cmd.output

with open('{}.cif'.format(ofname), 'w') as handle:
    PDBxFile.writeFile(fixer.topology, fixer.positions, handle)
