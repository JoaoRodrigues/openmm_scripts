#!/usr/bin/env python

"""
Adds solvent molecules (and optionally counter-ions) to a molecular system.

Saves the solvated structure in mmCIF format.

.2017. joaor@stanford.edu
"""

from __future__ import print_function, division

import argparse
import logging
import os
import random
import sys

import simtk.openmm.app as app
import simtk.openmm as mm
import simtk.unit as units

import _utils

# Format logger
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='[%(asctime)s] %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')

##
# Parse user input and options
ap = argparse.ArgumentParser(description=__doc__)

# Mandatory
ap.add_argument('structure', help='Input coordinate file (.cif or .pdb)')
# Options
ap.add_argument('--output', type=str, default=None,
                help='File name for solvated system. Will *always* use mmCIF format.')
ap.add_argument('--forcefield', type=str, default='amber14-all.xml',
                help='Force field to build the system with.')
ap.add_argument('--solvent', type=str, default='amber14/tip3p.xml',
                help='Solvent model to use in minimization.')
ap.add_argument('--neutralize', action='store_true',
                help='Adds counter-ions to neutralize total system charge.')
ap.add_argument('--ionic-strength', type=float, default=0.15,
                help='Molar concentration of counter-ions. Default is 0.15M.')
ap.add_argument('--cation', type=str, default='Na+',
                 help='Positive ion used to neutralize system charge.')
ap.add_argument('--anion', type=str, default='Cl-',
                 help='Negative ion used to neutralize system charge.')
ap.add_argument('--seed', type=int, default=917,
                help='Seed number for random number generator(s).')

cmd = ap.parse_args()

# Set random seed for reproducibility
random.seed(cmd.seed)

logging.info('Started')
logging.info('Using:')
logging.info('  initial structure: {}'.format(cmd.structure))
logging.info('  force field: {}'.format(cmd.forcefield))
logging.info('  solvent model: {}'.format(cmd.solvent))
logging.info('  random seed: {}'.format(cmd.seed))
if cmd.neutralize:
    msg = '  neutralizing system with {}M of counter-ions.'
    logging.info(msg.format(cmd.ionic_strength))
    logging.info('  cation: {}'.format(cmd.cation))
    logging.info('  anion: {}'.format(cmd.anion))

# Figure out input file format from extension
fname, fext = os.path.splitext(cmd.structure)

# Read structure
structure = app.PDBxFile(cmd.structure)
forcefield = app.ForceField(cmd.forcefield, cmd.solvent)
modeller = app.Modeller(structure.topology, structure.positions)
bvec = modeller.topology.getPeriodicBoxVectors()

# Water model
if 'tip3p' in cmd.solvent:
    solvent_model = 'tip3p'
elif 'spce' in cmd.solvent:
    solvent_model = 'spce'
elif 'tip4pew' in cmd.solvent:
    solvent_model = 'tip4pew'
elif 'tip5p' in cmd.solvent:
    solvent_model = 'tip5p'
else:
    raise ValueError('Unknown water model: {}'.format(cmd.solvent))

# Add hydrogens according to force field
if cmd.neutralize:
    modeller.addSolvent(forcefield, model=solvent_model, boxVectors=bvec,
                        neutralize=cmd.neutralize,
                        ionicStrength=cmd.ionic_strength*units.molar,
                        positiveIon=cmd.cation,
                        negativeIon=cmd.anion)
else:
    modeller.addSolvent(forcefield, model=solvent_model, boxVectors=bvec,
                        neutralize=False)

n_atm = modeller.topology.getNumAtoms()
resname_list = [r.name for r in modeller.topology.residues()]
n_waters = resname_list.count('HOH')
n_cation = resname_list.count('NA')
n_anion = resname_list.count('CL')

logging.info('Solvated System Details')
logging.info('  num. atoms    = {:6d}'.format(n_atm))
logging.info('  num. waters   = {:6d}'.format(n_waters))
logging.info('  num. ions     = {:6d} Na {:6d} Cl'.format(n_cation, n_anion))

# Write complete structure
if cmd.output:
    if not cmd.output.endswith('.cif'):
        _fname = cmd.output + '.cif'
    else:
        _fname = cmd.output
else:
    _fname = fname + '_solvated' + '.cif'

cif_fname = _utils.make_fname(_fname)
logging.info('Writing structure to \'{}\''.format(cif_fname))
with open(cif_fname, 'w') as handle:
    app.PDBxFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)
