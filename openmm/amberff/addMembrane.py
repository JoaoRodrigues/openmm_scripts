#!/usr/bin/env python

"""
Adds membrane to the system, and solvates (optionally including counter-ions).

Saves the final structure in mmCIF format.

.2019. joaor@stanford.edu
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
go = ap.add_argument_group('General Options')
go.add_argument('--output', type=str, default=None,
                help='File name for solvated system. Will *always* use mmCIF format.')
go.add_argument('--seed', type=int, default=917,
                help='Seed number for random number generator(s).')

ff = ap.add_argument_group('Force field Options')
ff.add_argument('--forcefield', type=str, default='amber14-all.xml',
                help='Force field to build the system with.')
ff.add_argument('--solvent', type=str, default='amber14/tip3p.xml',
                help='Solvent model to use in minimization.')

mem = ap.add_argument_group('Membrane Options')
mem.add_argument('--lipid', type=str, default='POPC',
                 help='Lipid type used to build membrane (default: POPC)')
mem.add_argument('--paddingXY', type=float, default=1.0,
                 help='Padding in XY dimensions (in nm) when adding membrane')
mem.add_argument('--neutralize', action='store_true',
                 help='Adds counter-ions to neutralize total system charge.')

sol = ap.add_argument_group('Solvation Options')
sol.add_argument('--ionic-strength', type=float, default=0.15,
                 help='Molar concentration of counter-ions. Default is 0.15M.')
sol.add_argument('--cation', type=str, default='K+',
                 help='Positive ion used to neutralize system charge.')
sol.add_argument('--anion', type=str, default='Cl-',
                 help='Negative ion used to neutralize system charge.')

cmd = ap.parse_args()

# Set random seed for reproducibility
random.seed(cmd.seed)

logging.info('Started')
logging.info('Using:')
logging.info('  initial structure: {}'.format(cmd.structure))
logging.info('  force field: {}'.format(cmd.forcefield))
logging.info('  lipid type(s): {}'.format(cmd.lipid))
logging.info('  padding XY: {}'.format(cmd.paddingXY))
logging.info('  solvent model: {}'.format(cmd.solvent))
if cmd.neutralize:
    msg = '  neutralizing system with {}M of counter-ions.'
    logging.info(msg.format(cmd.ionic_strength))
    logging.info('  cation: {}'.format(cmd.cation))
    logging.info('  anion: {}'.format(cmd.anion))
logging.info('  random seed: {}'.format(cmd.seed))

# Figure out input file format from extension
fname, fext = os.path.splitext(cmd.structure)

# Read structure
structure = app.PDBxFile(cmd.structure)
forcefield = app.ForceField(cmd.forcefield, cmd.solvent)
modeller = app.Modeller(structure.topology, structure.positions)

# Add Membrane w/wout ions
if cmd.neutralize:
    modeller.addMembrane(forcefield, lipidType=cmd.lipid,
                         minimumPadding=cmd.paddingXY,
                         neutralize=cmd.neutralize,
                         ionicStrength=cmd.ionic_strength*units.molar,
                         positiveIon=cmd.cation,
                         negativeIon=cmd.anion)
else:
    modeller.addMembrane(forcefield, lipidType=cmd.lipid,
                         minimumPadding=cmd.paddingXY,
                         neutralize=False)

n_atm = modeller.topology.getNumAtoms()
resname_list = [r.name for r in modeller.topology.residues()]
n_waters = resname_list.count('HOH')
n_cation = resname_list.count(cmd.cation[:-1].upper())
n_anion = resname_list.count(cmd.anion[:-1].upper())
n_lipid = resname_list.count(cmd.lipid[:3])
logging.info('Solvated System Details')
logging.info('  num. atoms    = {:6d}'.format(n_atm))
logging.info('  num. lipids     = {:6d} {:4s}s'.format(n_lipid, cmd.lipid))
logging.info('  num. waters   = {:6d}'.format(n_waters))
if cmd.neutralize:
    logging.info('  num. ions     = {:6d} {} {:6d} {}'.format(n_cation, cmd.cation,
                                                              n_anion, cmd.anion))

# Write complete structure
if cmd.output:
    if not cmd.output.endswith('.cif'):
        _fname = cmd.output + '.cif'
    else:
        _fname = cmd.output
else:
    _fname = fname + '_withMembrane' + '.cif'

cif_fname = _utils.make_fname(_fname)
logging.info('Writing structure to \'{}\''.format(cif_fname))
with open(cif_fname, 'w') as handle:
    app.PDBxFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)
