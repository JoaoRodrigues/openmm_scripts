#!/usr/bin/env python

"""
Processes a PDB/mmCIF structure through a specific forcefield and solvent model.

Removes all hydrogens atoms and re-adds according to the forcefield specification.

Saves the processed structure in mmCIF format.

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
                help='File name for completed system in mmCIF format.')
ap.add_argument('--keepH', action='store_true', default=False,
                help='Retain original hydrogen atoms in the structure')
ap.add_argument('--forcefield', type=str, default='amber14-all.xml',
                help='Force field XML file to build the system with.')
ap.add_argument('--solvent', type=str, default='amber14/tip3p.xml',
                help='Solvent XML file to build the system with.')
ap.add_argument('--platform', type=str, default=None,
                help='Platform to run calculations on.')
ap.add_argument('--seed', type=int, default=917,
                help='Seed number for random number generator(s).')

cmd = ap.parse_args()

# Set random seed for reproducibility
random.seed(cmd.seed)

# Figure out platform
if cmd.platform is not None:
    cmd.platform = mm.Platform.getPlatformByName(cmd.platform)

logging.info('Started')
logging.info('Using:')
logging.info('  initial structure: {}'.format(cmd.structure))
logging.info('  force field(s): {}'.format(cmd.forcefield))
logging.info('  random seed: {}'.format(cmd.seed))

# Set platform-specific properties
platform, plat_properties = _utils.get_platform(cmd.platform)

# Figure out input file format from extension
fname, fext = os.path.splitext(cmd.structure)
if fext == '.pdb':
    parser = app.PDBFile
elif fext == '.cif':
    parser = app.PDBxFile
else:
    logging.error('Format not supported: must be \'.pdb\' or \'.cif\'')
    sys.exit(1)

# Read structure
structure = parser(cmd.structure)
forcefield = app.ForceField(cmd.forcefield)
modeller = app.Modeller(structure.topology, structure.positions)

# Add hydrogens according to force field
if not cmd.keepH:
    logging.info('Removing and re-adding hydrogen atoms')
    _elem_H = app.element.hydrogen
    hydrogens = [a for a in modeller.topology.atoms() if a.element == _elem_H]
    modeller.delete(hydrogens)
    modeller.addHydrogens(forcefield=forcefield, pH=7.0)

# Write complete structure
if cmd.output:
    if not cmd.output.endswith('.cif'):
        _fname = cmd.output + '.cif'
    else:
        _fname = cmd.output
else:
    _fname = fname + '_H' + '.cif'

cif_fname = _utils.make_fname(_fname)
logging.info('Writing structure to \'{}\''.format(cif_fname))
with open(cif_fname, 'w') as handle:
    app.PDBxFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)
