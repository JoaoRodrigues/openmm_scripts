#!/usr/bin/env python

"""
Analyzes the dimensions of a system to create a PBC box of a given type
and dimensions (padding).

Saves the boxed structure in CIF format.

.2017. joaor@stanford.edu
"""

from __future__ import print_function, division

import argparse
import logging
import os
import random
import sys

import numpy as np

import simtk.openmm.app as app
import simtk.openmm as mm

# Format logger
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='[%(asctime)s] %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')


# cython-optmized pairwise distance function
# profiled to run in ~50% of the time of pdist
# much less memory hungry. No storage of all distances.
try:
    # add to PYTHONPATH current workdir and script dir
    sys.path.insert(0, os.curdir)
    sys.path.insert(0, os.path.dirname(__file__))

    from _pwdistance import pw_dist
except ImportError as e:
    logging.warning('Using numpy (slower) pwdist routine for simulation setup.')
    from scipy.spatial.distance import pdist

    def pw_dist(xyz_array):
        return np.amax(pdist(xyz_array, 'euclidean'))


def get_filename(name):
    """Finds and moves existing file with same name"""

    if not os.path.isfile(name):
        return name

    rootname = name
    num = 1
    while 1:
        name = '#{}.{}#'.format(rootname, num)
        if os.path.isfile(name):
            num += 1
        else:
            os.rename(rootname, name)
            break
    return rootname

##
# Parse user input and options
ap = argparse.ArgumentParser(description=__doc__)

# Mandatory
ap.add_argument('structure', help='Input coordinate file (.cif)')
# Options
ap.add_argument('--output', type=str, default=None,
                help='File name for PBC system in mmCIF format.')
ap.add_argument('--boxtype', choices=['cubic', 'dodecahedron'], type=str,
                default='dodecahedron',
                help='Geometry of the periodic box to build.')
ap.add_argument('--padding', type=float, default=1.1,
                help='Minimum distance between solute and edge of the periodic box.')

ap.add_argument('--seed', type=int, default=917,
                help='Seed number for random number generator(s).')

cmd = ap.parse_args()

# Set random seed for reproducibility
random.seed(cmd.seed)

# Figure out platform
logging.info('Started')
logging.info('Using:')
logging.info('  initial structure: {}'.format(cmd.structure))
logging.info('  random seed: {}'.format(cmd.seed))

# Split filename
fname, fext = os.path.splitext(cmd.structure)

# Read structure
structure = app.PDBxFile(cmd.structure)
modeller = app.Modeller(structure.topology, structure.positions)

# Add hydrogens according to force field
logging.info('Building periodic simulation box')
logging.info('  box type: {}'.format(cmd.boxtype))
logging.info('  padding distance: {:2.1f}'.format(cmd.padding))

# Build
# 0. Center system at origin
com_xyz = modeller.positions.mean()
for i, xyz_i in enumerate(modeller.positions):
    modeller.positions[i] = xyz_i - com_xyz

# 1. Move coordinates to numpy array for efficiency
_xyz = [(x._value, y._value, z._value) for x, y, z in modeller.positions]
xyz = np.array(_xyz, dtype=np.float)
xyz_size = np.amax(xyz, axis=0) - np.amin(xyz, axis=0)
xyz_diam = pw_dist(xyz)

d = xyz_diam + (cmd.padding * 2)

# rhombic dodecahedron box (square xy-plane)
if cmd.boxtype == 'dodecahedron':
    u = np.array((d, 0, 0))
    v = np.array((0, d, 0))
    w = np.array((d/2, d/2, np.sqrt(2)*d/2))
    box_vol = 0.5 * np.sqrt(2) * np.power(d, 3)
elif cmd.boxtype == 'cubic':
    u = np.array((d, 0, 0))
    v = np.array((0, d, 0))
    w = np.array((0, 0, d))
    box_vol = np.power(d, 3)
else:
    raise NotImplementedError('Unsupported box type: {}'.format(cmd.boxtype))

modeller.topology.setPeriodicBoxVectors((u, v, w))

n_atm = modeller.topology.getNumAtoms()
n_res = modeller.topology.getNumResidues()

logging.info('  num. atoms    = {:8d}'.format(n_atm))
logging.info('  num. residues = {:8d}'.format(n_res))
logging.info('  Solute Size   = {:8.3f} {:8.3f} {:8.3f}'.format(*xyz_size))
logging.info('  Box Diameter  = {:8.3f} nm'.format(d))
logging.info('  Box Volume    = {:8.3f} nm^3'.format(box_vol))
logging.info('  Box Vectors:')
logging.info('    u = {:6.3f} {:6.3f} {:6.3f}'.format(*u))
logging.info('    v = {:6.3f} {:6.3f} {:6.3f}'.format(*v))
logging.info('    w = {:6.3f} {:6.3f} {:6.3f}'.format(*w))

# Write structure with PBC
if cmd.output:
    if not cmd.output.endswith('.cif'):
        _fname = cmd.output + '.cif'
    else:
        _fname = cmd.output
else:
    _fname = fname + '_PBC' + '.cif'

cif_fname = get_filename(_fname)
logging.info('Writing structure to \'{}\''.format(cif_fname))
with open(cif_fname, 'w') as handle:
    app.PDBxFile.writeFile(modeller.topology, modeller.positions, handle)
