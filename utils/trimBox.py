#!/usr/bin/env python

"""
Trims a periodic box to fit to certain dimensions.

Calculates solute (protein) center of mass and defines distances
from that point.

.2019. joaor@stanford.edu
"""

from __future__ import print_function, division

import argparse
import logging
import os
import random
import sys

import numpy as np

import simtk.unit as units
import simtk.openmm.app as app

# Format logger
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='[%(asctime)s] %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')


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
ap.add_argument('structure', help='Input coordinate file (.cif or .pdb)')
# Options
ap.add_argument('--output', type=str, default=None,
                help='File name for final system, in mmCIF format.')
ap.add_argument('--solute', type=str, nargs='+',
                help='Solute chain(s)')
x_opts = ap.add_mutually_exclusive_group()
x_opts.add_argument('--distanceX', type=float,
                    help='Distance (in nm) from c.o.m. to keep atoms (x)')
x_opts.add_argument('--paddingX', type=float,
                    help='Distance (in nm) around protein to keep atoms (x)')
y_opts = ap.add_mutually_exclusive_group()
y_opts.add_argument('--distanceY', type=float,
                    help='Distance (in nm) from c.o.m. to keep atoms (y)')
y_opts.add_argument('--paddingY', type=float,
                    help='Distance (in nm) around protein to keep atoms (y)')
z_opts = ap.add_mutually_exclusive_group()
z_opts.add_argument('--distanceZ', type=float,
                    help='Distance (in nm) from c.o.m. to keep atoms (z)')
z_opts.add_argument('--paddingZ', type=float,
                    help='Distance (in nm) around protein to keep atoms (z)')

cmd = ap.parse_args()

if not any([cmd.paddingX, cmd.paddingY, cmd.paddingZ,
            cmd.distanceX, cmd.distanceY, cmd.distanceZ]):
    logging.error('--padding or --distance must be provided')
    sys.exit(1)

logging.info('Started')
logging.info('Using:')
logging.info('  initial structure: {}'.format(cmd.structure))
logging.info('  distance X: {}'.format(cmd.distanceX))
logging.info('  distance Y: {}'.format(cmd.distanceY))
logging.info('  distance Z: {}'.format(cmd.distanceZ))
logging.info('  padding X: {}'.format(cmd.paddingX))
logging.info('  padding Y: {}'.format(cmd.paddingY))
logging.info('  padding Z: {}'.format(cmd.paddingZ))

# Figure out input file format from extension
fname, fext = os.path.splitext(cmd.structure)

# Read structure
structure = app.PDBxFile(cmd.structure)
modeller = app.Modeller(structure.topology, structure.positions)

# Identify protein atoms
if cmd.solute:
    solute_ids = sorted(cmd.solute)
else:
    wat = set(('HOH', 'WAT'))
    # Ignore water, everyone else is solute
    solute_ids = set()
    for residue in modeller.topology.residues():
        if residue.name not in wat:
            solute_ids.add(residue.chain.id)
    solute_ids = sorted(solute_ids)

logging.info('Solute chains: {}'.format(solute_ids))

# Get center of mass of solute
_xyz = [(x._value, y._value, z._value) for x, y, z in modeller.positions]
xyz = np.array(_xyz, dtype=np.float)
others_idx, solute_idx = [], []
for chain in modeller.topology.chains():
    if chain.id not in solute_ids:
        idx_list = others_idx
    else:
        idx_list = solute_idx
    for atom in chain.atoms():
        idx_list.append(atom.index)

if not solute_idx:
    raise ValueError('Chains not found in structure.')

solute_com = xyz[solute_idx, :].mean(axis=0)
solute_xyz = xyz[solute_idx, :]
sizeX, sizeY, sizeZ = solute_xyz.max(axis=0) - solute_xyz.min(axis=0)

# If lipids, check which are on which lipids
resMeanZ = {}
membraneMeanZ = 0.0
numLipidAtoms = 0
for res in modeller.topology.residues():
    if res.name.startswith('POP'):
        numResAtoms = 0
        sumZ = 0.0
        for atom in res.atoms():
            numResAtoms += 1
            sumZ += xyz[atom.index][2]
        numLipidAtoms += numResAtoms
        membraneMeanZ += sumZ
        resMeanZ[res] = sumZ/numResAtoms
if numLipidAtoms:
    membraneMeanZ /= numLipidAtoms
    lipidLeaf = dict((res, 0 if resMeanZ[res] < membraneMeanZ else 1) for res in resMeanZ)
else:
    lipidLeaf = {}
logging.info('Lipid Molecules: {}'.format(len(lipidLeaf)))

# Calculate boundaries
if cmd.distanceX:
    max_dx = cmd.distanceX + solute_com[0]
    min_dx = solute_com[0] - cmd.distanceX
if cmd.distanceY:
    max_dy = cmd.distanceY + solute_com[1]
    min_dy = solute_com[1] - cmd.distanceY
if cmd.paddingX:
    max_dx = cmd.paddingX + solute_com[0] + sizeX/2
    min_dx = solute_com[0] - sizeX/2 - cmd.paddingX
if cmd.paddingY:
    max_dy = cmd.paddingY + solute_com[1] + sizeY/2
    min_dy = solute_com[1] - sizeY/2 - cmd.paddingY
if cmd.distanceZ:
    max_dz = cmd.distanceZ + solute_com[2]
    min_dz = solute_com[2] - cmd.distanceZ
if cmd.paddingZ:
    max_dz = cmd.paddingZ + solute_com[2] + sizeZ/2
    min_dz = solute_com[2] - sizeZ/2 - cmd.paddingZ

# Pick outside atoms
idx = []
if cmd.distanceX or cmd.paddingX:
    x = xyz[:, 0]
    idx += np.where(np.logical_or(x < min_dx, x > max_dx))[0].tolist()
if cmd.distanceY or cmd.paddingY:
    y = xyz[:, 1]
    idx += np.where(np.logical_or(y < min_dy, y > max_dy))[0].tolist()
if cmd.distanceZ or cmd.paddingZ:
    z = xyz[:, 2]
    idx += np.where(np.logical_or(z < min_dz, z > max_dz))[0].tolist()

# Delete flagged non-solute residues (that have atoms outside the boundaries)
# To avoid cutting an entire residue if only one atom is outside the boundary
# use a % criterion
_FRACTION = 0.70  # was 0.98
idx = set(idx) - set(solute_idx)
atoms = []
for residue in modeller.topology.residues():
    atom_idx = {a.index for a in residue.atoms()}
    fraction = len(atom_idx & idx) / len(atom_idx)
    if fraction > _FRACTION:
        # print(residue, fraction, len(atom_idx & idx))
        atoms += list(residue.atoms())
logging.info('Deleting {} atoms'.format(len(atoms)))
modeller.delete(atoms)

# Set unit cell dimensions
# Only change vectors that we modified
logging.info('Adjusting unit cell dimensions:')
vX, vY, vZ = modeller.topology.getUnitCellDimensions()
_xyz = [(x._value, y._value, z._value) for x, y, z in modeller.positions]
xyz = np.array(_xyz, dtype=np.float)
# boxX, boxY, boxZ = xyz.max(axis=0) - solute_xyz.min(axis=0)

if cmd.distanceX or cmd.paddingX:
    if cmd.distanceX:
        pad = cmd.distanceX
    else:
        pad = cmd.paddingX
    vX = (sizeX+2*pad)
    # vX = boxX
    logging.info('  vx: {}'.format(vX))
else:
    vX = vX.value_in_unit(units.nanometer)
if cmd.distanceY or cmd.paddingY:
    if cmd.distanceY:
        pad = cmd.distanceY
    else:
        pad = cmd.paddingY
    vY = (sizeY+2*pad)
    # vY = boxY
    logging.info('  vy: {}'.format(vY))
else:
    vY = vY.value_in_unit(units.nanometer)
if cmd.distanceZ or cmd.paddingZ:
    if cmd.distanceZ:
        pad = cmd.distanceZ
    else:
        pad = cmd.paddingZ
    vZ = (sizeZ+2*pad)
    # vZ = boxZ
    logging.info('  vz: {}'.format(vZ))
else:
    vZ = vZ.value_in_unit(units.nanometer)

modeller.topology.setUnitCellDimensions((vX, vY, vZ))

# Write complete structure
if cmd.output:
    if not cmd.output.endswith('.cif'):
        _fname = cmd.output + '.cif'
    else:
        _fname = cmd.output
else:
    _fname = fname + 'trimmed' + '.cif'

cif_fname = get_filename(_fname)
logging.info('Writing structure to \'{}\''.format(cif_fname))
with open(cif_fname, 'w') as handle:
    app.PDBxFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)
