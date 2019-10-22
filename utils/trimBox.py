#!/usr/bin/env python

"""
Trims a periodic box to fit to certain dimensions.

Calculates solute (protein) center of mass and defines distances
from that point. Structure must NOT be split between different periodic images.

.2019. joaor@stanford.edu
"""

from __future__ import print_function, division

import argparse
import logging
import os
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
xy_opts = ap.add_mutually_exclusive_group()
xy_opts.add_argument('--distance-XY', type=float, dest='distXY',
                     help=('Distance in the XY plane (in nm) from the c.o.m of'
                           ' the solute to keep atoms in the box.'))
xy_opts.add_argument('--padding-XY', type=float, dest='padXY',
                     help=('Distance in the XY plane (in nm) from the edge of'
                           ' the solute to keep atoms in the box.'))
z_opts = ap.add_mutually_exclusive_group()
z_opts.add_argument('--distance-Z', type=float, dest='distZ',
                    help=('Distance along Z axis (in nm) from c.o.m of the'
                          ' solute to keep atoms in the box.'))
z_opts.add_argument('--padding-Z', type=float, dest='padZ',
                    help=('Distance along Z axis (in nm) from c.o.m of the'
                          ' solute to keep atoms in the box.'))

cmd = ap.parse_args()

if not any([cmd.distXY, cmd.padXY, cmd.distZ, cmd.padZ]):
    emsg = 'You must provide at least one --padding or --distance option'
    logging.error(emsg)
    sys.exit(1)

logging.info('Started')
logging.info('Using:')
logging.info('  initial structure: {}'.format(cmd.structure))
if cmd.distXY:
    logging.info('  distance XY: {}'.format(cmd.distXY))
elif cmd.padXY:
    logging.info('  padding XY: {}'.format(cmd.padXY))
if cmd.distZ:
    logging.info('  distance Z: {}'.format(cmd.distanceZ))
elif cmd.padZ:
    logging.info('  padding Z: {}'.format(cmd.paddingZ))

# Figure out input file format from extension
fname, fext = os.path.splitext(cmd.structure)

# Read structure
structure = app.PDBxFile(cmd.structure)
modeller = app.Modeller(structure.topology, structure.positions)

# Identify solute atoms
if cmd.solute:  # user provided
    solute_ids = sorted(cmd.solute)
else:  # automatically from residue names
    non_solute = set(('HOH', 'WAT', 'K', 'NA', 'CL', 'POP'))
    solute_ids = set()
    for residue in modeller.topology.residues():
        if residue.name not in non_solute:
            solute_ids.add(residue.chain.id)
    solute_ids = sorted(solute_ids)

logging.info('Solute chains: {}'.format(solute_ids))

# Get center of mass of solute
_xyz = [(x._value, y._value, z._value) for x, y, z in modeller.positions]
xyz = np.array(_xyz, dtype=np.float)
solute_idx = []
for chain in modeller.topology.chains():
    if chain.id not in solute_ids:
        continue
    idx_list = solute_idx
    for atom in chain.atoms():
        idx_list.append(atom.index)

if not solute_idx:
    logging.error('Solute chains not found in structure.')
    sys.exit(1)

solute_com = xyz[solute_idx, :].mean(axis=0)
logging.info(f'Solute center of mass: {solute_com}')

# Get original system and solute dimensions/unit cell vectors
sys_natoms = len(xyz)
ssizeX, ssizeY, ssizeZ = xyz.max(axis=0) - xyz.min(axis=0)
logging.info(f'System size: x={ssizeX:6.3f} y={ssizeY:6.3f} z={ssizeZ:6.3f}')

solute_xyz = xyz[solute_idx, :]
sizeX, sizeY, sizeZ = solute_xyz.max(axis=0) - solute_xyz.min(axis=0)
logging.info(f'Solute size: x={sizeX:6.3f} y={sizeY:6.3f} z={sizeZ:6.3f}')
vX, vY, vZ = modeller.topology.getUnitCellDimensions().value_in_unit(units.nanometer)
logging.info(f'Box dimensions: x={vX:6.3f} y={vY:6.3f} z={vZ:6.3f}')
og_volume = vX * vY * vZ
logging.info(f'Box volume: {og_volume:6.3f}')

# Calculate boundaries
if cmd.distXY:
    max_dx = cmd.distXY + solute_com[0]
    min_dx = solute_com[0] - cmd.distXY
    max_dy = cmd.distXY + solute_com[1]
    min_dy = solute_com[1] - cmd.distXY
elif cmd.padXY:
    max_dx = cmd.padXY + solute_com[0] + (sizeX / 2)
    min_dx = solute_com[0] - (sizeX / 2) - cmd.padXY
    max_dy = cmd.padXY + solute_com[1] + (sizeY / 2)
    min_dy = solute_com[1] - (sizeY / 2) - cmd.padXY

if cmd.distZ:
    max_dz = cmd.distZ + solute_com[2]
    min_dz = solute_com[2] - cmd.distZ
elif cmd.padZ:
    max_dz = cmd.padZ + solute_com[2] + (sizeZ / 2)
    min_dz = solute_com[2] - (sizeZ / 2) - cmd.padZ

# Identify atoms to be deleted
idx = []
if cmd.distXY or cmd.padXY:
    x = xyz[:, 0]
    y = xyz[:, 1]
    idx += np.where(np.logical_or(x < min_dx, x > max_dx))[0].tolist()
    idx += np.where(np.logical_or(y < min_dy, y > max_dy))[0].tolist()
if cmd.distZ or cmd.padZ:
    z = xyz[:, 2]
    idx += np.where(np.logical_or(z < min_dz, z > max_dz))[0].tolist()

# Delete flagged non-solute residues (that have atoms outside the boundaries)
# To avoid cutting an entire residue if only one atom is outside the boundary
# use a % criterion
_FRACTION = 0.70
idx = set(idx) - set(solute_idx)
to_remove = []
for residue in modeller.topology.residues():
    atom_idx = {a.index for a in residue.atoms()}
    fraction = len(atom_idx & idx) / len(atom_idx)
    if fraction > _FRACTION:
        to_remove += list(residue.atoms())
n_removed = len(to_remove)
modeller.delete(to_remove)

logging.info((f'Deleted {n_removed} atoms:'
             f' {n_removed*100 / sys_natoms:6.3f} % of total'))

# Adjust unit cell dimensions
_xyz = [(x._value, y._value, z._value) for x, y, z in modeller.positions]
xyz = np.array(_xyz, dtype=np.float)

if cmd.distXY or cmd.padXY:
    logging.info('Adjusting XY dimensions')
    offset = cmd.distXY if cmd.distXY is not None else cmd.padXY
    vX = (sizeX + 2 * offset)
    vY = (sizeY + 2 * offset)

if cmd.distZ or cmd.padZ:
    logging.info('Adjusting Z dimensions')
    offset = cmd.distZ if cmd.distZ is not None else cmd.padZ
    vZ = (sizeZ + 2 * offset)

modeller.topology.setUnitCellDimensions((vX, vY, vZ))

# Print some stats to make user happy
new_volume = vX * vY * vZ
vol_ratio = new_volume / og_volume
vX, vY, vZ = modeller.topology.getUnitCellDimensions().value_in_unit(units.nanometer)

logging.info(f'Trimmed box dimensions: x={vX:6.3f} y={vY:6.3f} z={vZ:6.3f}')
logging.info(f'Trimmed/Initial Volume Ratio: {vol_ratio:6.3f}')

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
    app.PDBxFile.writeFile(modeller.topology, modeller.positions,
                           handle, keepIds=True)
