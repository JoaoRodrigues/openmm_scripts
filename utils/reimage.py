#!/usr/bin/env python

"""
Reimages a trajectory.

.2019. joaor@stanford.edu
"""

from __future__ import print_function, division

import argparse
import logging
import os
import sys

import mdtraj as md
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
ap.add_argument('trajectory', help='Input trajectory file (.dcd)')
ap.add_argument('topology', help='Input topology file (.cif)')

# Options
ap.add_argument('--output', type=str, default='reimaged',
                help='File name for final system, in mmCIF format.')
ap.add_argument('--stride', type=int, default=1,
                help='Store every n-th frame')
ap.add_argument('--keep_solvent', action='store_true',
                help='Keeps solvent in the simulation')

cmd = ap.parse_args()

logging.info('Started')
logging.info('Using:')
logging.info('  trajectory: {}'.format(cmd.trajectory))
logging.info('  topology: {}'.format(cmd.topology))
logging.info('  stride: {}'.format(cmd.stride))
logging.info('  solvent: {}'.format(cmd.keep_solvent))

# Read topology
mol = app.PDBxFile(cmd.topology)
top = md.Topology.from_openmm(mol.topology)

# Read trajectory
t = md.load(cmd.trajectory, top=top, stride=cmd.stride)

logging.info('Trajectory Details:')
logging.info('  no. of atoms: {}'.format(t.n_atoms))
logging.info('  no. of frames: {}'.format(t.n_frames))

# Remove dummy atoms just in case
atomsel = t.top.select('not name DUM')
t = t.atom_slice(atomsel, inplace=True)

# Remove solvent
if not cmd.keep_solvent:
    logging.info('Removing solvent atoms')
    atomsel = t.top.select('protein or resname POP')
    t_noHOH = t.atom_slice(atomsel, inplace=False)
    t = t_noHOH

logging.info('Imaging trajectory')
mols = t.top.find_molecules()  # use largest as anchor
reimaged = t.image_molecules(inplace=False, anchor_molecules=mols[:1],
                             other_molecules=mols[1:])

# Write reimaged trajectory
reimaged_fname = cmd.output + '.dcd'
logging.info('Writing reimaged trajectory to \'{}\''.format(reimaged_fname))
reimaged.save_dcd(reimaged_fname, force_overwrite=True)

# Write new topology (subset of atoms)
reimaged_top_fname = cmd.output + '.cif'
logging.info('Writing first frame to \'{}\''.format(reimaged_top_fname))

last_idx = reimaged.n_frames - 1
with open('{}'.format(reimaged_top_fname), 'w') as handle:
    app.PDBxFile.writeFile(reimaged.top.to_openmm(),
                           reimaged.openmm_positions(last_idx), handle, keepIds=True)
