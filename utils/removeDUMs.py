#!/usr/bin/env python

"""
Removes dummy atoms from trajectory/topology pair and
associated checkpoint/state files.

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
ap.add_argument('--output', type=str, default='noDummies',
                help='File name for final system, in mmCIF format.')

cmd = ap.parse_args()

logging.info('Started')
logging.info('Using:')
logging.info('  trajectory: {}'.format(cmd.trajectory))
logging.info('  topology: {}'.format(cmd.topology))

# Read topology
mol = app.PDBxFile(cmd.topology)
top = md.Topology.from_openmm(mol.topology)

# Read trajectory
t = md.load(cmd.trajectory, top=top)

logging.info('Trajectory Details:')
logging.info('  no. of atoms: {}'.format(t.n_atoms))
logging.info('  no. of frames: {}'.format(t.n_frames))


# Remove dummies
atomsel = t.top.select('not name DUM')
n_dums = t.top.n_atoms - len(atomsel)
logging.info(f'Removing {n_dums} dummy atoms')
t = t.atom_slice(atomsel, inplace=True)

# Write trajectory
fname = cmd.output + '.dcd'
logging.info('Writing new trajectory to \'{}\''.format(fname))
t.save_dcd(fname, force_overwrite=True)

# Write new topology (subset of atoms)
top_fname = cmd.output + '.cif'
logging.info('Writing first frame to \'{}\''.format(top_fname))

last_idx = t.n_frames - 1
with open('{}'.format(top_fname), 'w') as handle:
    app.PDBxFile.writeFile(t.top.to_openmm(),
                           t.openmm_positions(last_idx), handle, keepIds=True)
