#!/usr/bin/env python

"""
Dumps a DCD file to PDBs
"""

from __future__ import print_function, division

import argparse
import logging
import os
import sys

import numpy as np
import mdtraj as md
import simtk.openmm.app as app  # necessary for topology reading from mmCIF

# Format logger
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='[%(asctime)s] %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')


def check_file(fpath):
    """Returns absolute path of file if it exists and is readable,
       raises IOError otherwise"""

    if os.path.isfile(fpath):
        return os.path.abspath(fpath)
    else:
        raise IOError('File not found/readable: {}'.format(fpath))


if __name__ == '__main__':

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('topology', help='Topology file corresponding to DCD')
    ap.add_argument('trajectory', help='DCD trajectory file')

    ap.add_argument('--output', default=None,
                    help='Root for naming PDB files: root + _ + frame + .pdb (e.g. trj_1.pdb)')

    ap.add_argument('--stride', default=1, type=int,
                    help='Read only i-th frame. Default: reads all (i=1)')

    cmd = ap.parse_args()

    # Read/Parse Topology
    topology_fpath = check_file(cmd.topology)
    if topology_fpath.endswith('cif'):
        structure = app.PDBxFile(topology_fpath)
        topology = md.Topology.from_openmm(structure.topology)
    else:
        structure = md.load(cmd.topology)
        topology = structure.topology

    logging.info('Read topology from file: {}'.format(topology_fpath))

    # Read trajectory
    trajectory_fpath = check_file(cmd.trajectory)
    logging.info('Reading trajectory from file: {}'.format(trajectory_fpath))
    trj = md.load(trajectory_fpath, top=topology,
                  stride=cmd.stride)

    logging.info('Removing PBCs and imaging molecules')

    topology.create_standard_bonds()
    anchors = topology.find_molecules()
    sorted_bonds = sorted(topology.bonds, key=lambda x: x[0].index)
    sorted_bonds = np.asarray([[b0.index, b1.index] for b0, b1 in sorted_bonds])

    trj.image_molecules(inplace=True, anchor_molecules=anchors, sorted_bonds=sorted_bonds, make_whole=True)

    # Write PDBs
    logging.info('Writing {} PDB files of {} atoms'.format(trj.n_frames, trj.n_atoms))
    froot = 'frame' if cmd.output is None else cmd.output
    n_frames = len(str(len(trj)))  # 1: 1, 10: 2, 100: 3, ...
    for idx, frame in enumerate(trj, start=1):
        frame_name = froot + '_' + str(idx).zfill(n_frames) + '.pdb'
        frame.save(frame_name, force_overwrite=True)
        logging.info('Wrote frame {}/{} to \'{}\''.format(idx, trj.n_frames, frame_name))

    logging.info('Done')
