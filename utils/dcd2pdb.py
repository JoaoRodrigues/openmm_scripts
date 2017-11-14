#!/usr/bin/env python

"""
Dumps a DCD file to PDBs
"""

from __future__ import print_function, division

import argparse
import logging
import os
import sys

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
    ap.add_argument('trajectory', nargs='+', help='DCD trajectory file')

    ap.add_argument('--output', default=None,
                    help='Root for naming PDB files: root + _ + frame + .pdb (e.g. trj_1.pdb)')

    ap.add_argument('--keep-all', default=False, action='store_true',
                    help='Keeps all atoms, incl. solvent/lipids, in trajectory')
    ap.add_argument('--stride', default=1, type=int,
                    help='Read only i-th frame. Default: reads all (i=1)')

    cmd = ap.parse_args()

    # Read/Parse Topology
    topology_fpath = check_file(cmd.topology)
    if topology_fpath.endswith('cif'):
        structure = app.PDBxFile(topology_fpath)
        topology = md.Topology.from_openmm(structure)
    else:
        structure = md.load(cmd.topology)
        topology = structure.topology

    logging.info('Read topology from file: {}'.format(topology_fpath))

    if cmd.keep_all:
        atomsel = topology.select('all')
        logging.info('Writing all atoms')
    else:
        logging.info('Writing only protein atoms')
        atomsel = topology.select('protein')

    # Read trajectory
    trajectory_fpath = check_file(cmd.trajectory)
    logging.info('Reading trajectory from file:'.format(trajectory_fpath))
    trj = md.load(trajectory_fpath, top=topology,
                  stride=cmd.stride, atom_indices=atomsel)

    # Write PDBs
    logging.info('Writing {} PDB files of {} atoms'.format(trj.n_frames, trj.n_atoms))
    froot = 'frame' if cmd.output is None else cmd.output
    for idx, frame in enumerate(trj, start=1):
        frame_name = froot + '_' + str(idx) + '.pdb'
        frame.write(frame_name, force_overwrite=True)
        logging.info('Wrote frame {}/{} to \'{}\''.format(idx, trj.n_frames, frame_name))

    logging.info('Done')
