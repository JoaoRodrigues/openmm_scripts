#!/usr/bin/env python

"""
Merges DCD trajectory files.
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
    ap.add_argument('topology', help='Topology file corresponding to DCDs')
    ap.add_argument('trajectories', nargs='+', help='DCD files to merge')

    ap.add_argument('--output', default='merged.dcd',
                    help='Filename to save the merged trajectory to.')

    ap.add_argument('--keep-all', default=False, action='store_true',
                    help='Keeps all atoms, incl. solvent/lipids, in merged trajectory')
    ap.add_argument('--stride', default=1, type=int,
                    help='Write only i-th frame. Default: writes all (i=1)')

    cmd = ap.parse_args()

    # Read/Parse Topology
    topology_fpath = check_file(cmd.topology)
    logging.info('Reading topology from file: {}'.format(topology_fpath))

    if topology_fpath.endswith('cif'):
        structure = app.PDBxFile(topology_fpath)
        topology = md.Topology.from_openmm(structure.topology)
    else:
        structure = md.load(cmd.topology)
        topology = structure.topology

    if cmd.keep_all:
        atomsel = topology.select('all')
        logging.info('Keeping all atoms in merged trajectory')
    else:
        logging.info('Keeping only protein atoms in merged trajectory')
        atomsel = topology.select('protein')

    # Read/Merge trajectories
    n_chunks = len(cmd.trajectories)
    merged_trj = None

    logging.info('Attemping to merge {} trajectories'.format(n_chunks))
    for idx, trj_fpath in enumerate(cmd.trajectories, start=1):
        logging.info('Reading trajectory {}/{}'.format(idx, n_chunks))
        print(trj_fpath)
        if merged_trj is None:
            merged_trj = md.load(trj_fpath, top=topology,
                                 stride=cmd.stride, atom_indices=atomsel)
        else:
            trj = md.load(trj_fpath, top=topology,
                          stride=cmd.stride, atom_indices=atomsel)
            merged_trj = merged_trj + trj

        n_frames = merged_trj.n_frames
        logging.info('Merged frames: {}'.format(n_frames))

    # Center coordinates
    merged_trj.center_coordinates()

    # Write merged trajectory
    logging.info('Writing merged trajectory to \'{}\''.format(cmd.output))
    merged_trj.save_dcd(cmd.output, force_overwrite=True)

    # Write new topology (subset of atoms)
    root_fname, _ = os.path.splitext(cmd.output)
    logging.info('Writing first frame to \'{}.cif\''.format(root_fname))

    top_subset = topology.subset(atomsel)
    with open('{}.cif'.format(root_fname), 'w') as handle:
        app.PDBxFile.writeFile(top_subset.to_openmm(), merged_trj.openmm_positions(0), handle)

    logging.info('Done')
