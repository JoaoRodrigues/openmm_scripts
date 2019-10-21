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
# mdtraj topology loses chain information
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
# This is super annoying because the OpenMM->mdtraj screws up
# chain naming and residue numbering.
# We assume order is maintained within chains
# and that there are no gaps.
logging.info('Fixing chain/residue numbering')
new_topology = app.Topology()

# Extract sequences from OpenMM topology
seqs = {}
reslists = {}
for openmm_chain in mol.topology.chains():
    chain_seq = [r.name for r in openmm_chain.residues()]
    seqs[openmm_chain.id] = tuple(chain_seq)
    reslists[openmm_chain.id] = list(openmm_chain.residues())

# Mapping of old atom index to new atom index
atom_map = {}

mdtraj_top = reimaged.top.to_openmm()
for mdtraj_chain in mdtraj_top.chains():
    chain_seq = tuple([r.name for r in mdtraj_chain.residues()])
    for chain, seq in seqs.items():
        if chain_seq == seq:
            logging.debug('Chain {} <> {}'.format(mdtraj_chain.id, chain))
            break
    else:
        msg = 'Chain not matched in OpenMM topology: {}'
        logging.error(msg.format(mdtraj_chain.id))
        sys.exit(1)

    openmm_reslist = reslists[chain]
    mdtraj_reslist = [r for r in mdtraj_chain.residues()]
    assert len(openmm_reslist) == len(mdtraj_reslist)

    # Rebuild new topology
    new_chain = new_topology.addChain(chain)
    for resO, resM in zip(openmm_reslist, mdtraj_reslist):
        assert resO.name == resM.name
        new_res = new_topology.addResidue(resO.name, new_chain, 
                                          resO.id, resO.insertionCode)
        atomsO = list(resO.atoms())
        atomsM = list(resM.atoms())
        for aO, aM in zip(atomsO, atomsM):
            assert aO.name == aM.name
            new_atom = new_topology.addAtom(aO.name, aO.element, new_res, aO.id)
            atom_map[aO.index] = new_atom.index

logging.info('Adding bonds to new topology')
# Add bonds to new topology
atom_list =list(new_topology.atoms())
for b in mol.topology.bonds():
    ai, aj = b
    new_ai_idx = atom_map.get(ai.index, None)
    new_aj_idx = atom_map.get(aj.index, None)
    if new_ai_idx is not None and new_aj_idx is not None:
        new_ai = atom_list[new_ai_idx]
        new_aj = atom_list[new_aj_idx]
        new_topology.addBond(new_ai, new_aj, type=b.type, order=b.order)
    else:
        logging.warn(f'  ignoring bond: {b}. Error: {new_ai_idx} <> {new_aj_idx}')

# Add box vectors to topology (last frame)
boxvec = reimaged[-1].unitcell_vectors[0]
new_topology.setPeriodicBoxVectors(boxvec)

reimaged_top_fname = cmd.output + '.cif'
logging.info('Writing first frame to \'{}\''.format(reimaged_top_fname))

last_idx = reimaged.n_frames - 1
with open('{}'.format(reimaged_top_fname), 'w') as handle:
    app.PDBxFile.writeFile(new_topology,
                           reimaged.openmm_positions(last_idx), handle, keepIds=True)
