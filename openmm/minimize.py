#!/usr/bin/env python

"""
Molecular Dynamics Simulation of a System.
Uses a rhombic dodecahedron box and the Amber99sb-ILDN FF (w/ TIP3P).
Default padding of 1 nm from protein to edge of the box.
Neutralizes the system and adds ions to 0.15 M.

.2017. joaor@stanford.edu
"""

from __future__ import print_function, division

import argparse
import logging
import math
import os
import sys

import numpy as np
from scipy.spatial.distance import pdist

import simtk.openmm.app as app
import simtk.openmm as mm
import simtk.unit as units

# Format logger
logging.basicConfig(level=logging.INFO, 
                    format='[%(asctime)s] %(message)s', 
                    datefmt='%Y/%m/%d %H:%M:%S')

##
# Parse user input and options
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('pdb', help='Input PDB file')
ap.add_argument('-p', '--pad', type=float, help='Box Padding in nm', default=1.0)
user_args = ap.parse_args()

if not os.path.isfile(user_args.pdb):
    raise IOError('Could not read/open input file: {}'.format(user_args.pdb))

# Read PDB file and set forcefield
logging.info('Reading PDB file: {}'.format(user_args.pdb))
pdb = app.PDBFile(user_args.pdb)
forcefield = app.ForceField('amber99sbildn.xml', 'tip3p.xml')

# Processing structure and build box
logging.info('Adding missing atoms')
modeller = app.Modeller(pdb.topology, pdb.positions)
modeller.addHydrogens(forcefield, pH=7.0) # already does EM


# Build rhombic dodecahedron box (square xy-plane)
# 0. Center system at origin
com_xyz = modeller.positions.mean()
for i, xyz in enumerate(modeller.positions):
    modeller.positions[i] = xyz - com_xyz

# 1. Move coordinates to numpy array for efficiency
xyz = np.array([(x._value, y._value, z._value) for x, y, z in modeller.positions])
xyz_size = np.amax(xyz, axis=0) - np.amin(xyz, axis=0)
xyz_diam = np.max(pdist(xyz, 'euclidean'))
d = xyz_diam + user_args.pad*2

u = np.array((d, 0, 0))
v = np.array((0, d, 0))
w = np.array((d/2, d/2, np.sqrt(2)*d/2))
box_vol = 0.5 * np.sqrt(2) * np.power(d, 3)

n_atm = modeller.topology.getNumAtoms()
n_res = modeller.topology.getNumResidues()

logging.info('System      : {:6d} Atoms {:6d} Residues'.format(n_atm, n_res))
logging.info('System Size : {:6.3f} {:6.3f} {:6.3f}'.format(*xyz_size))
logging.info('Diameter    : {:6.3f}'.format(xyz_diam))
logging.info('Box Volume  : {:6.3f}'.format(box_vol))
logging.info('Box Vectors :')
logging.info('  u = {:6.3f} {:6.3f} {:6.3f}'.format(*u))
logging.info('  v = {:6.3f} {:6.3f} {:6.3f}'.format(*v))
logging.info('  w = {:6.3f} {:6.3f} {:6.3f}'.format(*w))

modeller.topology.setPeriodicBoxVectors((u, v, w))

# Solvate the Box and add counter ions at 0.15 M
logging.info('Solvating the system')
modeller.addSolvent(forcefield, model='tip3p', 
                    neutralize=True, ionicStrength=0.15*units.molar)

resname_list = [r.name for r in modeller.topology.residues()]
num_waters = resname_list.count('HOH')
num_cation = resname_list.count('NA')
num_anion = resname_list.count('CL')

logging.info('  num. waters: {:6d}'.format(num_waters))
logging.info('  num. ions: {:6d} Na {:6d} Cl'.format(num_cation, num_anion))

# Create System
system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME,
                                 nonbondedCutoff=1*units.nanometer, constraints=app.HBonds)

# Add restraints on protein heavy atoms
logging.info('Adding pos. res. on non-hydrogen protein atoms')
all_atoms = list(modeller.topology.atoms())
posre = mm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
posre_K = 1000.0 
posre.addGlobalParameter("k", posre_K*units.kilocalories_per_mole/units.angstroms**2)
posre.addPerParticleParameter("x0")
posre.addPerParticleParameter("y0")
posre.addPerParticleParameter("z0")
for i, atom_crd in enumerate(modeller.getPositions()):
    if all_atoms[i].element != 'H':
        posre.addParticle(i, atom_crd.value_in_unit(units.nanometers))
system.addForce(posre)

# Setup System
integrator = mm.LangevinIntegrator(300*units.kelvin, 1/units.picosecond, 0.002*units.picoseconds)
simulation = app.Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)

platform = simulation.context.getPlatform()
logging.info('Using platform: {}'.format(platform.getName()))

# Minimize
logging.info('Running energy minimization')
simulation.minimizeEnergy(maxIterations=500)

# Write minimized file
positions = simulation.context.getState(getPositions=True).getPositions()
mini_name = "{}_minimized.pdb".format(user_args.pdb[:-4])
app.PDBFile.writeFile(simulation.topology, positions, open(mini_name, 'w'))

# Remove restraints
#system.removeForce(posre)

logging.info('Done')
