#!/usr/bin/env python

"""
Performs energy minimization on existing structure using a given AMBER forcefield.

.2017. joaor@stanford.edu
"""

from __future__ import print_function, division

import argparse
import logging
import os
import random
import sys

import simtk.openmm.app as app
import simtk.openmm as mm
import simtk.unit as units

import _utils
import _restraints

# Format logger
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='[%(asctime)s] %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')


##
# Parse user input and options
ap = argparse.ArgumentParser(description=__doc__)

# Mandatory
ap.add_argument('structure', help='Input coordinate file (.cif)')
# Options
ap.add_argument('--output', type=str, default=None,
                help='File name for minimized structure. Will *always* use mmCIF format.')
ap.add_argument('--forcefield', type=str, default='amber14-all.xml',
                help='Force field XML file to parameterize system')
ap.add_argument('--solvent', type=str, default='amber14/tip3p.xml',
                help='Solvent XML file to parameterize solvent')
ap.add_argument('--platform', type=str, default=None,
                help='Platform to run calculations on.')
ap.add_argument('--seed', type=int, default=917,
                help='Seed number for random number generator(s).')

ap.add_argument('--posre', action='store_true',
                help='Add position restraints to non-solvent heavy atoms')
ap.add_argument('--posre_K', default=1000.0, type=float,
                help='Force constant for position restraints in kJ.mol/nm^2.')
ap.add_argument('--iterations', default=100, type=int,
                help='Number of iterations to minimize for. Default is 100.')

cmd = ap.parse_args()

# Set random seed for reproducibility
random.seed(cmd.seed)

logging.info('Started')
logging.info('Using:')
logging.info('  initial structure: {}'.format(cmd.structure))
logging.info('  force field: {}'.format(cmd.forcefield))
logging.info('  random seed: {}'.format(cmd.seed))

# Set platform-specific properties
platform, plat_properties = _utils.get_platform(cmd.platform)

# Split input file name
fname, fext = os.path.splitext(cmd.structure)

# Read structure
structure = app.PDBxFile(cmd.structure)
forcefield = app.ForceField(cmd.forcefield, cmd.solvent)

# Build system & integrator
logging.info('Setting up system and integrator')
system = forcefield.createSystem(structure.topology, nonbondedMethod=app.PME,
                                 nonbondedCutoff=1.0*units.nanometer,
                                 #constraints=app.HBonds,  # easy on the minimizer
                                 rigidWater=True)

integrator = mm.LangevinIntegrator(300*units.kelvin, 1.0/units.picosecond,
                                   2.0*units.femtosecond)
integrator.setRandomNumberSeed(cmd.seed)
integrator.setConstraintTolerance(0.00001)


# Add position restraints if requested
if cmd.posre:
    force = _restraints.make_heavy_atom_restraints(structure,
                                                   cmd.posre_K)
    system.addForce(force)

# Minimize
simulation = app.Simulation(structure.topology, system, integrator,
                            platform, plat_properties)
simulation.context.setPositions(structure.positions)

state = simulation.context.getState(getEnergy=True)
energy = state.getPotentialEnergy().value_in_unit_system(units.md_unit_system)
logging.info('Initial Potential Energy: {:10.3f}'.format(energy))

logging.info('Running minimization ...')
simulation.minimizeEnergy(maxIterations=cmd.iterations)

state = simulation.context.getState(getEnergy=True, getPositions=True)
energy = state.getPotentialEnergy().value_in_unit_system(units.md_unit_system)
logging.info('Final Potential Energy: {:10.3f}'.format(energy))

# Write minimized structure
if cmd.output:
    if not cmd.output.endswith('.cif'):
        _fname = cmd.output + '.cif'
    else:
        _fname = cmd.output
else:
    _fname = fname + '_EM' + '.cif'

cif_fname = _utils.make_fname(_fname)
logging.info('Writing structure to \'{}\''.format(cif_fname))
with open(cif_fname, 'w') as handle:
    minimized_positions = state.getPositions()
    app.PDBxFile.writeFile(structure.topology, minimized_positions, handle)
