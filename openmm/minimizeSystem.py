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

import numpy as np

import simtk.openmm.app as app
import simtk.openmm as mm
import simtk.unit as units

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
ap.add_argument('structure', help='Input coordinate file (.cif)')
# Options
ap.add_argument('--output', type=str, default=None,
                help='File name for minimized structure. Will *always* use mmCIF format.')
ap.add_argument('--forcefield', type=str, default='amber99sbildn.xml',
                help='Force field to build the system with.')
ap.add_argument('--solvent', type=str, default='tip3p.xml',
                help='Solvent model to use in minimization.')
ap.add_argument('--platform', type=str, default=None,
                help='Platform to run calculations on. Defaults to fastest available.')
ap.add_argument('--seed', type=int, default=917,
                help='Seed number for random number generator(s).')

ap.add_argument('--posre', choices=['heavy-atoms'], type=str,
                help='Add position restraints to a specific group of atoms. Solvent is *never* restrained.')
ap.add_argument('--posre_K', default=1000.0, type=float,
                help='Force constant for position restraints in kJ.mol/nm^2. Default is 1000.')
ap.add_argument('--iterations', default=100, type=int,
                help='Number of iterations to minimize for. Default is 100.')

cmd = ap.parse_args()

# Set random seed for reproducibility
random.seed(cmd.seed)

# Figure out platform
if cmd.platform is not None:
    cmd.platform = mm.Platform.getPlatformByName(cmd.platform)

logging.info('Started')
logging.info('Using:')
logging.info('  initial structure: {}'.format(cmd.structure))
logging.info('  force field: {}'.format(cmd.forcefield))
logging.info('  random seed: {}'.format(cmd.seed))

# Set platform-specific properties
properties = {}
if cmd.platform:
    platform_name = cmd.platform.getName()
    logging.info('  platform: {}'.format(platform_name))

    if platform_name == 'CUDA':
        properties = {'CudaPrecision': 'mixed'}

        # Slurm sets this sometimes
        gpu_ids = os.getenv('CUDA_VISIBLE_DEVICES')
        if gpu_ids:
            properties['DeviceIndex'] = gpu_ids

    elif platform_name == 'CPU':
        cpu_threads = os.getenv('SLURM_CPUS_PER_TASK')
        if cpu_threads:
            properties['Threads'] = cpu_threads

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
    logging.info('Adding harmonic position restraints on subset: {}'.format(cmd.posre))
    logging.info('  K = {:8.2f}'.format(cmd.posre_K))
    # Harmonic restraint
    posre = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    posre.addGlobalParameter("k", cmd.posre_K * (units.kilojoule_per_mole/units.nanometer**2))
    posre.addPerParticleParameter("x0")
    posre.addPerParticleParameter("y0")
    posre.addPerParticleParameter("z0")

    # Define subset of atoms to restraint
    if cmd.posre == 'heavy-atoms':
        # Lets assume simple things
        solvent = set(('HOH', 'NA', 'CL'))
        _elem_H = app.element.hydrogen
        all_atoms = list(structure.topology.atoms())

        n_posre_at = 0
        for i, atom_crd in enumerate(structure.positions):
            at = all_atoms[i]
            if at.residue.name not in solvent and at.element != _elem_H:
                n_posre_at += 1
                posre.addParticle(i, atom_crd.value_in_unit(units.nanometers))

        system.addForce(posre)
        logging.info('Restrained {} out of {} atoms'.format(n_posre_at, i))

# Minimize
simulation = app.Simulation(structure.topology, system, integrator,
                            cmd.platform, properties)
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

cif_fname = get_filename(_fname)
logging.info('Writing structure to \'{}\''.format(cif_fname))
with open(cif_fname, 'w') as handle:
    minimized_positions = state.getPositions()
    app.PDBxFile.writeFile(structure.topology, minimized_positions, handle)
