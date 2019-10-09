#!/usr/bin/env python

"""
Introduces velocities/kinetic energy in a system, to match a given
temperature (in Kelvin). Equivalent to an NVT equilibration with
position restraints on non-solvent heavy atoms. Lipid bilayers
are restrained on phosphate atoms in the Z dimension only.

Raises the temperature of the system gradually up to the desired
target value (e.g. 50, 100, 150, 200, ..., 300, 310), running a
number of MD steps at each stage.

Outputs a portable state (.xml) file with positions and velocities,
to allow restarting and/or continuation.

.2019. joaor@stanford.edu
"""

from __future__ import print_function, division

import argparse
import logging
import math
import os
import random
import re
import sys

import numpy as np

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
                help='Root name for output files. Default is input file name.')
ap.add_argument('--forcefield', type=str, default='amber14-all.xml',
                help='Force field to build the system with (XML format).')
ap.add_argument('--solvent', type=str, default='amber14/tip3p.xml',
                help='Solvent model to use in minimization (XML format).')

ap.add_argument('--xyz-frequency', dest='xyz_freq', type=int, default=None,
                help='Frequency (number of steps) to write coordinates.')
ap.add_argument('--log-frequency', dest='log_freq', type=int, default=None,
                help='Frequency (number of steps) to log run parameters.')

ap.add_argument('--platform', type=str, default=None,
                choices=('OpenCL', 'CUDA', 'CPU', 'Reference'),
                help='Platform to run calculations on.')

ap.add_argument('--seed', type=int, default=917,
                help='Seed number for random number generator(s).')

ap.add_argument('--temperature', default=310, type=float,
                help='Target temperature, in Kelvin. Default is 310.')
ap.add_argument('--ladder-step-temperature', default=50, type=int,
                help='Temperature increase per heating stage. Default is 50 K')
ap.add_argument('--ladder-num-steps', default=1000, type=int,
                help='Number of MD steps per heating stage. Default is 1000')

ap.add_argument('--restraint-k', default=500, type=int,
                help='Force constant for position restraints. Default is 500')

ap.add_argument('--hmr', action='store_true', default=False,
                help='Use Hydrogen Mass Repartitioning.')
ap.add_argument('--membrane', action='store_true', default=False,
                help='Enables options for membranes, e.g. restraints, tension')

ap.add_argument('--gentle', action='store_true', default=False,
                help='Auto settings: lower ladder-step-temp and longer -num-steps')

cmd = ap.parse_args()

logging.info('Started')

# Set random seed for reproducibility
random.seed(cmd.seed)

# Figure out platform
platform, plat_properties = _utils.get_platform(cmd.platform)

logging.info('Simulation Details:')
logging.info(f'  random seed  : {cmd.seed}')
logging.info(f'  structure    : {cmd.structure}')
logging.info(f'  force field  : {cmd.forcefield}')
logging.info(f'  solvent model: {cmd.solvent}')
logging.info(f'  temperature  : {cmd.temperature} K')
logging.info(f'  restraints K : {cmd.restraint_k} kcal/mol/A^2')
logging.info(f'  membrane     : {cmd.membrane}')
logging.info(f'  HMR          : {cmd.hmr}')

# Make rootname for output files
basename = os.path.basename(cmd.structure)
fname, fext = os.path.splitext(basename)

if cmd.output is None:
    rootname = fname + '_Heat'
else:
    rootname = cmd.output

# Read in structure data and setup OpenMM system
structure = app.PDBxFile(cmd.structure)
forcefield = app.ForceField(cmd.forcefield, cmd.solvent)

md_temp = cmd.ladder_step_temperature * units.kelvin  # initial T

md_step = 2.0*units.femtosecond
md_fric = 1.0/units.picosecond
md_nbct = 1.0*units.nanometer
md_hamu = None
md_cstr = app.HBonds
surface_tension = 0*units.bar*units.nanometer  # amber lipids are tensionless
if cmd.hmr:  # adapt for HMR if necessary
    md_step *= 2.5  # make 5 fs
    md_hamu = 4*units.amu
    md_cstr = app.AllBonds

# Build system & integrator
logging.info('Setting up system and integrator')
system = forcefield.createSystem(structure.topology, nonbondedMethod=app.PME,
                                 nonbondedCutoff=md_nbct,
                                 constraints=md_cstr,
                                 hydrogenMass=md_hamu,
                                 ewaldErrorTolerance=0.0005,
                                 rigidWater=True)

integrator = mm.LangevinIntegrator(md_temp, md_fric, md_step)
integrator.setRandomNumberSeed(cmd.seed)
integrator.setConstraintTolerance(0.00001)

# Restraint heavy atoms
# force = _restraints.make_heavy_atom_restraints(structure, cmd.restraint_k)
force = _restraints.make_heavy_atom_restraints_v2(system, structure,
                                                  cmd.restraint_k)
system.addForce(force)

# Restraint lipid headgroups in Z
if cmd.membrane:
    # force = _restraints.make_lipid_restraints(structure, cmd.restraint_k)
    force = _restraints.make_lipid_restraints_v2(system, structure,
                                                 cmd.restraint_k)

    system.addForce(force)

# Setup simulation
simulation = app.Simulation(structure.topology, system, integrator,
                            platform, plat_properties)
simulation.context.setPositions(structure.positions)
simulation.context.setVelocitiesToTemperature(md_temp)

# Setup writer/logger frequencies
freq = max(1, math.floor(cmd.ladder_num_steps / 10))
if cmd.hmr:
    # Time step is 5 fs
    xyz_freq = cmd.xyz_freq if cmd.xyz_freq is not None else freq
    log_freq = cmd.log_freq if cmd.log_freq is not None else freq
else:
    # Time step is 2 fs
    xyz_freq = cmd.xyz_freq if cmd.xyz_freq is not None else freq
    log_freq = cmd.log_freq if cmd.log_freq is not None else freq

# Calculate total simulation length in steps
n_stages = math.ceil(cmd.temperature / cmd.ladder_step_temperature)
n_total_steps = n_stages * cmd.ladder_num_steps

# Setup Reporters
dcd_fname = _utils.make_fname(rootname + '.dcd')
cpt_fname = _utils.make_fname(rootname + '.cpt')
log_fname = _utils.make_fname(rootname + '.log')
dcd = app.DCDReporter(dcd_fname, xyz_freq)
cpt = app.CheckpointReporter(cpt_fname, xyz_freq)
state = app.StateDataReporter(log_fname, log_freq,
                              step=True,
                              potentialEnergy=True,
                              kineticEnergy=True,
                              temperature=True,
                              progress=True,
                              remainingTime=True,
                              totalSteps=n_total_steps,
                              speed=True,
                              separator='\t')


simulation.reporters.append(dcd)
simulation.reporters.append(cpt)
simulation.reporters.append(state)

logging.info(f'Writing coordinates to \'{dcd_fname}\'')
logging.info(f'Writing checkpoint file to \'{cpt_fname}\'')
logging.info(f'Writing simulation log to \'{log_fname}\'')


if cmd.gentle:  # for tricky systems
  cmd.ladder_step_temperature = 10
  cmd.ladder_num_steps = 2000

# Run simulation
counter = 0
cur_temp = 0
while 1:
    counter += 1
    cur_temp += cmd.ladder_step_temperature
    cur_temp = min(cmd.temperature, cur_temp)
    logging.info(f'Stage {counter}/{n_stages}: heating system to {cur_temp}K')

    simulation.integrator.setTemperature(cur_temp * units.kelvin)
    simulation.step(cmd.ladder_num_steps)

    if cur_temp >= cmd.temperature:
        break

# Write state file (without restraining forces)
xml_fname = _utils.make_fname(rootname + '.xml')
logging.info(f'Writing state file to \'{xml_fname}\'')
system = simulation.system
n_rest_forces = 1
if cmd.membrane:
    n_rest_forces += 1
while n_rest_forces:
    system.removeForce(system.getNumForces() - 1)
    n_rest_forces -= 1

# Reinitialize context. Keep velocities, positions.
state = simulation.context.getState(getPositions=True, getVelocities=True)
xyz, vel = state.getPositions(), state.getVelocities()
simulation.context.reinitialize(preserveState=False)
simulation.context.setPositions(xyz)
simulation.context.setVelocities(vel)

simulation.saveState(xml_fname)

# Write last frame as mmCIF
cif_fname = _utils.make_fname(rootname + '.cif')
logging.info(f'Writing final structure to \'{cif_fname}\'')
with open(cif_fname, 'w') as handle:
    app.PDBxFile.writeFile(structure.topology, xyz, handle, keepIds=True)

# Write system without dummy atoms
# Easier to redo system object
# and set positions/velocities manually.
model = app.Modeller(structure.topology, structure.positions)
dummy = [c for c in model.topology.chains() if c.id.startswith('DUM')]
model.delete(dummy)  # delete entire chains

n_ini_atoms = model.topology.getNumAtoms()

logging.info('Writing system without dummy (restraint) atoms')
system = forcefield.createSystem(model.topology, nonbondedMethod=app.PME,
                                 nonbondedCutoff=md_nbct,
                                 constraints=md_cstr,
                                 hydrogenMass=md_hamu,
                                 ewaldErrorTolerance=0.0005,
                                 rigidWater=True)

integrator = mm.LangevinIntegrator(md_temp, md_fric, md_step)
simulation = app.Simulation(model.topology, system, integrator)
simulation.context.setPositions(xyz[:n_ini_atoms])
simulation.context.setVelocities(vel[:n_ini_atoms])

xml_fname = _utils.make_fname(rootname + '_noDUM' + '.xml')
logging.info(f'Writing dummy-less state to \'{xml_fname}\'')
simulation.saveState(xml_fname)

# Write last frame as mmCIF
cif_fname = _utils.make_fname(rootname + '_noDUM' + '.cif')
logging.info(f'Writing dummy-less structure to \'{cif_fname}\'')
with open(cif_fname, 'w') as handle:
    app.PDBxFile.writeFile(model.topology, xyz[:n_ini_atoms], handle, keepIds=True)

logging.info('Finished')
