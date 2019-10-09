#!/usr/bin/env python

"""
Runs a molecular dynamics simulation.

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

ap.add_argument('--state', type=str,
                help='Checkpoint/XML file to read positions/velocities from.')

ap.add_argument('--seed', type=int, default=917,
                help='Seed number for random number generator(s).')

ap.add_argument('--temperature', default=310, type=float,
                help='Target temperature, in Kelvin. Default is 310.')
ap.add_argument('--pressure', default=1.0, type=float,
                help='Target pressure, in bar. Default is 1.0.')
ap.add_argument('--barostat', default='isotropic',
                choices=('isotropic', 'membrane'),
                help='Type of barostat.')


ap.add_argument('--runtime', default=5, type=float,
                help='Simulation length in nanoseconds. Default 5.')
ap.add_argument('--continuation', action='store_true',
                help='Reads elapsed simulation time from checkpoint/state files.')

ap.add_argument('--restraint-heavy-atom', action='store_true', default=False,
                help='Apply position restraints to non-solvent heavy atoms')
ap.add_argument('--restraint-lipids', action='store_true', default=False,
                help='Apply position restraints to lipid head groups')
ap.add_argument('--restraint-heavy-atom-k', default=500, type=int,
                help='Force constant for heavy atom restraints. Default: 500')
ap.add_argument('--restraint-lipids-k', default=500, type=int,
                help='Force constant for lipid restraints. Default: 500')


ap.add_argument('--hmr', action='store_true', default=False,
                help='Use Hydrogen Mass Repartitioning.')
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
logging.info(f'  barostat     : {cmd.barostat}')
logging.info(f'  pressure     : {cmd.pressure} bar')
logging.info(f'  runtime      : {cmd.runtime} ns')
logging.info(f'  heavy-atom restraints : {cmd.restraint_heavy_atom}')
if cmd.restraint_heavy_atom:
    logging.info(f'    K = {cmd.restraint_heavy_atom_k} kJ/mol/nm^2')
logging.info(f'  lipid restraints      : {cmd.restraint_lipids}')
if cmd.restraint_lipids:
    logging.info(f'    K = {cmd.restraint_lipids_k} kJ/mol/nm^2')
logging.info(f'  HMR          : {cmd.hmr}')

# Make rootname for output files
basename = os.path.basename(cmd.structure)
fname, fext = os.path.splitext(basename)

if cmd.output is None:
    rootname = fname + '_EqNVT'
else:
    rootname = cmd.output

# Read in structure data and setup OpenMM system
structure = app.PDBxFile(cmd.structure)

# Remove dummy atoms (mass 0) just in case
model = app.Modeller(structure.topology, structure.positions)
dummy_idx = [a for a in model.topology.atoms() if a.element is None]
n_dummies = len(dummy_idx)
if n_dummies:
    logging.info(f'Removing {n_dummies} dummy atoms from input')
    model.delete(dummy_idx)
    structure.topology = model.topology
    structure.positions = model.positions

forcefield = app.ForceField(cmd.forcefield, cmd.solvent)

md_temp = cmd.temperature * units.kelvin

md_step = 2.0*units.femtosecond
md_fric = 1.0/units.picosecond
md_nbct = 1.0*units.nanometer
md_hamu = None
md_cstr = app.HBonds

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

# Setup pressure
md_pres = cmd.pressure * units.bar
if cmd.barostat == 'isotropic':
    b = mm.MonteCarloBarostat(md_pres, md_temp, 25)
elif cmd.barostat == 'membrane':
    surface_tension = 0*units.bar*units.nanometer  # amber lipids = tensionless
    b = mm.MonteCarloMembraneBarostat(md_pres, surface_tension,
                                      md_temp,
                                      mm.MonteCarloMembraneBarostat.XYIsotropic,
                                      mm.MonteCarloMembraneBarostat.ZFree,
                                      25)
system.addForce(b)

# Setup integrator and temperature coupling
integrator = mm.LangevinIntegrator(md_temp, md_fric, md_step)
integrator.setRandomNumberSeed(cmd.seed)
integrator.setConstraintTolerance(0.00001)

# Restrain heavy atoms
if cmd.restraint_heavy_atom:
    # force = _restraints.make_heavy_atom_restraints(structure,
    #                                                cmd.restraint_heavy_atom_k)
    force = _restraints.make_heavy_atom_restraints_v2(system, structure,
                                                      cmd.restraint_heavy_atom_k)
    system.addForce(force)

# Restrain lipid headgroups in Z
if cmd.restraint_lipids:
    # force = _restraints.make_lipid_restraints(structure,
    #                                           cmd.restraint_lipids_k)
    force = _restraints.make_lipid_restraints_v2(system, structure,
                                                 cmd.restraint_lipids_k)
    system.addForce(force)

# Setup simulation
simulation = app.Simulation(structure.topology, system, integrator,
                            platform, plat_properties)
simulation.context.setPositions(structure.positions)
simulation.context.setVelocitiesToTemperature(md_temp)

# Load checkpoint/state file
if cmd.state:
    if cmd.state.endswith('.xml'):  # is XML state file
        logging.info(f'Loading XML state file: {cmd.state}')
        simulation.loadState(cmd.state)
        logging.info(f'  resetting simulation time')
        simulation.context.setTime(0.0)  # resets simulation time
        cmd.runtime = cmd.runtime * units.nanosecond

    elif cmd.state.endswith('.cpt'):  # is binary checkpoint
        logging.info(f'Loading binary checkpoint file: {cmd.state}')
        simulation.loadCheckpoint(cmd.state)

        if cmd.continuation:
            # Adjust remaining running time
            run_time = simulation.context.getState().getTime()
            run_time_val = run_time.value_in_unit(units.nanosecond)
            logging.info(f'  {run_time_val:8.2f}/{cmd.runtime:8.2f} ns completed')

            expected_t = cmd.runtime * units.nanosecond
            cmd.runtime = (expected_t - run_time).in_units_of(units.nanosecond)
        else:  # restart from 0
            simulation.context.setTime(0.0)
            cmd.runtime = cmd.runtime * units.nanosecond
    else:
        raise Exception(f'State file format not recognized: {cmd.state}')
else:
    cmd.runtime = cmd.runtime * units.nanosecond

# Assert we actually have to run something.
if cmd.runtime <= 0.00001 * units.nanosecond:
    logging.info('Equilibration completed. Apparently. Maybe ask for more?')
    logging.info('Finished')
    sys.exit(0)

# Setup writer/logger frequencies
# Default: 0.01 ns
if cmd.hmr:
    # Time step is 5 fs
    xyz_freq = cmd.xyz_freq if cmd.xyz_freq is not None else 2000
    log_freq = cmd.log_freq if cmd.log_freq is not None else 2000
else:
    # Time step is 2 fs
    xyz_freq = cmd.xyz_freq if cmd.xyz_freq is not None else 5000
    log_freq = cmd.log_freq if cmd.log_freq is not None else 5000

# Calculate total simulation length in steps
n_steps = math.ceil(cmd.runtime / md_step.in_units_of(units.nanoseconds))
# n_steps is dimensionless (ns/ns)

# Setup Reporters
dcd_fname = _utils.make_fname_serial(rootname + '.dcd')
cpt_fname = _utils.make_fname_serial(rootname + '.cpt')
log_fname = _utils.make_fname_serial(rootname + '.log')
dcd = app.DCDReporter(dcd_fname, xyz_freq)
cpt = app.CheckpointReporter(cpt_fname, xyz_freq)
state = app.StateDataReporter(log_fname, log_freq,
                              step=True,
                              time=True,
                              potentialEnergy=True,
                              kineticEnergy=True,
                              temperature=True,
                              progress=True,
                              remainingTime=True,
                              volume=True,
                              totalSteps=n_steps,
                              speed=True,
                              separator='\t')


simulation.reporters.append(dcd)
simulation.reporters.append(cpt)
simulation.reporters.append(state)

logging.info(f'Writing coordinates to \'{dcd_fname}\'')
logging.info(f'Writing checkpoint file to \'{cpt_fname}\'')
logging.info(f'Writing simulation log to \'{log_fname}\'')


# Run simulation
simulation.step(n_steps)

# Write state file (without restraining forces)
xml_fname = _utils.make_fname_serial(rootname + '.xml')
logging.info(f'Writing state file to \'{xml_fname}\'')
system = simulation.system
n_rest_forces = sum([cmd.restraint_heavy_atom, cmd.restraint_lipids])
while n_rest_forces:
    system.removeForce(system.getNumForces() - 1)
    n_rest_forces -= 1

# Reinitialize context. Keep velocities, positions.
state = simulation.context.getState(getPositions=True, getVelocities=True)
vx, vy, vz = state.getPeriodicBoxVectors()
xyz, vel = state.getPositions(), state.getVelocities()
simulation.context.reinitialize(preserveState=False)
simulation.context.setPositions(xyz)
simulation.context.setVelocities(vel)
simulation.context.setPeriodicBoxVectors(vx, vy, vz)
simulation.saveState(xml_fname)

# Write last frame as mmCIF
cif_fname = _utils.make_fname_serial(rootname + '.cif')
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
simulation.context.setPeriodicBoxVectors(vx, vy, vz)

xml_fname = _utils.make_fname(rootname + '_noDUM' + '.xml')
logging.info(f'Writing dummy-less state to \'{xml_fname}\'')
simulation.saveState(xml_fname)

# Write last frame as mmCIF
cif_fname = _utils.make_fname(rootname + '_noDUM' + '.cif')
logging.info(f'Writing dummy-less structure to \'{cif_fname}\'')
with open(cif_fname, 'w') as handle:
    app.PDBxFile.writeFile(model.topology, xyz[:n_ini_atoms], handle, keepIds=True)

logging.info('Finished')

