#!/usr/bin/env python

"""
Runs production MD on an equilibrated (hopefully) system using an AMBER FF.

Starts from an equilibrated mmCIF structure and an OpenMM XML state file.

.2017. joaor@stanford.edu
"""

from __future__ import print_function, division

import argparse
import logging
import os
import random
import re
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


def get_part_filename(name, suffix='_part_'):
    """
    Function to find and rename file with same name. Meant for serial files
    """

    rootname, ext = os.path.splitext(name)
    # look for existing parts
    prev = [f for f in os.listdir('.')
            if f.startswith(rootname + suffix) and f.endswith(ext)]

    if prev:
        # Get last part number
        re_partnum = re.compile('{}([0-9]+)\{}'.format(rootname + suffix, ext))
        finder = lambda x: int(re_partnum.search(x).group(1))
        part_num = max([finder(f) for f in prev]) + 1
        return rootname + suffix + str(part_num) + ext

    elif os.path.isfile(name): # First part already there
        exst_part = rootname + suffix + '0'  + ext
        os.rename(name, exst_part)
        return rootname + suffix + '1' + ext

    else:
        return name


##
# Parse user input and options
ap = argparse.ArgumentParser(description=__doc__)

# Mandatory
ap.add_argument('structure', help='Input structure file (.cif)')
ap.add_argument('state', help='Input state file (.xml)')

# Options
ap.add_argument('--output', type=str, default=None,
                help='Root name for output files.')
ap.add_argument('--forcefield', type=str, default='amber99sbildn.xml',
                help='Force field to build the system with.')
ap.add_argument('--solvent', type=str, default='tip3p.xml',
                help='Solvent model to use in minimization.')

ap.add_argument('--write-frequency', dest='wfreq', type=int, default=None,
                help='Frequency (number of steps) to write DCD and checkpoint file(s).')
ap.add_argument('--log-frequency', dest='pfreq', type=int, default=None,
                help='Frequency (number of steps) to write simulation details (log file).')

ap.add_argument('--platform', type=str, default=None,
                help='Platform to run calculations on. Defaults to fastest available.')
ap.add_argument('--seed', type=int, default=917,
                help='Seed number for random number generator(s).')
ap.add_argument('--continuation', action='store_true',
                help='Look for existing checkpoint files to continue a previous run.')

ap.add_argument('--runtime', type=float, default=50.0,
                help='Time (in ns) to simulate system for.')
ap.add_argument('--temperature', default=300, type=float,
                help='Temperature at which to simulate the system, in Kelvin. Default is 300.')
ap.add_argument('--isobaric', action='store_true', help='Adds pressure coupling to the system (NpT ensemble)')
ap.add_argument('--pressure', default=1.0, type=float,
                help='Pressure at which to simulate the system, in bar. Default is 1.')
ap.add_argument('--hmr', action='store_true',
                help='Use Hydrogen Mass Repartitioning.')

cmd = ap.parse_args()

# Set random seed for reproducibility
random.seed(cmd.seed)

# Figure out platform
if cmd.platform is not None:
    cmd.platform = mm.Platform.getPlatformByName(cmd.platform)

logging.info('Started')
logging.info('Using:')
logging.info('  structure    : {}'.format(cmd.structure))
logging.info('  state        : {}'.format(cmd.state))
logging.info('  force field  : {}'.format(cmd.forcefield))
logging.info('  solvent model: {}'.format(cmd.solvent))
logging.info('  temperature  : {} K'.format(cmd.temperature))
logging.info('  pressure     : {} bar'.format(cmd.pressure))
logging.info('  random seed  : {}'.format(cmd.seed))
logging.info('  using HMR    : {}'.format('Yes' if cmd.hmr else 'No'))

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

if cmd.output:
    rootname = cmd.output
else:
    rootname = fname + '_Prod'

# Read structure
structure = app.PDBxFile(cmd.structure)
forcefield = app.ForceField(cmd.forcefield, cmd.solvent)

# Define simulation parameters
md_temp = cmd.temperature*units.kelvin
md_pres = cmd.pressure*units.bar

md_step = 2*units.femtosecond
md_fric = 1/units.picosecond
md_nbct = 1.0*units.nanometer
md_hamu = None
md_cstr = app.HBonds

if cmd.hmr:
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

if cmd.isobaric:
    logging.info('  NpT ensemble')
    system.addForce(mm.MonteCarloBarostat(md_pres, md_temp, 25))
else:
    logging.info('  NVT ensemble')

integrator = mm.LangevinIntegrator(md_temp, md_fric, md_step)
integrator.setRandomNumberSeed(cmd.seed)
integrator.setConstraintTolerance(0.00001)

# Setup simulation
simulation = app.Simulation(structure.topology, system, integrator,
                            cmd.platform, properties)
simulation.context.setPositions(structure.positions)
simulation.context.setVelocitiesToTemperature(md_temp)

# Is there a checkpoint file for this simulation?
run_cpt = rootname + '.cpt'
if cmd.continuation and os.path.isfile(run_cpt):
    logging.info('Found existing checkpoint file: \'{}\''.format(run_cpt))

    simulation.loadCheckpoint(run_cpt)

    # How much have we run for already?
    run_time = simulation.context.getState().getTime()
    run_time_val = run_time.value_in_unit(units.nanosecond)

    logging.info('  {:8.2f}/{:8.2f} ns completed'.format(run_time_val, cmd.runtime))

    # Adjust remaining time
    expected_t = cmd.runtime * units.nanosecond
    prod_time = (expected_t - run_time).in_units_of(units.nanosecond)
else:
    simulation.loadState(cmd.state)
    # reset time in context to avoid running from completed Eq
    logging.info('Resetting context time')
    simulation.context.setTime(0.0)
    prod_time = cmd.runtime * units.nanosecond

prod_time_val = prod_time.value_in_unit(units.nanosecond)
if prod_time_val <= 0.00001:
    logging.info('Simulation completed. Apparently. Maybe ask for more?')
    sys.exit(0)

logging.info('Running production simulation for {} ns'.format(prod_time_val))

n_steps = prod_time / md_step.value_in_unit(units.nanoseconds)
n_steps = int(n_steps.value_in_unit(units.nanoseconds)) + 1  # rounding errors

# Setup reporters
# Save DCD/CPT/etc every 0.01 ns
if cmd.hmr:
    # Time step is 5 fs
    # 200.000 steps give 1 ns
    wfreq = cmd.wfreq if cmd.wfreq is not None else 2000
    pfreq = cmd.pfreq if cmd.pfreq is not None else 2000
else:
    # Time step is 2 fs
    # 500.000 steps give 1 ns
    wfreq = cmd.wfreq if cmd.wfreq is not None else 5000
    pfreq = cmd.pfreq if cmd.pfreq is not None else 5000

run_dcd = rootname + '.dcd'
run_dcd = get_part_filename(run_dcd)

run_log = rootname + '.log'
run_log = get_part_filename(run_log)

dcd = app.DCDReporter(run_dcd, wfreq)
cpt = app.CheckpointReporter(run_cpt, wfreq)

state = app.StateDataReporter(run_log, pfreq,
                              step=True,
                              time=True,
                              progress=True,
                              potentialEnergy=True,
                              kineticEnergy=True,
                              temperature=True,
                              remainingTime=True,
                              speed=True,
                              totalSteps=n_steps, separator='\t')

simulation.reporters.append(dcd)
simulation.reporters.append(cpt)
simulation.reporters.append(state)

# Actually do stuff
simulation.step(n_steps)

# Write state
xml_fname = get_filename(rootname + '.xml')
logging.info('Writing state file to \'{}\''.format(xml_fname))
simulation.saveState(xml_fname)

# Write last frame as mmCIF
cif_fname = get_filename(rootname + '.cif')
logging.info('Writing last frame to \'{}\''.format(cif_fname))
with open(cif_fname, 'w') as handle:
    eq_xyz = simulation.context.getState(getPositions=True).getPositions()
    app.PDBxFile.writeFile(structure.topology, eq_xyz, handle)
