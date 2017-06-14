#!/usr/bin/env python

"""
Molecular Dynamics Simulation of a System.
System must have been prepared with GROMACS, using the CHARMM36 FF, solvated
and neutralized, but not minimized (not necessarily at least).

.2017. joaor@stanford.edu
"""

from __future__ import print_function, division

import argparse
import logging
import os
import random
import re
import sys

import simtk.openmm.app as app
import simtk.openmm as mm
import simtk.unit as units

##
# Parse user input and options
ap = argparse.ArgumentParser(description=__doc__)

# Mandatory
ap.add_argument('gro', help='Input coordinates file (.gro)')
ap.add_argument('top', help='Input topology file (.top)')

# Output Options
out_opts = ap.add_argument_group('Output Options')
out_opts.add_argument('--log', type=str, help='Log file name')

# Sim Options
sim_opts = ap.add_argument_group('Simulation Options')
gmx_top = os.path.join(os.getenv('GMXDATA') or '', 'top')
sim_opts.add_argument('--ffdir', default=gmx_top,
                      help='GROMACS top/ folder [default: {}]'.format(gmx_top))
sim_opts.add_argument('--seed', type=int, default=917,
                      help='Random Number Seed [default: 917]')
sim_opts.add_argument('--restart', action='store_true',
                      help='Ignores any Checkpoint files in the folder.')

sim_opts.add_argument('--equilibration', type=float, default=0.5,
                      help='Time in ns for equilibration [default: 5.0]')
sim_opts.add_argument('--production', type=float, default=5.0,
                      help='Time in ns for production [default: 10.0]')

opt_plat = ap.add_mutually_exclusive_group()
opt_plat.add_argument('--cpu', action="store_true", help='Use CPU platform')
opt_plat.add_argument('--cuda', action="store_true", help='Use CUDA platform')
ap.set_defaults(cpu=False, cuda=True)

user_args = ap.parse_args()

# Set random seed for all Python processed (integrators too, further below)
random.seed(user_args.seed)

# Format logger
if not user_args.log:
    logfile = sys.stdout
else:
    logfile = open(user_args.log, 'w')

logging.basicConfig(stream=logfile,
                    level=logging.INFO,
                    format='[%(asctime)s] %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')

logging.info('Starting Simulation')

# Print some info on system/versions
openmm_version = mm.Platform_getOpenMMVersion()
logging.info('OpenMM Version {}'.format(openmm_version))

# Define platform: CPU/CUDA
gpu_res = os.getenv('CUDA_VISIBLE_DEVICES')
cpu_res = os.getenv('SLURM_CPUS_PER_TASK')
properties = {}
if user_args.cuda:
    if not gpu_res:
        logging.error('No CUDA GPUs detected')
        sys.exit(1)

    platform = mm.Platform.getPlatformByName('CUDA')

    num_gpu = len(gpu_res.split(','))
    if num_gpu == 1:
        properties = {}
    else:
        logging.info('Using {}/{} CPUs/GPUs'.format(cpu_res, num_gpu))
        properties['DeviceIndex'] = gpu_res
else:
    platform = mm.Platform.getPlatformByName('CPU')
    properties['Threads'] = cpu_res

logging.info('Using platform: {}'.format(platform.getName()))

##
# Read structure, coordinates, and force field definitions
logging.info('Reading GRO file: {}'.format(user_args.gro))
if not os.path.isfile(user_args.gro):
    raise IOError('Could not read/open input file: {}'.format(user_args.gro))
else:
    rootname = os.path.basename(user_args.gro)[:-4]
    system_gro = app.GromacsGroFile(user_args.gro)
    box_vectors = system_gro.getPeriodicBoxVectors()

logging.info('Reading TOP file: {}'.format(user_args.top))
logging.info('Reading FF definitions from {}'.format(user_args.ffdir))
if not os.path.isdir(user_args.ffdir):
    raise IOError('Could not read/open folder: {}'.format(user_args.ffdir))

if not os.path.isfile(user_args.top):
    raise IOError('Could not read/open input file: {}'.format(user_args.top))
else:
    system_top = app.GromacsTopFile(user_args.top,
                                    periodicBoxVectors=box_vectors,
                                    includeDir=user_args.ffdir)


##
# Build System
logging.info('Building system objects')
system = system_top.createSystem(nonbondedMethod=app.PME,
                                 nonbondedCutoff=1.2*units.nanometers,
                                 constraints=app.HBonds)

# Add switching
# Is there a better way?
for force in system.getForces():
    if force.__class__.__name__ == 'NonBondedForce':
        force.setUseSwitchingFunction(True)
        force.setSwitchingDistance(1.0*units.nanometers)

n_atm = system_top.topology.getNumAtoms()
resname_list = [r.name for r in system_top.topology.residues()]
n_res = len(resname_list)
num_waters = resname_list.count('HOH')
num_cation = resname_list.count('NA')
num_anion = resname_list.count('CL')

logging.info('System: {:6d} Atoms {:6d} Residues'.format(n_atm, n_res))
logging.info('        num. waters: {:6d}'.format(num_waters))
logging.info('        num. ions: {:6d} Na {:6d} Cl'.format(num_cation, num_anion))  # noqa: E501

##
# Build System
md_temp = 300*units.kelvin
md_step = 0.002*units.picoseconds
md_fric = 1/units.picosecond
md_pres = 1*units.bar

integrator = mm.LangevinIntegrator(md_temp, md_fric, md_step)
integrator.setRandomNumberSeed(user_args.seed)
system.addForce(mm.MonteCarloBarostat(md_pres, md_temp, 25))

simulation = app.Simulation(system_top.topology, system, integrator,
                            platform=platform, platformProperties=properties)
simulation.context.setPositions(system_gro.positions)
simulation.context.setVelocitiesToTemperature(md_temp)
reporters = simulation.reporters

state = simulation.context.getState(getEnergy=True, getPositions=True)
pot_ene = state.getPotentialEnergy().value_in_unit_system(units.md_unit_system)
logging.info('Initial Potential Energy: {:10.3f}'.format(pot_ene))

##
# Continuation?
cpt_file = "{}_md.chk".format(rootname)
if not os.path.isfile(cpt_file) or user_args.restart:

    if os.path.isfile(cpt_file):
        os.remove(cpt_file)

    # Restrained Energy Minimization
    logging.info('Adding pos. res. on non-hydrogen protein atoms')
    all_atoms = list(system_top.topology.atoms())

    posre = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    posre_K = 10.0
    posre.addGlobalParameter("k", posre_K*units.kilojoule_per_mole/units.nanometer**2)  # noqa: E501
    posre.addPerParticleParameter("x0")
    posre.addPerParticleParameter("y0")
    posre.addPerParticleParameter("z0")

    solvent = set(('HOH', 'NA', 'CL'))
    n_at = 0
    hydrogen = app.element.hydrogen
    for i, atom_crd in enumerate(system_gro.getPositions()):
        at = all_atoms[i]
        if at.residue.name not in solvent and at.element != hydrogen:
            n_at += 1
            posre.addParticle(i, atom_crd.value_in_unit(units.nanometers))
    system.addForce(posre)
    logging.info('{}/{} atoms restrained (Fc={:8.3f} kJ/mol.nm^2)'.format(n_at, len(all_atoms), posre_K))  # noqa: E501

    logging.info('Running energy minimization')
    simulation.minimizeEnergy(maxIterations=1000,
                              tolerance=10*units.kilojoule/units.mole)

    state = simulation.context.getState(getEnergy=True, getPositions=True)
    pot_ene = state.getPotentialEnergy().value_in_unit_system(units.md_unit_system)  # noqa: E501
    logging.info('Potential Energy after minimization: {:10.3f}'.format(pot_ene))  # noqa: E501

    # Remove restraints
    logging.info('Releasing position restraints')
    system.removeForce(0)

    # Save minimized system to disk
    positions = state.getPositions()
    with open("{}_EM.pdb".format(rootname), 'w') as handle:
        app.PDBFile.writeFile(simulation.topology, positions, handle)

    ##
    # Setup reporters: logger and checkpointing
    reporters.append(app.StateDataReporter(logfile, 5000, step=True,
                                           time=True,
                                           potentialEnergy=True,
                                           kineticEnergy=True,
                                           totalEnergy=True, temperature=True,
                                           speed=True,
                                           separator='\t'))

    reporters.append(app.CheckpointReporter(cpt_file, 5000))  # Save every 10 ps # noqa: E501

    ##
    # Equilibration
    nvt_state = '{}_NVT.xml'.format(rootname)
    if os.path.isfile(nvt_state):
        logging.info('Found saved NVT equilibrated state: {}'.format(nvt_state))  # noqa: E501
        simulation.loadState(nvt_state)
    else:
        time_in_ns = user_args.equilibration
        n_of_steps = time_in_ns / (0.002/1000)
        logging.info('NPT equilibration system at 300K for {} ns'.format(time_in_ns))  # noqa: E501
        simulation.integrator.setTemperature(md_temp)
        simulation.step(n_of_steps)
        simulation.saveState('{}_NVT.xml'.format(rootname))

    ##
    # Production
    dcd_name = '{}_md_0.dcd'.format(rootname)
    time_in_ns = user_args.production


elif os.path.isfile(cpt_file) and not user_args.restart:
    # Load CPT file
    logging.info('Loading last checkpoint file: {}'.format(cpt_file))
    with open(cpt_file, 'rb') as handle:
        simulation.context.loadCheckpoint(handle.read())

    # Figure out how much time we have simulated already
    simulated_time = simulation.context.getState().getTime()
    simulated_time_in_ns = simulated_time.in_units_of(units.nanoseconds)

    # Adjust remaining simulation time
    total_time = user_args.production * units.nanosecond
    remaining_time = total_time - simulated_time_in_ns
    time_in_ns = remaining_time.value_in_unit(units.nanosecond)

    # Adjust dcd name
    dcd_regex = re.compile('{}_md'.format(rootname) + "_([0-9])+\.dcd")
    finder = lambda x: re.match(dcd_regex, x)  # noqa: E731
    extensions = set(map(finder, os.listdir('.'))) - set([None])
    if extensions:
        ext_no = map(int, [dcd.group(1) for dcd in extensions if dcd])
        last_no = int(sorted(ext_no)[-1]) + 1
        dcd_name = '{}_md_{}.dcd'.format(rootname, last_no)
    else:
        dcd_name = '{}_md_{}.dcd'.format(rootname, 0)

##
# Production
time_in_ns = round(time_in_ns, 5)
n_of_steps = int(time_in_ns / (0.002/1000))
simulation.reporters = []

if n_of_steps:
    reporters = simulation.reporters

    reporters.append(app.StateDataReporter(logfile, 5000, step=True,
                                           time=True, totalSteps=n_of_steps,
                                           potentialEnergy=True,
                                           kineticEnergy=True,
                                           totalEnergy=True, temperature=True,
                                           speed=True, progress=True,
                                           remainingTime=True, volume=True,
                                           separator='\t'))


    reporters.append(app.DCDReporter(dcd_name, 5000)) # 1o ps
    reporters.append(app.CheckpointReporter(cpt_file, 5000))  # Save every 10 ps
    logging.info('Running production simulation for {} ns'.format(time_in_ns))
    simulation.integrator.setTemperature(md_temp)
    simulation.step(n_of_steps)

logging.info('Simulation Finished')
logging.info('Saving xml state to file: {}_prod.xml'.format(rootname))
simulation.saveState('{}_prod.xml'.format(rootname))

logging.info('Bye bye..')
logging.shutdown()
