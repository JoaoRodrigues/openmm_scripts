#!/usr/bin/env python

"""
Molecular Dynamics Simulation of a System.

Uses a rhombic dodecahedron box and the Amber99sb-ILDN FF (w/ TIP3P).
Default padding of 1.1 nm from protein to edge of the box.
Neutralizes the system and adds ions to 0.15 M.

.2017. joaor@stanford.edu
"""

from __future__ import print_function, division

import argparse
import logging
import os
import random
import re
import sys
from datetime import datetime

import numpy as np

# cython-optmized pairwise distance function
# profiled to run in ~50% of the time of pdist
# much less memory hungry. No storage of all distances.
try:
    # add to PYTHONPATH current workdir and script dir
    sys.path.insert(0, os.curdir)
    sys.path.insert(0, os.path.dirname(__file__))

    from _pwdistance import pw_dist
    opt_pw = True
except ImportError, e:
    opt_pw = False
    from scipy.spatial.distance import pdist

    def pw_dist(xyz_array):
        return np.amax(pdist(xyz_array, 'euclidean'))

import simtk.openmm.app as app
import simtk.openmm as mm
import simtk.unit as units


def _compute_dof(system):
    # Compute the number of degrees of freedom.
    # For temperature reporting during MD
    dof = 0
    for i in range(system.getNumParticles()):
        if system.getParticleMass(i) > 0*units.dalton:
            dof += 3
    dof -= system.getNumConstraints()
    if any(type(system.getForce(i)) == mm.CMMotionRemover for i in range(system.getNumForces())):
        dof -= 3
    return dof


##
# Parse user input and options
ap = argparse.ArgumentParser(description=__doc__)

# Mandatory
ap.add_argument('pdb', help='Input coordinates file (.pdb)')

# Output Options
out_opts = ap.add_argument_group('Output Options')
out_opts.add_argument('--log', type=str, help='Log file name')

# Sim Options
sim_opts = ap.add_argument_group('Simulation Options')
sim_opts.add_argument('--pad', type=float, default=1.1,
                      help='Box Padding in nm [default: 1.1nm]')
sim_opts.add_argument('--seed', type=int, default=917,
                      help='Random Number Seed [default: 917]')
sim_opts.add_argument('--continuation', action='store_true',
                      help='Loads latest matching checkpoint file in the folder.')

sim_opts.add_argument('--ff', type=str, choices=['amber99sbildn.xml'], 
                      default='amber99sbildn.xml',
                      help='Force field to use in the simulation.')
sim_opts.add_argument('--solvent', type=str, choices=['tip3p.xml'],
                      default='tip3p.xml',
                      help='Solvent model to use in the simulation.')

sim_opts.add_argument('--hmr', action='store_true',
                      help='Make use of Hydrogen Mass Repartitioning to increase dt.')

sim_opts.add_argument('--equilibration', type=float, default=0.5,
                      help='Time in ns for equilibration [default: 0.5]')
sim_opts.add_argument('--production', type=float, default=5.0,
                      help='Time in ns for production [default: 5.0]')

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

logging.info('Started')

if not os.path.isfile(user_args.pdb):
    raise IOError('Could not read/open input file: {}'.format(user_args.pdb))
else:
    rootname = os.path.basename(user_args.pdb)[:-4]
    minimized_system = '{}_EM.pdb'.format(rootname)
    equilibrated_system = '{}_NVT.xml'.format(rootname)
    production_cpt = '{}_md.cpt'.format(rootname)
    production_dcd = '{}_md.dcd'.format(rootname)
    production_xml = '{}_md.xml'.format(rootname)

if not opt_pw:
    logging.warning('Using numpy (slower) pwdist routine for simulation setup.')

# Get openmm version
openmm_version = mm.Platform_getOpenMMVersion()

# Define platform: CPU/CUDA
gpu_res = os.getenv('CUDA_VISIBLE_DEVICES') or 0
cpu_res = os.getenv('SLURM_CPUS_PER_TASK') or 1
properties = {'Precision': 'single'}
if user_args.cuda:
    if not gpu_res:
        logging.error('No CUDA GPUs detected. Consider using the CPU platform.')
        sys.exit(1)

    platform = mm.Platform.getPlatformByName('CUDA')

    num_gpu = len(gpu_res.split(','))
    if num_gpu == 1:
        properties = {}
    else:
        properties['DeviceIndex'] = gpu_res
    res_str = '  using {}/{} CPUs/GPUs'.format(cpu_res, num_gpu)
else:
    platform = mm.Platform.getPlatformByName('CPU')
    res_str = '  using {} CPU threads'.format(cpu_res)
    properties['Threads'] = cpu_res

logging.info('OpenMM info:')
logging.info('  version {}'.format(openmm_version))
logging.info('  platform: {}'.format(platform.getName()))
logging.info(res_str)

##
# Read coordinates
logging.info('Building initial structure:')
logging.info('  coordinates from {}'.format(user_args.pdb))
logging.info('  \'{}\' force field'.format(user_args.ff))
logging.info('  \'{}\' water model'.format(user_args.solvent))

##
# Processing structure and build box if necessary
do_minimization = not user_args.continuation or not os.path.isfile(minimized_system)
if do_minimization:

    pdb = app.PDBFile(user_args.pdb)
    forcefield = app.ForceField(user_args.ff, user_args.solvent)

    logging.info('Adding hydrogen atoms and minimizing their positions')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield, pH=7.0, platform=platform)  # already does EM

    logging.info('Building periodic simulation box')
    # Build rhombic dodecahedron box (square xy-plane)
    # 0. Center system at origin
    com_xyz = modeller.positions.mean()
    for i, xyz_i in enumerate(modeller.positions):
        modeller.positions[i] = xyz_i - com_xyz

    # 1. Move coordinates to numpy array for efficiency
    _xyz = [(x._value, y._value, z._value) for x, y, z in modeller.positions]
    xyz = np.array(_xyz, dtype=np.float)
    xyz_size = np.amax(xyz, axis=0) - np.amin(xyz, axis=0)
    xyz_diam = pw_dist(xyz)

    d = xyz_diam + user_args.pad*2
    u = np.array((d, 0, 0))
    v = np.array((0, d, 0))
    w = np.array((d/2, d/2, np.sqrt(2)*d/2))
    box_vol = 0.5 * np.sqrt(2) * np.power(d, 3)

    n_solute_res = modeller.topology.getNumResidues()

    modeller.topology.setPeriodicBoxVectors((u, v, w))

    # Solvate the Box and add counter ions at 0.15 M
    logging.info('Adding solvent to periodic box (0.15M salt)')
    modeller.addSolvent(forcefield, model=user_args.solvent[:-4],
                        neutralize=True, ionicStrength=0.15*units.molar)

    n_atm = modeller.topology.getNumAtoms()
    resname_list = [r.name for r in modeller.topology.residues()]
    n_waters = resname_list.count('HOH')
    n_cation = resname_list.count('NA')
    n_anion = resname_list.count('CL')

    logging.info('System details:')
    logging.info('  num. atoms: {:6d}'.format(n_atm))
    logging.info('  num. residues: {:6d}'.format(n_solute_res))
    logging.info('  num. waters: {:6d}'.format(n_waters))
    logging.info('  num. ions: {:6d} Na {:6d} Cl'.format(n_cation, n_anion))
    logging.info('  Size        : {:6.3f} {:6.3f} {:6.3f}'.format(*xyz_size))
    logging.info('  Diameter    : {:6.3f}'.format(xyz_diam))
    logging.info('  Box Volume  : {:6.3f}'.format(box_vol))
    logging.info('  Box Vectors :')
    logging.info('    u = {:6.3f} {:6.3f} {:6.3f}'.format(*u))
    logging.info('    v = {:6.3f} {:6.3f} {:6.3f}'.format(*v))
    logging.info('    w = {:6.3f} {:6.3f} {:6.3f}'.format(*w))

    # trick?
    pdb.topology = modeller.topology
    pdb.positions = modeller.positions
else:
    logging.info('Loading existing minimized system:')
    logging.info('  {}'.format(minimized_system))

    pdb = app.PDBFile(minimized_system)
    forcefield = app.ForceField(user_args.ff, user_args.solvent)

##
# Build System
logging.info('Setting up simulation')

md_temp = 300*units.kelvin
md_step = 0.002*units.picoseconds
md_fric = 1/units.picosecond
md_pres = 1*units.bar
md_nbct = 1.0*units.nanometer
md_hamu = None
md_cstr = app.HBonds

if user_args.hmr:
    md_step *= 2.5  # make 5fs
    md_hamu = 4*units.amu
    md_cstr = app.AllBonds
    logging.info('  HMR  = True')

logging.info('  T    = {:4d} K'.format(md_temp.value_in_unit(units.kelvin)))
logging.info('  P    = {:4d} bar'.format(md_pres.value_in_unit(units.bar)))
logging.info('  dt   = {:3.1f} fs'.format(md_step.value_in_unit(units.femtosecond)))
logging.info('  nbct = {:3.1f} nm'.format(md_nbct.value_in_unit(units.nanometer)))

##
# Create System and setup Integrator
system = forcefield.createSystem(pdb.topology,
                                 nonbondedMethod=app.PME,
                                 nonbondedCutoff=md_nbct,
                                 constraints=md_cstr, 
                                 hydrogenMass=md_hamu,
                                 ewaldErrorTolerance=0.0005)

integrator = mm.LangevinIntegrator(md_temp, md_fric, md_step)
integrator.setRandomNumberSeed(user_args.seed)  # reproducibility
integrator.setConstraintTolerance(1e-5)

system.addForce(mm.MonteCarloBarostat(md_pres, md_temp, 25))  # barostat

##
# Create Context
context = mm.Context(system, integrator, platform, properties)
context.setPositions(pdb.positions)
context.setVelocitiesToTemperature(md_temp)

state = context.getState(getEnergy=True, getPositions=True)
pot_ene = state.getPotentialEnergy().value_in_unit_system(units.md_unit_system)
logging.info('Initial Potential Energy: {:10.3f}'.format(pot_ene))

##
# Starting from minimized system? Skip minimization
if do_minimization:
    n_iter = 100
    em_tol = 10

    logging.info('Performing Energy Minimization')
    logging.info('  max. iterations: {:5d}'.format(n_iter))
    logging.info('  e. tol: {:6.3f} kJ/mol'.format(em_tol))
    mm.LocalEnergyMinimizer.minimize(context, em_tol*units.kilojoule/units.mole, n_iter)

    state = context.getState(getEnergy=True, getPositions=True)
    pot_ene = state.getPotentialEnergy().value_in_unit_system(units.md_unit_system)
    logging.info('Potential Energy after minimization: {:10.3f}'.format(pot_ene))

    # Save minimized system to disk
    positions = state.getPositions()
    with open(minimized_system, 'w') as handle:
        app.PDBFile.writeFile(pdb.topology, positions, handle)

##
# Starting from equilibrated system? Skip equilibration
do_equilibration = not user_args.continuation or not os.path.isfile(equilibrated_system)
if do_equilibration:
    t_in_ns = user_args.equilibration
    total_steps = t_in_ns / md_step.value_in_unit(units.nanoseconds)
    logging.info('Equilibrating system under NPT:')
    logging.info('  {:6.3f} ns'.format(t_in_ns))

    current_step = 0
    _steps = 500
    reporter_freq = 500
    dof = _compute_dof(system) * units.MOLAR_GAS_CONSTANT_R

    logging.info('Progress\tTime (ps)\tTotal Energy (kJ/mol)\tTemperature (K)\tSpeed (ns/day)')
    while current_step < total_steps:
        steps_left = total_steps - current_step
        if steps_left < 0.0000001:
            break

        elif steps_left < _steps:
            _steps = steps_left

        start_time = datetime.now()
        integrator.step(_steps)
        end_time = datetime.now()

        current_step += _steps

        if current_step % reporter_freq == 0:
            state = context.getState(getEnergy=True)

            progress = current_step*100.0/total_steps
            sim_time = state.getTime().value_in_unit(units.picosecond)
            kin_ene = state.getKineticEnergy()
            pot_ene = state.getPotentialEnergy()
            tot_ene = (kin_ene + pot_ene).value_in_unit(units.kilojoules_per_mole)
            cur_temp = (2*kin_ene/dof).value_in_unit(units.kelvin)

            _elapsed = end_time - start_time
            elapsed = _elapsed.seconds + _elapsed.microseconds*1e-6
            speed = md_step.value_in_unit(units.nanosecond)*_steps*86400/elapsed

            logging.info('{:>5.1f}%\t{:>8.3f}\t{:8.3f}\t{:>8.2f}\t{:>6.2f}'.format(progress, sim_time, 
                                                                                  tot_ene, cur_temp, speed))

    # Save state to file
    state = context.getState(getPositions=True, getVelocities=True, getParameters=True)
    xml = mm.XmlSerializer.serialize(state)
    with open(equilibrated_system, 'w') as handle:
        handle.write(xml)

else:
    # Load saved state
    logging.info('Loading saved equilibration state:')
    logging.info('  {}'.format(equilibrated_system))
    context.reinitialize()
    with open(equilibrated_system) as handle:
        deserialized_xml = mm.XmlSerializer.deserialize(handle.read())
    context.setState(deserialized_xml)

##
# Production
total_time = user_args.production * units.nanoseconds
remainder = total_time

if user_args.continuation:
    # Look for checkpoint file and load.
    if os.path.isfile(production_cpt):
        logging.info('Loading checkpoint file:')
        logging.info('  {}'.format(production_cpt))
        
        with open(production_cpt, 'rb') as handle:
            context.loadCheckpoint(handle.read())

        elapsed_sim_time = context.getState().getTime()
        elapsed_ns = elapsed_sim_time.value_in_unit(units.nanosecond)
        total_ns = total_time.value_in_unit(units.nanosecond)
        logging.info('  {:8.2f}/{:8.2f} ns completed'.format(elapsed_ns, total_time.value_in_unit(units.nanosecond)))

        # Adjust remaining time
        remainder = (total_time - elapsed_sim_time).in_units_of(units.nanoseconds)
        if remainder.value_in_unit(units.nanosecond) < 0:
            logging.info('Production run finished. Maybe increase requested time?')
            sys.exit(1)

# Rename DCD file
dcd_rootname = production_dcd[:-4]  # remove .dcd
existing_dcd = [f for f in os.listdir('.') if f.endswith('.dcd')]
if len(existing_dcd) == 1:  # add part0 suffix
    prev_dcd = existing_dcd[0]
    new_fname = '{}_part0.dcd'.format(dcd_rootname)
    os.rename(prev_dcd, new_fname)
    production_dcd = '{}_part1.dcd'.format(dcd_rootname)

elif len(existing_dcd) > 1:  # check last suffix
    partno_regex = re.compile('_part([0-9]+)\.dcd')
    finder = lambda x: re.match(partno_regex, x).group(1)
    lstno = sorted(map(finder, existing_dcd))[0]
    production_dcd = '{}_part{}.dcd'.format(dcd_rootname, lstno + 1)

# Actually run the simulation
total_steps = remainder.value_in_unit(units.nanoseconds) / md_step.value_in_unit(units.nanoseconds)
logging.info('Running production simulation:')
logging.info('  saving to {}'.format(production_dcd))
logging.info('  {:6.3f} ns'.format(remainder.value_in_unit(units.nanoseconds)))

current_step = 0
_steps = 500

reporter_freq = 5000
dcd_freq = 5000
cpt_freq = 5000

dof = _compute_dof(system) * units.MOLAR_GAS_CONSTANT_R

_dcd_handle = open(production_dcd, 'wb')
dcd_file = app.DCDFile(_dcd_handle, pdb.topology, integrator.getStepSize(), 0, dcd_freq)

logging.info('Progress\tTime (ps)\tTotal Energy (kJ/mol)\tTemperature (K)\tSpeed (ns/day)')
while current_step < total_steps:

    steps_left = total_steps - current_step
    if steps_left < 0.0000001:
        break
    elif steps_left < _steps:
        _steps = steps_left

    start_time = datetime.now()
    integrator.step(_steps)
    end_time = datetime.now()

    current_step += _steps

    state = context.getState(getEnergy=True, getPositions=True)

    if current_step % reporter_freq == 0:

        progress = current_step*100.0/total_steps
        sim_time = state.getTime().value_in_unit(units.picosecond)
        kin_ene = state.getKineticEnergy()
        pot_ene = state.getPotentialEnergy()
        tot_ene = (kin_ene + pot_ene).value_in_unit(units.kilojoules_per_mole)
        cur_temp = (2*kin_ene/dof).value_in_unit(units.kelvin)

        _elapsed = end_time - start_time
        elapsed = _elapsed.seconds + _elapsed.microseconds*1e-6
        speed = md_step.value_in_unit(units.nanosecond)*_steps*86400/elapsed

        logging.info('{:>5.1f}%\t{:>8.3f}\t{:8.3f}\t{:>8.2f}\t{:>6.2f}'.format(progress, sim_time, 
                                                                                  tot_ene, cur_temp, speed))

    if current_step % cpt_freq == 0:
        # Save cpt to file
        with open(production_cpt, 'wn') as handle:
            handle.write(context.createCheckpoint())

    if current_step % dcd_freq == 0:
        dcd_file.writeModel(state.getPositions(), periodicBoxVectors=state.getPeriodicBoxVectors())

# Save at the very end
with open(production_cpt, 'wn') as handle:
    handle.write(context.createCheckpoint())

dcd_file.writeModel(state.getPositions(), periodicBoxVectors=state.getPeriodicBoxVectors())
_dcd_handle.close()

logging.info('Finished')
logging.shutdown()
