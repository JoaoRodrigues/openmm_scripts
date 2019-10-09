#!/usr/bin/env python

"""
OpenMM Restraint Forces shared between modules

.2019. joaor@stanford.edu
"""

from __future__ import print_function, division

import logging
import os
import sys

import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as units

# Setup logger
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='[%(asctime)s] %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')


LIPIDS = set(('POP', ))
WATERS = set(('HOH', ))


def make_heavy_atom_restraints(structure, k):
    """Creates a harmonic restraining force for non-solute heavy atoms.

    Parameters
    ----------
    structure (Structure): OpenMM object containing topology and positions
                                  for a particular 3D structure.

    k (int): force constant (in kcal/mol/A^2) for the resulting restraint.

    Returns
    -------
    force (CustomExternalForce): object corresponding to restraining force.
    """

    logging.info(f'Creating heavy atom restraint force (k={k})')

    # Harmonic restraint in X, Y, and Z
    expr = '0.5*k*periodicdistance(x, y, z, x0, y0, z0)^2'
    force_k = k * (units.kilocalorie_per_mole/units.angstrom**2)

    force = mm.CustomExternalForce(expr)
    force.addGlobalParameter("k", force_k)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    atom_list = list(structure.topology.atoms())
    n_atoms = len(atom_list)

    # Restraint heavy atoms only: C, O, N, S, and P
    elements = set((app.element.carbon, app.element.oxygen,
                    app.element.nitrogen, app.element.sulfur,
                    app.element.phosphorus))

    ATOMSET = WATERS | LIPIDS

    counter = 0
    for idx, atom_xyz in enumerate(structure.positions):
        atom = atom_list[idx]
        if atom.residue.name not in ATOMSET and atom.element in elements:
            atom_xyz_nm = atom_xyz.value_in_unit(units.nanometers)
            force.addParticle(idx, atom_xyz_nm)
            counter += 1

    logging.info(f'Restrained {counter} out of {n_atoms} atoms')
    return force


def make_heavy_atom_restraints_v2(system, structure, k=100):
    """Restraints protein heavy atoms using massless dummy atoms.

    Parameters
    ----------
    system (System): OpenMM System object.

    structure (Structure): OpenMM object containing topology and positions
                           for a particular 3D structure.

    k (int): force constant (in kJ/mol/nm^2) for the resulting restraint.

    Returns
    -------
    force (CustomExternalForce): object corresponding to restraining force.
    """

    logging.info(f'Creating heavy atom restraint force (k={k})')

    # Get nonbonded force
    nonbonded = None
    for f in system.getForces():
        if isinstance(f, mm.NonbondedForce):
            nonbonded = f
            break
    else:
        raise ValueError('Forcefield does not define a nonbonded Force')

    atom_list = list(structure.topology.atoms())
    n_atoms = len(atom_list)

    # Restraint heavy atoms only: C, O, N, S, and P
    elements = set((app.element.carbon, app.element.oxygen,
                    app.element.nitrogen, app.element.sulfur,
                    app.element.phosphorus))

    ATOMSET = WATERS | LIPIDS

    # Define harmonic bond force to bind dummies to atoms.
    force = mm.HarmonicBondForce()
    force.setUsesPeriodicBoundaryConditions(True)

    d_to_dummy = 0 * units.nanometers  # 0 distance between atom and dummy
    force_k = k * (units.kilojoules_per_mole/units.nanometer**2)

    # Add a chain to hold dummy atoms
    dummy_chain = structure.topology.addChain('DUMP')
    # Create container for residues
    seen_residues = set()

    res_counter, counter = 0, 0
    for idx, atom_xyz in enumerate(structure.positions[:]):
        atom = atom_list[idx]
        if atom.residue.name not in ATOMSET and atom.element in elements:

            dummy = system.addParticle(0)  # massless
            nonbonded.addParticle(0, 1, 0)  # no interactions
            nonbonded.addException(idx, dummy, 0, 1, 0)  # same as above
            force.addBond(idx, dummy, d_to_dummy, force_k)
            counter += 1

            # add dummy atom to structure.positions
            structure.positions.append(atom_xyz)

            # add dummy to topology
            if atom.residue not in seen_residues:
                rname = f'D{res_counter}'
                dummy_res = structure.topology.addResidue(rname, dummy_chain)
                res_counter += 1
                seen_residues.add(atom.residue)

            structure.topology.addAtom('DUM', None, dummy_res)

    logging.info(f'Restrained {counter} out of {n_atoms} atoms')

    return force


def make_lipid_restraints(structure, k):
    """Creates a harmonic restraining force for lipid headgroups in Z.

    Restraints phosphate atoms in Z dimension only. Currently supports
    POPC and POPE lipids (because of residue name matching.)

    Parameters
    ----------
    structure (Structure): OpenMM object containing topology and positions
                                  for a particular 3D structure.

    k (int): force constant (in kcal/mol/A^2) for the resulting restraint.

    Returns
    -------
    force (CustomExternalForce): object corresponding to restraining force.
    """

    logging.info(f'Creating lipid headgroup restraint force (k={k})')

    # Harmonic restraint in X, Y, and Z
    expr = '0.5*k*(z - z0)^2'
    force_k = k * (units.kilocalorie_per_mole/units.angstrom**2)

    force = mm.CustomExternalForce(expr)
    force.addGlobalParameter("k", force_k)
    force.addPerParticleParameter("z0")

    atom_list = list(structure.topology.atoms())
    n_atoms = len(atom_list)

    elemP = app.element.phosphorus

    counter = 0
    for idx, atom_xyz in enumerate(structure.positions):
        atom = atom_list[idx]
        if atom.residue.name.startswith('POP') and atom.element == elemP:
            atom_z_nm = [atom_xyz.value_in_unit(units.nanometers)[2]]  # list
            force.addParticle(idx, atom_z_nm)
            counter += 1

    logging.info(f'Restrained {counter} out of {n_atoms} atoms')
    return force


def make_lipid_restraints_v2(system, structure, k=100):
    """Restraints lipid phosphate atoms using massless dummy atoms.

    Parameters
    ----------
    system (System): OpenMM System object.

    structure (Structure): OpenMM object containing topology and positions
                           for a particular 3D structure.

    k (int): force constant (in kJ/mol/nm^2) for the resulting restraint.

    Returns
    -------
    force (CustomExternalForce): object corresponding to restraining force.
    """

    _LIPIDS = LIPIDS  # move to local scope

    logging.info(f'Creating lipid headgroup restraint force (k={k})')

    # Get nonbonded force
    nonbonded = None
    for f in system.getForces():
        if isinstance(f, mm.NonbondedForce):
            nonbonded = f
            break
    else:
        raise ValueError('Forcefield does not define a nonbonded Force')

    atom_list = list(structure.topology.atoms())
    n_atoms = len(atom_list)

    # Restraint heavy atoms only: C, O, N, S, and P
    elements = set((app.element.phosphorus, ))

    # Define harmonic bond force to bind dummies to atoms.
    force = mm.HarmonicBondForce()
    force.setUsesPeriodicBoundaryConditions(True)

    d_to_dummy = 0 * units.nanometers  # 0 distance between atom and dummy
    force_k = k * (units.kilojoules_per_mole/units.nanometer**2)

    # Add a chain to hold dummy atoms
    dummy_chain = structure.topology.addChain('DUML')
    # Create container for residues
    seen_residues = set()

    res_counter, counter = 0, 0
    for idx, atom_xyz in enumerate(structure.positions[:]):
        atom = atom_list[idx]
        if atom.residue.name in _LIPIDS and atom.element in elements:

            dummy = system.addParticle(0)  # massless
            nonbonded.addParticle(0, 1, 0)  # no interactions
            nonbonded.addException(idx, dummy, 0, 1, 0)  # same as above
            force.addBond(idx, dummy, d_to_dummy, force_k)
            counter += 1

            # add dummy atom to structure.positions
            structure.positions.append(atom_xyz)

            # add dummy to topology
            if atom.residue not in seen_residues:
                rname = f'D{res_counter}'
                dummy_res = structure.topology.addResidue(rname, dummy_chain)
                res_counter += 1
                seen_residues.add(atom.residue)

            structure.topology.addAtom('DUM', None, dummy_res)

    logging.info(f'Restrained {counter} out of {n_atoms} atoms')

    return force