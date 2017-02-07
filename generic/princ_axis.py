#!/usr/bin/env python

"""
Calculates the inertia tensor of a protein and aligns 
the largest principal component with the reference Z axis.

Uses Ca only.

.2017. Joao Rodrigues
"""

from __future__ import print_function

import argparse
import mdtraj as md
import numpy as np

# Atomic mass
#_MASS_C = 12.01070
_MASS_C = 13.019
#_MASS_C = 1.008

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('pdb', help='Input PDB file')
cmd = ap.parse_args()

# Read PDB Ca atoms to numpy array
pdb = md.load(cmd.pdb)
ca_pdb = pdb.atom_slice(pdb.topology.select('name CA'))
atoms = ca_pdb.xyz[0]
print('[=] Read {} CA atoms'.format(len(atoms)))

# Center coordinates on origin
com_xyz = atoms.mean(axis=0)
atoms -= com_xyz
print('[=] Shifting coordinates to origin: {:10.3f} {:10.3f} {:10.3f}'.format(*com_xyz))

# Calculate inertia tensor
# formulae from wikipedia (center of mass on origin)
ixx = _MASS_C * np.sum(atoms[:,1] ** 2 + atoms[:,2] ** 2)
iyy = _MASS_C * np.sum(atoms[:,0] ** 2 + atoms[:,2] ** 2)
izz = _MASS_C * np.sum(atoms[:,0] ** 2 + atoms[:,1] ** 2)
ixy = iyx = _MASS_C * -1 * np.sum(atoms[:,0] * atoms[:,1])
ixz = izx = _MASS_C * -1 * np.sum(atoms[:,0] * atoms[:,2])
iyz = izy = _MASS_C * -1 * np.sum(atoms[:,1] * atoms[:,2])

print('[=] Inertia Tensor')
print('  [ {:10.3f} {:10.3f} {:10.3f} ]'.format(ixx, ixy, ixz))
print('  [ {:10.3f} {:10.3f} {:10.3f} ]'.format(iyx, iyy, iyz))
print('  [ {:10.3f} {:10.3f} {:10.3f} ]'.format(izx, izy, izz))

# Diagonalize the tensor to get principal axes
I = np.array([ [ixx, ixy, ixz],
               [iyx, iyy, iyz],
               [izx, izy, izz] ])

eval, evec = np.linalg.eig(I)
print('[=] Principal Axes')
for ax in range(3):
    print('  [ {0[0]:10.3f} {0[1]:10.3f} {0[2]:10.3f} ] ({1:10.3f})'.format(evec[:,ax], eval[ax]))

# Get largest princ. axis
lpc_i = np.argmax(eval)
lpc = evec[:,lpc_i]

print('[=] Largest principal axis: [ {0[0]:10.3f} {0[1]:10.3f} {0[2]:10.3f} ] ({1:10.3f})'.format(lpc, eval[lpc_i]))
# Build rotation matrix to align onto target vector (0,0,1)
# based on GMX Code
targetvec = np.array([0,0,1], dtype=np.float)
costheta = np.dot(lpc, targetvec) / ((np.linalg.norm(lpc) * np.linalg.norm(targetvec)))

sintheta = np.sqrt(1 - costheta**2)
rotvec = np.cross(lpc, targetvec) / np.linalg.norm(np.cross(lpc, targetvec))
print('[=] Rotation Axis: [ {0[0]:10.3f} {0[1]:10.3f} {0[2]:10.3f} ])'.format(rotvec))

ux, uy, uz = rotvec

R = np.empty((3,3), dtype=np.float)
R[0][0] = ux**2 + (1.0 - ux**2) * costheta
R[0][1] = ux * uy * (1 - costheta) - (uz * sintheta)
R[0][2] = ux * uz * (1 - costheta) + (uy * sintheta)
R[1][0] = ux * uy * (1 - costheta) + (uz * sintheta)
R[1][1] = uy**2 + (1 - uy**2) * costheta
R[1][2] = uy * uz * (1 - costheta) - (ux * sintheta)
R[2][0] = ux * uz * (1 - costheta) - (uy * sintheta)
R[2][1] = uy * uz * (1 - costheta) + (ux * sintheta)
R[2][2] = uz**2 + (1 - uz**2) * costheta

print('[=] Rotation Matrix about axis ({} {} {})'.format(*targetvec))
print('  [ {:10.3f} {:10.3f} {:10.3f} ]'.format(R[0][0], R[0][1], R[0][2]))
print('  [ {:10.3f} {:10.3f} {:10.3f} ]'.format(R[1][0], R[1][1], R[1][2]))
print('  [ {:10.3f} {:10.3f} {:10.3f} ]'.format(R[2][0], R[2][1], R[2][2]))

# Output rotated structure
pdb.xyz[0] = np.dot(pdb.xyz[0], R.T)
pdb.save('{}_alnZ.pdb'.format(cmd.pdb[:-4]))
