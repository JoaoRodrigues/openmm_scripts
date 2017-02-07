#!/usr/bin/env python

"""
Script to calculate boxes around a protein.
"""

from __future__ import print_function

import argparse
import sys

import numpy as np
from numpy.linalg import norm
import scipy.spatial.distance as ssd

_BOXES = ['dodecahedron_hx', 'dodecahedron_sq', 'octahedron', 'cube']

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('pdb', help='PDB File')
ap.add_argument('-p', '--pad', type=float, help='Distance to edge of box', default=0.0)
ap.add_argument('-b', '--box', type=str, help='Box Geometry', choices=_BOXES, default='cube')
cmd = ap.parse_args()

# Read PDB atoms to numpy array
atoms = []
with open(cmd.pdb) as handle:
    for line in handle:
        if line.startswith('ATOM'):
            x, y, z = [line[i:i+8] for i in range(30, 54, 8)]
            atoms.append((x,y,z))
atoms = np.array(atoms, dtype=np.float)
atoms /= 10 # A to nm

# Get dimensions, diameter, and center of mass
xyz_size = np.amax(atoms, axis=0) - np.amin(atoms, axis=0)
xyz_diam = np.max(ssd.pdist(atoms, 'euclidean')) # can be optimized with numba
xyz_com = np.sum(atoms, axis=0) / atoms.shape[0]

# Translate to origin
atoms -= xyz_com
assert np.allclose(np.sum(atoms, axis=0) / atoms.shape[0], np.array([0,0,0]))

# Calculate box vectors
# Formula from GMX manual
#
# where d is the diameter of the protein (plus padding)
d = xyz_diam + cmd.pad*2

# rhombic dodecahedron (xy-hexagon)
#  u = (d, 0, 0)
#  v = (d/2, sqrt(3)*d/2, 0)
#  w = (d/2, sqrt(3)*d/6, sqrt(6)*d/3)
if cmd.box == 'dodecahedron_hx':
    u = np.array((d, 0, 0))
    v = np.array((d/2, np.sqrt(3)*d/2, 0))
    w = np.array((d/2, np.sqrt(3)*d/6, np.sqrt(6)*d/3))
    vol = 0.5 * np.sqrt(2) * np.power(d, 3)

# rhombic dodecahedron (xy-square)
#  u = (d, 0, 0)
#  v = (0, d, 0)
#  w = (d/2, d/2, sqrt(2)*d/2)
elif cmd.box == 'dodecahedron_sq':
    u = np.array((d, 0, 0))
    v = np.array((0, d, 0))
    w = np.array((d/2, d/2, np.sqrt(2)*d/2))
    vol = 0.5 * np.sqrt(2) * np.power(d, 3)

# rhombic octahedron
#  u = (d, 0, 0)
#  v = (d/3, 2*sqrt(2)*d/3, 0)
#  w = (-d/3, sqrt(2)*d/3, sqrt(6)*d/3)
elif cmd.box == 'octahedron':
    u = np.array((d, 0, 0))
    v = np.array((d/3, 2*np.sqrt(2)*d/3, 0))
    w = np.array((-d/3, np.sqrt(2)*d/3, np.sqrt(6)*d/3))
    vol = 4.0/9.0 * np.sqrt(3) * np.power(d, 3)

elif cmd.box == 'cube':
    u = np.array((d, 0, 0))
    v = np.array((0, d, 0))
    w = np.array((0, 0, d))
    vol = np.power(d, 3)

# Box angles: alpha, beta, gamma
def vecangle(v1, v2):
    cos = np.dot(v1, v2) / (norm(v1) * norm(v2))
    return np.degrees(np.arccos(cos))

a = vecangle(u, v)
b = vecangle(u, w)
g = vecangle(v, w)

print('System Size      : {:6.3f} {:6.3f} {:6.3f}'.format(*xyz_size))
print('Center of Mass   : {:6.3f} {:6.3f} {:6.3f}'.format(*xyz_com))
print('Diameter         : {:6.3f}'.format(xyz_diam))
print('Box Vectors      : {:6.3f} {:6.3f} {:6.3f}'.format(norm(u), norm(v), norm(w)))
print('Box Angles       : {:6.2f} {:6.2f} {:6.2f}'.format(a, b, g))
print('Box Volume       : {:6.2f}'.format(vol))
print('Volume of Sphere : {:6.2f}'.format(4/3 * np.pi * np.power(d/2, 3)))
