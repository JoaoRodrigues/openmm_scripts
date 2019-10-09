#!/usr/bin/env python

"""
Shared utility functions across all scripts/modules.

.2019. joaor@stanford.edu
"""

from __future__ import print_function, division

import logging
import os
import re
import sys

import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as units

# Setup logger
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='[%(asctime)s] %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')

# IO/File Utilities


def make_fname(name):
    """Creates a unique file name for an output file.

    Creates backup files if the filename already exists,
    to prevent overwriting.

    Parameters
    ----------
    name (str): file name to create/check.

    Returns
    -------
    fname (str): validated unique file name.
    """

    if not os.path.isfile(name):
        return name

    fname = name
    num = 1
    while 1:
        name = '#{}.{}#'.format(fname, num)
        if os.path.isfile(name):
            num += 1
        else:
            os.rename(fname, name)
            break
    return fname


def make_fname_serial(name, suffix='_part_'):
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

# OpenMM Utilities


def get_platform(platform_name=None):
    """Creates a Platform object to run calculations on.


    Parameters
    ----------
    platform_fname (str): name of platform to create. One of:
                          CPU, CUDA, OPENCL, Reference. Leave empty (None)
                          to pick fastest available.

    Returns
    -------
    platform (Platform): OpenMM Platform object.

    properties (dict): dictionary containing property names and values for
                       a particular Platform. Examples include 'CudaPrecision'
                       and 'DeviceIndex' for CUDA, or 'Threads' for the CPU
                       platform. These values are obtained from environment
                       variables or set to defaults.
    """

    # Set platform
    if platform_name is None:  # pick fastest Platform
        p = None
        for idx in range(4):
            try:
                _p = mm.Platform.getPlatform(idx)
            except Exception as e:
                pass  # not all hardware supports all 4 platforms.
            else:
                if p is None:
                    p = _p
                else:
                    _ps = _p.getSpeed()
                    ps = p.getSpeed()
                    if _ps > ps:
                        p = _p
        if p is None:
            raise Exception('Could not automatically pick a Platform')

        pname = p.getName()
        logging.info(f'Automatically loaded Platform \'{pname}\'')
    else:
        try:
            p = mm.Platform.getPlatformByName(platform_name)
            pname = p.getName()
            logging.info(f'Loaded Platform \'{pname}\'')
        except Exception:
            raise Exception(f'Could not load Platform \'{platform_name}\'')

    # Set properties
    properties = {}
    if pname == 'CUDA':
        properties['DeterministicForces'] = 'True'  # slower, but better
        properties['CudaPrecision'] = 'mixed'  # default
        # CUDA_VISIBLE_DEVICES exposes only the GPU IDs to OpenMM.
        # These are 'renumbered', from zero, by the CUDA API.
        devices = os.getenv('CUDA_VISIBLE_DEVICES', default='0')
        devices_str = [str(i) for i, _ in enumerate(devices.split(','))]
        properties['DeviceIndex'] = ','.join(devices_str)
    elif pname == 'CPU':
        num_threads = os.getenv('OPENMM_CPU_THREADS', default='1')  # safe
        properties['Threads'] = num_threads

    return(p, properties)
