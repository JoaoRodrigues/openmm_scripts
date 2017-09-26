#!/usr/bin/env bash

module load GCC
module load binutils/2.25

if [ "$#" -ne 1 ]
then
  echo "usage: $0 source.pyx"
  exit 1
fi

pyxFile=$1
if [ ! -f ${pyxFile%%.pyx}.c ]
then
  cython -a $pyxFile
fi

PYTHON_INCLUDE=$( python -c "from distutils import sysconfig; print sysconfig.get_python_inc()")
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I${PYTHON_INCLUDE} -o ${pyxFile%%.pyx}.so ${pyxFile%%.pyx}.c

