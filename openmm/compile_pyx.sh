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
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/home/stanford/levittm/shared/software/miniconda2/include/python2.7 -o ${pyxFile%%.pyx}.so ${pyxFile%%.pyx}.c

