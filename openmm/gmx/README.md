# OpenMM Scripts to run MD from GMX files prepared with the CHARMM force field.

# Examples
To equilibrate a system out of the box:  
`python equilibrateSystem.py protein.gro protein.top`

To add position restraints to solute heavy atoms:  
`python equilibrateSystem.py protein.gro protein.top --posre heavy-atoms`

To change the force constant of the restraints (default is 1000 kJ.mol-1.nm-2):  
`python equilibrateSystem.py protein.gro protein.top --posre heavy-atoms --posre_K 100`

To simulate under constant pressure (isotropic!):  
`python equilibrateSystem.py protein.gro protein.top --isobaric`

For other options:  
`python equilibrateSystem.py -h`
