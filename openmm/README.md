#OpenMM Python scripts

These folders contain various Python scripts to setup and run simulations
using OpenMM. Since some options are forcefield-specific, the scripts are
divided accordingly.

## AMBER FF
The scripts under the `amberff` folder are to be used with the AMBER family of
forcefields. By default, the scripts use `amberff14sb` and `tip3p` as the water
model. Lipids use the `lipid17` parameter set.

Each of the scripts performs a specific _task_ in the simulation setup and should
be used roughly in the following order:
- `buildSystem.py`: adds hydrogens and ensures atom/residue names are valid according
to the forcefield naming rules.
- `setPeriodicBox.py`: calculates the box vectors to create a periodic box that encloses
the system.
- `minimizeSystem.py`: performs restrained (heavy atoms only) energy minimization. To perform
an unrestrained EM, just set to force constant to 0.
- `solvateBox.py`/`addMembrane.py`: add solvent (water), counter-ions, and a membrane bilayer
to the system. Use one or the other.
- `heatSystem.py`: introduces velocities to the system by slowly increasing temperature up to
a desired value. You can control how slow/gradual this increase is.
- `equilibrate_NxT.py`: run restrained (by default) MD on a system, with/without pressure coupling.
- `runProduction.py`: runs unrestrained production MD.

You should prepare your system using the scripts in the `utils` folder, and then feed a mmCIF
to `buildSystem.py` to start your simulation setup.

# Known issues:
  - Cubic boxes only. Triclinic simulation boxes give weird issues.

