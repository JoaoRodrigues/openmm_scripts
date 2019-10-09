#!/usr/bin/env bash
#
# BASH Script to run soluble isotropic simulation using
# amberff14sb.

# Set seed for random number generation
# Change this between different replicates
# MD is deterministic
SEED=917

# Set devices available for simulations
# Only useful for CUDA platform
export CUDA_VISIBLE_DEVICES="0"

# Set shortcut for OpenMM/MD script directory
SDIR="../md_scripts/openmm/"

# Set path to initial structure
initialStructure="system_ini.cif"

## You should not need to edit anything below unless you want
## to change simulation parameters/protocols.

# Copy compiled '.so' file for distance calculations
cp ${SDIR}/lib/*.so .

# Set scripts for AMBER (much better pun with CHARMM)
SDIR=${SDIR}/amberff

if [ ! -f "$initialStructure" ]
then
  echo "Initial structure not found: $initialStructure"
  exit 1
fi

# (1) Build 'system': add Hs, process through force field.
if [ ! -f "system_H.cif" ]
then
  echo ">> Building system ..."
  python ${SDIR}/buildSystem.py $initialStructure \
    --seed $SEED \
    --output system_H.cif &> buildSystem.runlog
  [[ "$?" -ne 0 ]] && exit 1
else
  echo "'buildSystem' finished successfully before..."
fi

# (2) Add periodic box
if [ ! -f "system_PBC.cif" ]
then
  echo ">> Adding periodic boundary conditions ..."
  python ${SDIR}/setPeriodicBox.py system_H.cif \
    --seed $SEED \
    --padding 1.1 \
    --boxtype cubic \
    --output system_PBC.cif &> setPeriodicBox.runlog
  [[ "$?" -ne 0 ]] && exit 1
else
  echo "'setPeriodicBox' finished successfully before..."
fi

# (3) Do restrained energy minimization to optimize Hs and clear strains
if [ ! -f "system_EM.cif" ]
then
  echo ">> Minimizing system in vacuum ..."
  python ${SDIR}/minimizeSystem.py system_PBC.cif \
    --iterations 1000 \
    --posre \
    --seed $SEED \
    --output system_EM.cif &> system_EM.runlog
  [[ "$?" -ne 0 ]] && exit 1
else
  echo "'minimizeSystem' in vacuum finished successfully before..."
fi

# (4) Add solvent and counter ions to physiological concentration
if [ ! -f "system_ions.cif" ]
then
  echo ">> Solvating solute and neutralizing system ..."
  python ${SDIR}/solvateBox.py system_EM.cif \
    --neutralize \
    --seed $SEED \
    --output system_ions.cif &> solvateBox.runlog
  [[ "$?" -ne 0 ]] && exit 1
else
  echo "'solvateBox' finished successfully before..."
fi

# (5) Minimize the solvated system to optimize solvent and ion positions
if [ ! -f "system_ions_EM.cif" ]
then
  echo ">> Minimizing solvated system ..."
  python ${SDIR}/minimizeSystem.py system_ions.cif \
    --posre \
    --seed $SEED \
    --output system_ions_EM.cif &> system_ions_EM.runlog
  [[ "$?" -ne 0 ]] && exit 1
else
  echo "'minimizeSystem' in explicit solvent finished successfully before..."
fi

# (6) Heat the system to the desired temperature in gradual steps
if [ ! -f "system_heated.cif" ]
then
  echo ">> Heating system ..."
  python ${SDIR}/heatSystem.py system_ions_EM.cif \
    --temperature 310 \
    --seed $SEED \
    --output system_heated &> heatSystem.runlog
  [[ "$?" -ne 0 ]] && exit 1
else
  echo "'minimizeSystem' in explicit solvent finished successfully before..."
fi

#
# Equilibrate the system
#
# (7) Do first round of equilibration: NVT
if [ ! -f "Eq_NVT.cif" ]
then
  echo ">> Equilibration under NVT ..."
  python ${SDIR}/equilibrate_NVT.py system_ions_EM.cif \
    --temperature 310 \
    --seed $SEED \
    --state system_310K.cpt \
    --runtime 5 \
    --restraint-heavy-atom \
    --restraint-heavy-atom-k 500 \
    --output Eq_NVT &> Eq_NVT.runlog

  [[ "$?" -ne 0 ]] && exit 1
else
  echo "NVT finished successfully before..."
fi

# (8) Do second round of equilibration: NPT, strong restraints
if [ ! -f "Eq_NPT_k500.cif" ]
then
  echo ">> Equilibration under NPT (k=500) ..."
  python ${SDIR}/equilibrate_NPT.py Eq_NVT.cif \
    --temperature 310 \
    --barostat isotropic \
    --seed $SEED \
    --state Eq_NVT.cpt \
    --runtime 5 \
    --restraint-heavy-atom \
    --restraint-heavy-atom-k 500 \
    --output Eq_NPT_k500 &> Eq_NPT_k500.runlog
  [[ "$?" -ne 0 ]] && exit 1
else
  echo "NPT (k=500) finished successfully before..."
fi

# (9) Do third round of equilibration: NPT, weak restraints
if [ ! -f "Eq_NPT_k250.cif" ]
then
  echo ">> Equilibration under NPT (k=250) ..."
  python ${SDIR}/equilibrate_NPT.py Eq_NPT_k500.cif \
    --temperature 310 \
    --barostat isotropic \
    --seed $SEED \
    --state Eq_NPT_k500.cpt \
    --runtime 5 \
    --restraint-heavy-atom \
    --restraint-heavy-atom-k 250 \
    --output Eq_NPT_k250 &> Eq_NPT_k250.runlog
  [[ "$?" -ne 0 ]] && exit 1
else
  echo "NPT (k=250) finished successfully before..."
fi

# (10) Do fourth round of equilibration: NPT, very weak restraints
if [ ! -f "Eq_NPT_k50.cif" ]
then
  echo ">> Equilibration under NPT (k=50) ..."
  python ${SDIR}/equilibrate_NPT.py Eq_NPT_k250.cif \
    --temperature 310 \
    --barostat isotropic \
    --seed $SEED \
    --state Eq_NPT_k250.cpt \
    --runtime 5 \
    --restraint-heavy-atom \
    --restraint-heavy-atom-k 50 \
    --output Eq_NPT_k50 &> Eq_NPT_k50.runlog
  [[ "$?" -ne 0 ]] && exit 1
else
  echo "NPT (k=50) finished successfully before..."
fi

# (11) Last round of equilibration. No restraints.
if [ ! -f "Eq_NPT_noPR.cif" ]
then
  echo ">> Equilibration under NPT (k=0) ..."
  python ${SDIR}/equilibrate_NPT.py Eq_NPT_k50.cif \
    --temperature 310 \
    --barostat isotropic \
    --seed $SEED \
    --state Eq_NPT_k50.cpt \
    --runtime 5 \
    --output Eq_NPT_noPR &> Eq_NPT_noPR.runlog
  [[ "$?" -ne 0 ]] && exit 1
else
  echo "NPT (k=0) finished successfully before..."
fi


# Check equilibration finished successfully
if [ ! -f "Eq_NPT_noPR.xml" ]
then
  echo "Equilibration did not finish successfully"
  exit 1
fi

exit 1

# (12) Run production simulation
if [ ! -f "production.cif" ]
then
  echo ">> Running production simulation ..."
  python ${SDIR}/runProduction.py equilibrate_NPT_k0.cif \
    --state equilibrate_NPT_k0.xml \
    --write-frequency 10000 \
    --log-frequency 1000 \
    --temperature 310 \
    --isobaric \
    --runtime 100 \
    --seed $SEED \
    --output production
else
  echo "'runProduction' finished successfully before..."
  echo "Nothing to do here .."
fi
