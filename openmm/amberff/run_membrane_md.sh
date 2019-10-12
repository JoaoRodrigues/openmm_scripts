#!/usr/bin/env bash
#
# BASH Script to run a membrane simulation using
# amberff14sb (semi-isotropic pressure coupling).
# Default lipid is POPC.

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
if [ ! -f "solute_H.cif" ]
then
  echo ">> Building system in vacuum ..."
  python ${SDIR}/buildSystem.py $initialStructure \
    --seed $SEED \
    --output solute_H.cif &> buildSystem.runlog
  [[ "$?" -ne 0 ]] && exit 1
else
  echo "'buildSystem' finished successfully before..."
fi

# (2) Add periodic box
if [ ! -f "solute_PBC.cif" ]
then
  echo ">> Adding periodic boundary conditions ..."
  python ${SDIR}/setPeriodicBox.py solute_H.cif \
    --seed $SEED \
    --padding 1.1 \
    --boxtype cubic \
    --output solute_PBC.cif &> setPeriodicBox.runlog
  [[ "$?" -ne 0 ]] && exit 1
else
  echo "'setPeriodicBox' finished successfully before..."
fi

# (3) Do restrained energy minimization to optimize Hs and clear strains
if [ ! -f "solute_EM.cif" ]
then
  echo ">> Minimizing system in vacuum ..."
  python ${SDIR}/minimizeSystem.py solute_PBC.cif \
    --iterations 1000 \
    --posre \
    --seed $SEED \
    --output solute_EM.cif &> solute_EM.runlog
  [[ "$?" -ne 0 ]] && exit 1
else
  echo "'minimizeSystem' in vacuum finished successfully before..."
fi

# (4) Build membrane and solvate system (neutralize to .15M)
if [ ! -f "membrane.cif" ]
then
  echo ">> Adding membrane and neutralizing system ..."
  python ${SDIR}/addMembrane.py solute_EM.cif \
    --lipid POPC \
    --paddingXY 1.1 \
    --neutralize \
    --seed $SEED \
    --output membrane.cif &> addMembrane.runlog
  [[ "$?" -ne 0 ]] && exit 1
else
  echo "'addMembrane' finished successfully before..."
fi

# (5) Minimize the solvated system to optimize solvent and ion positions
if [ ! -f "membrane_EM.cif" ]
then
  echo ">> Minimizing membrane system ..."
  python ${SDIR}/minimizeSystem.py membrane.cif \
    --posre \
    --seed $SEED \
    --output membrane_EM.cif &> membrane_EM.runlog
  [[ "$?" -ne 0 ]] && exit 1
else
  echo "'minimizeSystem' for membrane system finished successfully before..."
fi

# (6) Heat the system to the desired temperature in gradual steps
if [ ! -f "system_heated.cif" ]
then
  echo ">> Heating system ..."
  python ${SDIR}/heatSystem.py membrane_EM.cif \
    --membrane \
    --temperature 310 \
    --seed $SEED \
    --output system_heated &> heatSystem.runlog
  [[ "$?" -ne 0 ]] && exit 1
else
  echo "'heatSystem' finished successfully before..."
fi

#
# Equilibrate the system
#
# (7) Do first round of equilibration: NVT
if [ ! -f "Eq_NVT.cif" ]
then
  echo ">> Equilibration under NVT ..."
  python ${SDIR}/equilibrate_NVT.py system_heated.cif \
    --temperature 310 \
    --seed $SEED \
    --state system_heated.cpt \
    --runtime 2.5 \
    --restraint-heavy-atom \
    --restraint-heavy-atom-k 500 \
    --restraint-lipids \
    --restraint-lipids-k 500 \
    --output Eq_NVT &> Eq_NVT.runlog

  [[ "$?" -ne 0 ]] && exit 1
else
  echo "NVT finished successfully before..."
fi

# (8) Do second round of equilibration: NPT
# Strong restraints on both lipid headgroups and protein HAs
if [ ! -f "Eq_NPT_k500_k500.cif" ]
then
  echo ">> Equilibration under NPT (k=500/500) ..."
  python ${SDIR}/equilibrate_NPT.py Eq_NVT.cif \
    --temperature 310 \
    --barostat membrane \
    --seed $SEED \
    --state Eq_NVT.cpt \
    --runtime 2.5 \
    --restraint-heavy-atom \
    --restraint-heavy-atom-k 500 \
    --restraint-lipids \
    --restraint-lipids-k 500 \
    --output Eq_NPT_k500_k500 &> Eq_NPT_k500_k500.runlog
  [[ "$?" -ne 0 ]] && exit 1
else
  echo "NPT (k=500/500) finished successfully before..."
fi

# (9a) Release lipid head group restraints
if [ ! -f "Eq_NPT_k500_k100.cif" ]
then
  echo ">> Equilibration under NPT (k=500/100) ..."
  python ${SDIR}/equilibrate_NPT.py Eq_NPT_k500_k500.cif \
    --temperature 310 \
    --barostat membrane \
    --seed $SEED \
    --state Eq_NPT_k500_k500.cpt \
    --runtime 2.5 \
    --restraint-heavy-atom \
    --restraint-heavy-atom-k 500 \
    --restraint-lipids \
    --restraint-lipids-k 100 \
    --output Eq_NPT_k500_k100 &> Eq_NPT_k500_k100.runlog
  [[ "$?" -ne 0 ]] && exit 1
else
  echo "NPT (k=500/100) finished successfully before..."
fi

# (9b) No Lipid Restraints
if [ ! -f "Eq_NPT_k500.cif" ]
then
  echo ">> Equilibration under NPT (k=500/0) ..."
  python ${SDIR}/equilibrate_NPT.py Eq_NPT_k500_k100.cif \
    --temperature 310 \
    --barostat membrane \
    --seed $SEED \
    --state Eq_NPT_k500_k100_noDUM.cpt \
    --runtime 2.5 \
    --restraint-heavy-atom \
    --restraint-heavy-atom-k 500 \
    --output Eq_NPT_k500 &> Eq_NPT_k500.runlog
  [[ "$?" -ne 0 ]] && exit 1
else
  echo "NPT (k=500/0) finished successfully before..."
fi

# (10a) Gradually release protein heavy atom restraints
if [ ! -f "Eq_NPT_k250.cif" ]
then
  echo ">> Equilibration under NPT (k=500/0) ..."
  python ${SDIR}/equilibrate_NPT.py Eq_NPT_k500.cif \
    --temperature 310 \
    --barostat membrane \
    --seed $SEED \
    --state Eq_NPT_k500.cpt \
    --runtime 2.5 \
    --restraint-heavy-atom \
    --restraint-heavy-atom-k 250 \
    --output Eq_NPT_k250 &> Eq_NPT_k250.runlog
  [[ "$?" -ne 0 ]] && exit 1
else
  echo "NPT (k=250) finished successfully before..."
fi

# (10b)
if [ ! -f "Eq_NPT_k50.cif" ]
then
  echo ">> Equilibration under NPT (k=500/0) ..."
  python ${SDIR}/equilibrate_NPT.py Eq_NPT_k250.cif \
    --temperature 310 \
    --barostat membrane \
    --seed $SEED \
    --state Eq_NPT_k250.cpt \
    --runtime 2.5 \
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
    --barostat membrane \
    --seed $SEED \
    --state Eq_NPT_k50_noDUM.xml \
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

#
# Production Simulation
#

# (12) Run production simulation
# 5 fs time step with HMR
# xyz every .1 ns, log every .1 ns
if [ ! -f "production.cif" ]
then
  echo ">> Running production simulation ..."
  python ${SDIR}/runProduction.py Eq_NPT_noPR_noDUM.cif \
    --hmr \
    --state Eq_NPT_noPR_noDUM.xml \
    --xyz-frequency 20000 \
    --log-frequency 20000 \
    --barostat membrane \
    --runtime 200 \
    --seed $SEED \
    --output production
else
  echo "'runProduction' finished successfully before..."
  echo "Nothing to do here .."
fi

