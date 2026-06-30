# GReg-InputGen
A Python utility for automatically preparing molecule–graphene input coordinates for geometry optimization and GRegOptimizer simulations.

The script aligns a molecular adsorbate onto a graphene sheet, constructs a trimmed hydrogen-saturated graphene fragment for efficient geometry optimization with xTB, and finally reconstructs the optimized adsorbate on the original graphene sheet for subsequent GRegOptimizer calculations.

# Installation
## Requirements:
Python 3.9+ \
NumPy \
RDKit

## Optional:
xTB \
OVITO

## Install Python dependencies:
Using Conda (recommended): \
conda install -c conda-forge numpy rdkit

or with pip: \
pip install numpy

RDKit is strongly recommended to be installed through Conda.

## Install xTB:
Download xTB from: \
https://github.com/grimme-lab/xtb

Ensure that the executable is available in your PATH.

Verify with: \
xtb --version


## Install OVITO (optional):
By default the script searches for: \
C:\Program Files\OVITO_Basic\ovito.exe

If installed elsewhere, edit function: \
open_in_ovito()

# Input
## The script requires two XYZ files:
graphene.xyz \
molecule.xyz

# Basic Usage
## Run the complete workflow:
python input_gen.py graphene.xyz molecule.xyz

## Run with visualization:
python input_gen.py graphene.xyz molecule.xyz --view

## Only prepare xTB input:
python input_gen.py graphene.xyz molecule.xyz --skip-xtb

## Reuse an existing xTB optimization:
python input_gen.py graphene.xyz molecule.xyz \
    --skip-xtb \
    --optimized-xyz xtb_optimization/xtbopt.xyz


## Command Line Options:
-o, --output |	Output aligned structure \
-v, --view |	Open generated coordinates in OVITO \
-d, --distance |	Molecule–graphene distance (Å) \
--buffer |	Graphene trimming buffer (Å) \
--xtb |	Path to xTB executable \
--method |	xTB method (gfnff, gfn1, gfn2) \
--solvent |	Optional ALPB solvent \
--bond-cutoff |	Carbon–carbon cutoff for neighbour analysis (Å) \
--edge-ch |	Edge C–H bond length (Å) \
--xtb-input |	Filename of xTB-ready XYZ \
--xtb-workdir |	xTB working directory \
--skip-xtb |	Skip geometry optimization \
--optimized-xyz |	Existing optimized xTB structure

# Output file description
graphene_with_molecule.xyz	Initial aligned coordinates \
xtb_ready.xyz	Trimmed hydrogen-saturated xTB input \
xtb_optimization/xtb_optimization.log	xTB log \
xtb_optimization/xtbopt.xyz	Optimized xTB coordinates \
optimized_on_full_graphene.xyz	Final coordinates

# Example commandline usage
If you want to use specific methods/parameters: \
python input_gen.py graphene.xyz molecule.xyz \
    --distance 3.5 \
    --method gfn2 \
    --solvent toluene \
    --view
    
# Notes
Only the molecular coordinates are optimized during xTB calculations. \
Graphene coordinates are fixed. \
OVITO visualization is optional and enabled using '-v'.

# License
This project is released under the MIT License.
