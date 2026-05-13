# ITransverse.jl

**ITransverse** is a Julia package for studying quantum dynamics and thermodynamics of 1D quantum chains using matrix-product-state (MPS) and matrix-product-operator (MPO) methods.

The core workflow is based on the **transverse (temporal) MPO** formulation: instead of time-evolving a state forward step by step, the package assembles the full time-evolution network (the *temporal MPO*, or tMPO) into a single MPS/MPO object that can then be compressed and analysed with standard tensor-network tools.

## Key features

- **tMPO assembly** – builds forward, folded, and folded-with-initial-state temporal MPOs for Ising, Potts, XXZ, and Floquet chains.
- **Light-cone growth** – iteratively extends a temporal MPS in the time direction using a truncated MPO-MPS contraction, controlled by a [`ConeParams`](@ref) object.
- **Power method** – finds the dominant eigenstate of the transfer matrix via repeated MPO-MPS application; both symmetric and asymmetric variants are available.
- **Truncation sweeps** – several algorithms for compressing an MPS that results from an MPO-MPS product (naive RTM, density-matrix, zip-up, symmetric RTM).
- **Entropy & spectral diagnostics** – von Neumann, Rényi, generalised SVD, and density-matrix–based entropies; mutual-information utilities.
- **TEBD** – basic time-evolving block decimation for real- and imaginary-time evolution.

## Getting started

```julia
using ITransverse

# Build Ising model parameters
mp = IsingParams(1.0, 0.4, 0.0)   # Jtwo, gperp, hpar

# Build tMPOParams (scheme defaults to Murg())
tp = tMPOParams(mp; dt=0.1, nbeta=0)

# Construct the forward tMPO blocks
b = FwtMPOBlocks(tp)

# Build the full forward tMPS up to time T = 20
psi = fw_tMPS(b, 20)
```

## Contents

```@contents
Pages = ["algorithms.md", "api/chain_models.md", "api/tmpo.md",
         "api/truncation.md", "api/power_method.md", "api/lightcone.md",
         "api/entropies.md", "api/contractions.md", "api/tebd.md",
         "api/itutils.md"]
Depth = 2
```
