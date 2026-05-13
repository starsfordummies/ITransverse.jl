# ITensor Utilities

Low-level helpers for working with ITensors: custom SVD routines, MPS utilities,
matrix decompositions, and size estimation.

## MPS utilities

```@docs
pMPS
overlap_noconj
normbyfactor
normalize_for_overlap!
match_siteinds
match_siteinds!
phys_ind
gaugefix_left
logfidelity
```

The following utilities are exported but currently lack docstrings:
`allsiteinds`, `replace_linkinds!`, `halfsite`, `fidelity`, `gen_fidelity`.

## SVD / eigenvalue routines

```@docs
matrix_svd
truncated_svd
symm_svd
symm_oeig
```

## ITensor helpers

```@docs
randsymITensor
isid
pinvten
randITensor_decayspec
```

## Size estimation

```@docs
beta_lims
```
