# Truncation & Sweeps

Algorithms for compressing an MPS resulting from an MPO-MPS product.

## Core contraction / application

```@docs
tcontract
tapplys
applyn
```

See [`ttruncate!`](@ref) (documented on the [Algorithms](../algorithms.md) page) for the underlying SVD sweep.

## RTM-based sweeps

```@docs
truncate_sweep
truncate_sweep_sym
```

`truncate_sweep_rtm`, `truncate_lsweep_sym`, `truncate_rsweep_sym` are additional sweep variants (currently undocumented).

## RTM bond kernels

The per-bond decomposition the RTM sweeps (`alg = "RTM"` / `"RTMeig"`) hand their local
reduced transition matrix to. `svd_ERL` keeps the dominant singular subspaces and inserts two
isometries; `eig_rtm` keeps the dominant eigenvalues and inserts the RTM's oblique spectral
projector, which makes the truncated overlap exactly the sum of the kept eigenvalues at the
cost of an accuracy floor set by `cond(X)`.

```@docs
svd_ERL
eig_rtm
```

## Left-right truncation

```@docs
tlrapply
trapply
tlapply
TruncLR
```

## Canonical form

```@docs
gen_canonical
```
