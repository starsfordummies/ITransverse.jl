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
