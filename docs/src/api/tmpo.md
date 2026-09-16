# tMPO Construction

Functions for assembling the temporal MPO/MPS networks.

## Block types

```@docs
FwtMPOBlocks
FoldtMPOBlocks
```

## Boundary states

The initial (`bl`) and final (`tr`/`fold_op`) states closing the temporal chain can be
product states (vectors) or **non-product** ones, ie. columns of a boundary MPS with
bond dimension > 1:

```
 ⟨ϕ_f|      B———B———B———B———B      <- top    boundary (`tr` / `fold_op`)
             |   |   |   |   |
 U           W———W———W———W———W
             |   |   |   |   |
 |ψ_0⟩       A———A———A———A———A      <- bottom boundary (`bl`)

            ⟨L|  T   T   T  |R⟩
```

* **Product states** (rank-1) are contracted into the first/last tensor of the chain,
  which keeps `Nt` sites.
* **Non-product states** carry the bonds of the boundary MPS. Those bonds run along the
  *spatial* direction, so they become **site** indices of the temporal chain: the
  boundary tensor is appended as an *extra site*. A tMPO with a non-product `bl` has
  `Nt+1` sites, with non-product `bl` *and* `tr` it has `Nt+2`. Bulk columns (tMPO) are
  rank-3 `(phys, left, right)`, edge columns (tMPS) rank-2 `(phys, bond)`.

Index convention:

1. The leg contracted into the temporal chain (physical/space leg of the boundary MPS) is
   the only one tagged `"Site"`.
2. The spatial bonds are a prime pair `(s, s')` of a single index `s`: `s` (unprimed)
   points **right** - it is the leg the column is contracted from, cf. `applyn(T, ψR)` -
   and `s'` (primed) points **left**. All columns share the same boundary bond index.
3. Rank-2 (edge) tensors keep a single *unprimed* leg `s`, whichever side it is on.

```julia
bl  = boundary_tensor(A; phys=σ, left=l, right=r)   # rank-3, bulk column
blL = close_boundary(bl, vL; side=:left)            # rank-2, leftmost column
blR = close_boundary(bl, vR; side=:right)           # rank-2, rightmost column

b = FwtMPOBlocks(tMPOParams(mp; dt, init_state=bl))

T = fw_tMPO(b, ts; tr=up_state)                     # Nt+1 sites
L = fw_tMPS(b, ts; LR=:left,  bl=blL, tr=up_state)
R = fw_tMPS(b, ts; LR=:right, bl=blR, tr=up_state)
```

### Conjugation

`bl` is the ket |bl⟩; the top boundary of the unfolded builders (`fw_*`, `fwback_*`) is the
bra ⟨tr|, so it is **conjugated** by default (`dagger_tr=true`) and the network is the
amplitude ⟨tr|U…U|bl⟩. Pass `dagger_tr=false` to use `tr` as given - but use the *same* value
for the tMPO and for both edge tMPS, otherwise the columns describe different networks. It
only makes a difference for complex boundary states.

For the folded builders, `FoldtMPOBlocks` folds the bulk initial state into `b.rho0`;
the matching edge tensors are obtained by closing it with *folded* edge vectors,
`close_boundary(b.rho0, fold_boundary(vL); side=:left)`.

```@docs
boundary_tensor
close_boundary
fold_boundary
to_boundary
check_boundary
attach_boundary_bottom!
attach_boundary_top!
tMPO_in
```

`boundary_phys_ind`, `boundary_bond_ind`, `boundary_linkdim`, `is_product_boundary` and
`n_boundary_sites` query a boundary state.

## Forward tMPO

`fw_tMPO`, `fw_tMPS` – build a forward temporal MPO / MPS up to time `T`.

## Forward-backward tMPO

```@docs
fwback_tMPO
```

`fwback_tMPS` \u2013 convenience wrapper that returns the MPS form directly.

## Folded tMPO

```@docs
folded_tMPO
folded_tMPO_in
folded_tMPO_ext
```

`folded_tMPS`, `folded_left_tMPS`, `folded_right_tMPS` also construct folded temporal MPS variants.

## Generic constructor

```@docs
construct_tMPS_tMPO
```
