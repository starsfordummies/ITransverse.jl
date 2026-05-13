# tMPO Construction

Functions for assembling the temporal MPO/MPS networks.

## Block types

```@docs
FwtMPOBlocks
FoldtMPOBlocks
```

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
