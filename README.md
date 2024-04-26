# ITransverse.jl
transverse contraction for temporal MPS 

# Conventions

We start with ITensors conventions, so for an MPS we'd have something like 


```
      p
      |
 L----o----R
```

and for an MPO 

```
      p'
      |
 L----o----R
      |
      p
```

So that the application of an MPO to an MPS is simply their product and then noprime the remaining physical index.

## Rotated / Transverse MPS-MPO

We rotate our space vectors to the left by 90°, ie 

```
    |p'              |R               |p'new
    |                |                |
L---o---R   =>   p'--o--p   =  Lnew---o---Rnew
    |                |                |
    |p               |L               |pnew
```

so the convention is L -> p, R -> p', p -> R, p' -> L

For building the tMPO, we contract with the operator `fold_op` on the *left*
and the initial state `init_state` on the *right*, ie. 

```
          p'
      |   |   |   |
[op]X=(W)=(W)=(W)=(W)=o [in]
      |   |   |   |
          p
```
