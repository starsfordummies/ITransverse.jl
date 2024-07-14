# ITransverse.jl

This package provide several routines for the transverse contraction of 2D tensor networks,
it was mainly developed for the characterization of temporal matrix product states 
in the time evolution of one-dimensional quantum many-body systems. 

It is built on top of the excellent ITensors library, and tries to reuse most of its features whenever possible.

In order to install it, from julia

```julia
julia> using Pkg

julia> Pkg.add(url="https://github.com/starsfordummies/ITransverse.jl.git")
```

# Literature 

The basic idea of the transverse contraction was proposed in [1], see also [2,3]. 
The light cone algorithms was proposed in [4]
For alternate methods of compressing the temporal MPS, see [5]

[1] M. C. Bañuls, M. B. Hastings, F. Verstraete, and J. I. Cirac, 
[Matrix Product States for  Dynamical Simulation of Infinite Chains](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.102.240603), Phys. Rev. Lett. 102, 240603 (2009)
[2] A. Müller-Hermes, J.I. Cirac and M.C. Bañuls, 
[Tensor network techniques for the computation of dynamical observables in one-dimensional quantum spin systems](https://doi.org/10.1088/1367-2630/14/7/075003),  New J. Phys. 14 075003 (2012)

[3] M.B. Hastings and R. Mahajan,
[Connecting Entanglement in Time and Space: Improving the Folding Algorithm](https://doi.org/10.1103/PhysRevA.91.032306)
 Phys.Rev.A 91 (2015) 3, 032306

[4] M. Frías-Pérez and M.C. Bañuls, [Light cone tensor network and time evolution](https://doi.org/10.1103/PhysRevB.106.115117),
Phys.Rev.B 106 (2022) 11, 115117

[5] S. Carignano, C.R. Marimón and L. Tagliacozzo, [Temporal entropy and the complexity of computing the expectation value of local operators after a quench](https://inspirehep.net/literature/2679401)
Phys.Rev.Res. 6 (2024) 3, 033021 


# Motivation: time evolution of 1D quantum systems

The 2D tensor network associated with the time evolution of a quantum chain is 

```

^          |  |  |  |  |  |  |  |
|          o--o--o--o--o--o--o--o
|          |  |  |  |  |  |  |  |
| t        o--o--o--o--o--o--o--o  U(dt)
|          |  |  |  |  |  |  |  |
|          o--o--o--o--o--o--o--o  U(dt)
|          |  |  |  |  |  |  |  |
|          o--o--o--o--o--o--o--o  |psi0>
|
|------------------------------------->   x

```

Suppose we are interested in calculating the expectation value of a local operator on a given site (or few sites), 
the network is  


```
   
|          o--o--o--o--o--o--o--o  <psi0
|          |  |  |  |  |  |  |  |
| t        o--o--o--o--o--o--o--o  Udag(dt)
|          |  |  |  |  |  |  |  |
|          o--o--o--o--o--o--o--o  Udag(dt)
|          |  |  |  |  |  |  |  |
|          o--o--o--o--o--o--o--o  
v          |  |  |  |  |  |  |  |
           |  |  |  x  x  |  |  |    <- local ops, ..111XX111..
           |  |  |  |  |  |  |  |
^          o--o--o--o--o--o--o--o
|          |  |  |  |  |  |  |  |
| t        o--o--o--o--o--o--o--o  U(dt)
|          |  |  |  |  |  |  |  |
|          o--o--o--o--o--o--o--o  U(dt)
|          |  |  |  |  |  |  |  |
|          o--o--o--o--o--o--o--o  |psi0>
|
|------------------------------------->   x

```

This can be __folded__ in a smaller network with doubled tensors, 

```
   

           ^  ^  ^  x  x  ^  ^  ^      <- local ops
           H  H  H  H  H  H  H  H
^          o==o==o==o==o==o==o==o
|          H  H  H  H  H  H  H  H
| t        o==o==o==o==o==o==o==o  U(dt)
|          H  H  H  H  H  H  H  H
|          o==o==o==o==o==o==o==o  U(dt)
|          H  H  H  H  H  H  H  H
|          o==o==o==o==o==o==o==o  |psi0>
|
|------------------------------------->   x

where `=` and `H` denote doubled horizontal and vertical legs, 
and `^` are just vectorized identities to close the network from the top. 
```

Rather than working with top and bottom, we can introduce left and right "temporal MPS" (tMPS), 
with respect to the central MPO columns (which can be seen as transfer matrices). The full contraction 
of the network is then given by 

```
                x  x 
           *====o==o====*  
           |    H  H    |
^          *====o==o====*
|          |    H  H    |
| t        *====o==o====*
|          |    H  H    |
|          *====o==o====*
|          |    H  H    |
|          *====o==o====*
|         <L|  O1  O2  |R> 
|---------------------------->   x
```

So here we develop transverse algorithms for updating the left and right dominant vectors of the transfer matrix, ie. a column
of the network depicted above. We include both a power method, as well as a light cone algorithm which exploits the causal structure
of the network when local operators are involved. 

These algorithms in any case are quite generic and can be used for other scenarios as well.

# Conventions

We start with ITensors conventions, so for an MPS we'd have something like (physical legs here point upwards, ie.
we think of applying an MPO to an MPS like a tetris brick falling from the top - in order words, if we think of time 
evolution operators, time would go upwards)


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

The application of an MPO to an MPS is simply their product (ITensors automatically contract equal indices) and then noprime the remaining physical index.


## Rotated / Transverse MPS-MPO

In order to make contact with the usual wiring in our brains that works with horizontal chains,
after defining our tMPS it may be useful to think of "rotating" the network by 90 degrees, redefining the 
old virtual indices of the MPO as new _temporal physical_ indices. In the same way, we can relabel the physical 
(spatial) indices as _temporal virtual_ indices.

More specifically, our current convention is the following:

We rotate our space vectors to the *right* by 90°, ie 

```
    |p'               |L                 |p'new
    |                 |                  |
L---o---R   ==>   p---o---p'   =  Lnew---o---Rnew
    |                 |                  |
    |p                |R                |pnew
```

so the index renaming convention is 

``` 
(old)   (new)
 L    ->  p'
 R    ->  p
 p    ->  L
 p'   ->  R
```


For building the tMPO, we contract with the operator `fold_op` on the *right*
and the initial state `init_state` on the *left*, ie. 

```
              p'
         |    |    |    |
[in] X==(W)==(W)==(W)==(W)==o [op]
         |    |    |    |
              p

------------------------------>time
```

With our rotation, the MPS with the physical legs is the "right" part of the network, |R>,
which we depict with the legs pointing upwards

```
^
|          |  |  |  |  |  |  |  |
|          o--o--o--o--o--o--o--o  TM
|          |  |  |  |  |  |  |  |
|          o--o--o--o--o--o--o--o  |R>
|
|------------------------------------------>
                       
```

## Why it is a non-trivial problem

In principle here we have more options to truncate than the usual update using SVDs/eigenvalues of RDMs: more specifically,
we can truncate on the reduced *transition* matrices (RTM) built from the left and right vectors, T~|R><L|. 
These objects are however *non-hermitian*, so the eigenvalue problems becomes much less straightforward (we are typically doing 
dynamics so all our quantities are complex). 
Fortunately, for many cases our TMs end up being symmetric left-right in space, (related to having translationally invariant Hamiltonians), which allows 
for the use of algorithms which have higher numerical stability. 


# High-level routines

We provide high-level functions for the following algorithms (more details below):
-power method
-symmetric power method 
-light cone 

## Parameters 
We have a few basic structs which define the models and truncations and are carried around in the programs. 
Most notably we have
- `model_params` for building the basic model Hamiltonians
- `tmpo_params` for building the temporal MPOs (including the initial states and closing operators)
- `TruncParams` which specify the cutoffs, max bond dimensions, as well as the truncation scheme when applicable
- `PMParams` where we store also additional parameters required for the power methods
- `ConeParams` where we store also additional parameters required for the light cone algorithms

The simplest way to get an idea is probably to look at the various main files in the mains/ folder 
and in the function documentations

## tMPO builders 

we can build both folded and unfolded tMPOs, the idea is to save the main building blocks (the tMPO tensors) 
in some structs like `FoldtMPOBlocks` and `FwtMPOBlocks` and build the tMPOS from them. Relevant functions are
`folded_tMPO` and `fw_tMPO` 

## Sweeps and truncation methods

Depending on the algorithm chosen, we can truncate either by performing the standard optimization based on reduced density 
matrices (`RDM`) associated with the left and right vectors individually, or optimize the overlap <L|R> or <L|Operators|R>, 
depending on the problem. For this, the algorithms are rather based on optimizing reduced *transition* matrices, 
so we label them as `RTMxx`

## Power method 

### Update including operator 

the function is `powermethod`, which is the relevant one for a setup like folded tMPO 
(we need to update including an operator or the dominant tMPS end up being trivial).

### Update without an operator 

TODO

### Symmetric power method
In a setup like Loschmidt echo, if our tMPOs are symmetric left-right we can optimize
the overlap <Rbar|TM TM|R>, where <Rbar| = |R>^T  (ie. we don't conjugate the ket in order to make the bra from it,
simply transpose). Then we can make use of different algorithms based on symmetric SVD or eigenvalue decompositions underneath.

## Light cone 

We initialize the cone using `init_cone(tp::tmpo_params)`, then evolve it using `run_cone(..)`

# Generalized and standard entropies

We provide tools for diagonalizing the RTMs and computing all sorts of generalized entropies from them. One can 
- diagonalize directly the RTM T_t (for very short tMPS only, of course)
- compute Tr(T_t^2) by contracting twice the left and right dominant vectors
- *If* we have a symmetric RTM (<L| = <Rbar|) we can put the RTM in a symmetric gauge where we have orthogonal matrices,
  so that we can compute the eigenvalues by diagonalizing site by site some environments of size ~chi^2

# Models

So far Ising with transverse + parallel fields has been thoroughly tested and should work, the relevant function
to build the exp(Hising) we use is `build_expH_ising_murg`

The Potts model is also defined and should work, but is probably currently broken with the latest million API changes 

The XXZ model is also implemented, but I haven't checked in a million years 

# WIP: GPU Extensions 

In principle it should work by simply doing a `using CUDA` and by putting the tmpo_params on GPU, `NDTensors.gpu(tp::tmpo_params)`. 
The idea is that the programs build everything down from there on GPU. 
There are likely bugs 
