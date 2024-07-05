# ITransverse.jl

This package provide several routines for the transverse contraction of 2D tensor networks,
it was mainly developed for the characterization of temporal matrix product states 
in the time evolution of one-dimensional quantum many-body systems. 

It is built on top of the excellent ITensors library, and tries to reuse most of its features whenever possible.




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

The 2D network associated with the time evolution of a quantum chain is 

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
                    x  x            <- local ops
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
|         <L|   O1 O2  |R> 
|---------------------------->   x
```

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


# Main high-level routines

We provide high-level functions for the following algorithms

## Power method 

### Update including operator 

the function is `powermethod`

### Symmetric power method

## Light cone 

# Underlying subroutines

## Sweeps 

## Build tMPO 

## Symmetric eigenvalue/SVD decompositions

# Models


# WIP: GPU Extensions 