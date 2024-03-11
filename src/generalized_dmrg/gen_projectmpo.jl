import ITensors: AbstractProjMPO

function _makeL_gen!(P::AbstractProjMPO, psi::MPS, k::Int)::Union{ITensor,Nothing}
    # Save the last `L` that is made to help with caching
    # for DiskProjMPO
    ll = P.lpos
    if ll ≥ k
      # Special case when nothing has to be done.
      # Still need to change the position if lproj is
      # being moved backward.
      P.lpos = k
      return nothing
    end
    # Make sure ll is at least 0 for the generic logic below
    ll = max(ll, 0)
    L = lproj(P)
    while ll < k
      L = L * psi[ll + 1] * P.H[ll + 1] * prime(psi[ll + 1])
      P.LR[ll + 1] = L
      ll += 1
    end
    # Needed when moving lproj backward.
    P.lpos = k
    return L
  end
  
  function makeL_gen!(P::AbstractProjMPO, psi::MPS, k::Int)
    _makeL_gen!(P, psi, k)
    return P
  end
  
  function _makeR_gen!(P::AbstractProjMPO, psi::MPS, k::Int)::Union{ITensor,Nothing}
    # Save the last `R` that is made to help with caching
    # for DiskProjMPO
    rl = P.rpos
    if rl ≤ k
      # Special case when nothing has to be done.
      # Still need to change the position if rproj is
      # being moved backward.
      P.rpos = k
      return nothing
    end
    N = length(P.H)
    # Make sure rl is no bigger than `N + 1` for the generic logic below
    rl = min(rl, N + 1)
    R = rproj(P)
    while rl > k
      R = R * psi[rl - 1] * P.H[rl - 1] * prime(psi[rl - 1])
      P.LR[rl - 1] = R
      rl -= 1
    end
    P.rpos = k
    return R
  end
  
  function makeR_gen!(P::AbstractProjMPO, psi::MPS, k::Int)
    _makeR_gen!(P, psi, k)
    return P
  end
  
  """
      position_gen!(P::ProjMPO, psi::MPS, pos::Int)
  
  """
  function position_gen!(P::AbstractProjMPO, psi::MPS, pos::Int)
    makeL_gen!(P, psi, pos - 1)
    makeR_gen!(P, psi, pos + nsite(P))

    return P
  end
  


"""
Retrieve the number of unprojected (open)
site indices of the ProjMPO object `P`
"""
nsite(P::AbstractProjMPO) = P.nsite
