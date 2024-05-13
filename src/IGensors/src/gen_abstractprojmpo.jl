"""
 Same as _makeL_gen! but without the dag() in the prime(psi[ll+1])
"""
function _makeL_gen!(P::AbstractProjMPO, psi::gMPS, k::Int)::Union{ITensor,Nothing}
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

function makeL_gen!(P::AbstractProjMPO, psi::gMPS, k::Int)
  _makeL_gen!(P, psi, k)
  return P
end

function _makeR_gen!(P::AbstractProjMPO, psi::gMPS, k::Int)::Union{ITensor,Nothing}
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

function makeR_gen!(P::AbstractProjMPO, psi::gMPS, k::Int)
  _makeR_gen!(P, psi, k)
  return P
end

"""
Same as position!() but calling makeL/R_gen
"""
function position_gen!(P::AbstractProjMPO, psi::gMPS, pos::Int)
  makeL_gen!(P, psi, pos - 1)
  makeR_gen!(P, psi, pos + nsite(P))
  return P
end




##################################################################
# Try to build envs as old + eps*new 
###############################################


# Same as _makeL_gen! but without the dag() in the prime(psi[ll+1])
function _makeL_gen_wprev!(P::AbstractProjMPO, psi::MPS, k::Int, Ai::ITensor, epsilon::Float64)::Union{ITensor,Nothing}
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
    del = delta(uniqueinds(Ai,psi[ll+1]),uniqueinds(psi[ll+1], Ai))
    L = L * (Ai*del + epsilon*psi[ll + 1]) * P.H[ll + 1] * prime(Ai*del + epsilon*psi[ll + 1])
    P.LR[ll + 1] = L
    ll += 1
  end
  # Needed when moving lproj backward.
  P.lpos = k
  return L
end

function makeL_gen_wprev!(P::AbstractProjMPO, psi::MPS, k::Int, Ai::ITensor, epsilon::Float64)
  _makeL_gen_wprev!(P, psi, k, Ai, epsilon)
  return P
end

function _makeR_gen_wprev!(P::AbstractProjMPO, psi::MPS, k::Int, Ai::ITensor, epsilon::Float64)::Union{ITensor,Nothing}
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
    del = delta(uniqueinds(Ai,psi[rl-1]),uniqueinds(psi[rl-1], Ai))
    R = R * (Ai*del + epsilon*psi[rl - 1]) * P.H[rl - 1] * prime(Ai*del + epsilon*psi[rl - 1])
    P.LR[rl - 1] = R
    rl -= 1
  end
  P.rpos = k
  return R
end

function makeR_gen_wprev!(P::AbstractProjMPO, psi::MPS, k::Int, Ai::ITensor, epsilon::Float64)
  _makeR_gen_wprev!(P, psi, k, Ai, epsilon)
  return P
end

function position_gen_wprev!(P::AbstractProjMPO, psi::MPS, pos::Int, prev_Ai::ITensor, epsilon::Float64)
  makeL_gen_wprev!(P, psi, pos - 1, prev_Ai, epsilon)
  makeR_gen_wprev!(P, psi, pos + nsite(P), prev_Ai, epsilon)
  return P
end
