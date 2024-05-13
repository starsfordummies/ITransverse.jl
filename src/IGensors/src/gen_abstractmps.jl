"""
    MPS

A finite size matrix product state type.
Keeps track of the orthogonality center.
"""
mutable struct gMPS <: AbstractMPS
  data::Vector{ITensor}
  llim::Int
  rlim::Int
  gllim::Int
  grlim::Int
  gMPS(data, llim, rlim) = new(data, llim, rlim, 0, 999)
  gMPS(inMPS::MPS) = new(inMPS.data, inMPS.llim, inMPS.rlim, 0, 999)
end



"""
 Gen left-rightlims 
"""

function norm2_gen(a::ITensor)
  return scalar(a * a)
end

function norm_gen(a::ITensor)
  return sqrt(norm2_gen(a))
end

""" Computes the generalized norm of an MPS (phi|phi) = <phi^*|phi>
"""
function norm2_gen(ll::MPS)
    norm2 = inner(dag(ll),ll) 
    return norm2
end
function norm_gen(ll::MPS)
    return sqrt(norm2_gen(ll))
end


""" Hacky - doesn't keep track of ortho """
function normalize_gen(M::AbstractMPS; (lognorm!)=[])
  return normalize_gen!(deepcopy(M))
end

""" Hacky - doesn't keep track of ortho """
function normalize_gen!(M::AbstractMPS)
  nrm = norm_gen(M)
  z = nrm/length(M)
  for n in eachindex(M)
    M[n] ./= z
  end
  return M
end



leftglim(m::gMPS) = m.gllim

rightglim(m::gMPS) = m.grlim

function setleftglim!(m::gMPS, new_ll::Integer)
  return m.gllim = new_ll
end

function setrightglim!(m::gMPS, new_rl::Integer)
  return m.grlim = new_rl
end

"""
More helper functions for ortho
"""
function set_gen_ortho_lims!(ψ::gMPS, r::UnitRange{Int})
  setleftglim!(ψ, first(r) - 1)
  setrightglim!(ψ, last(r) + 1)
  return ψ
end

function set_gen_ortho_lims(ψ::gMPS, r::UnitRange{Int})
  return set_gen_ortho_lims!(copy(ψ), r)
end

reset_gen_ortho_lims!(ψ::gMPS) = set_gen_ortho_lims!(ψ, 1:length(ψ))

isgenortho(m::gMPS) = leftglim(m) + 1 == rightglim(m) - 1

# Could also define as `only(ortho_lims)`
function genorthocenter(m::gMPS)
  !isgenortho(m) && error(
    "$(typeof(m)) has no well-defined orthogonality center, orthogonality center is on the range $(gen_ortho_lims(m)).",
  )
  return leftglim(m) + 1
end


"""
  Generalized version of `orthogonalize[!]`
  If we call `ortho_gen(psi,j)`, we bring it to the mixed `generalized` canonical form 
  ▷-▷-▷-▷-▷---o---◁--◁--◁ 
  1 2 .. j-1  j  j+1 .. 

  so that (no conjugation)

  ▷--          --
  |       =   |
  ▷--          --
  
  Also computes the (generalized) entropies in the process
"""
function orthogonalize_gen_ents!(M::gMPS, j::Int; normalize, method, verbose=false, cutoff=1e-14)

  maxdim = maxlinkdim(M)

  if verbose
    @info("Gen-orthogonalizing around $j, current gen. l/r lims : $(leftglim(M)), $(rightglim(M))")
  end

  sites = []
  ents = []

  @debug_check begin
    if !(1 <= j <= length(M))
      error("Input j=$j to `orthogonalize_gen!` out of range (valid range = 1:$(length(M)))")
    end
  end
  while leftglim(M) < (j - 1)
    (leftglim(M) < 0) && setleftglim!(M, 0)
    b = leftglim(M) + 1
    linds = uniqueinds(M[b], M[b + 1])
    lb = linkind(M, b)
    if !isnothing(lb)
      ltags = tags(lb)
    else
      ltags = TagSet("Link,l=$b")
    end
    #println("factorizing(L>) for b=$b")
    L, R, spec, _ = factorize_gen(M[b], linds; tags=ltags, maxdim, which_decomp=method, cutoff)
    #@show b, inds(L), inds(R)
    M[b] = L
    M[b + 1] *= R
    normalize && (M[b+1] /= norm_gen(M[b+1]))
    # if abs(norm_gen(M)) > 2
    #   @warn "!gen norm is large! $(norm_gen(M))"
    # end

    setleftglim!(M, b)
    if rightglim(M) < leftglim(M) + 2
      setrightglim!(M, leftglim(M) + 2)
    end

    #@info "[L] $b"
    if b ∉ sites
      push!(sites, b)
      push!(ents, sum(log.(spec.eigs)/sum(spec.eigs)))
    end
  end

  N = length(M)

  while rightglim(M) > (j + 1)
    (rightglim(M) > (N + 1)) && setrightglim!(M, N + 1)
    b = rightglim(M) - 2
    rinds = uniqueinds(M[b + 1], M[b])
    lb = linkind(M, b)
    if !isnothing(lb)
      ltags = tags(lb)
    else
      ltags = TagSet("Link,l=$b")
    end
    #println("factorizing(<R) for b=$(b+1)")
    L, R, spec, _ = factorize_gen(M[b + 1], rinds; tags=ltags, maxdim, which_decomp=method, cutoff)
    M[b + 1] = L
    M[b] *= R
    normalize && (M[b] /= norm_gen(M[b]))
    # if abs(norm_gen(M)) > 2
    #   @warn "!gen norm is large! $(norm_gen(M))"
    # end

    #@info "[R] $b"
    if b ∉ sites
      push!(sites, b)
      push!(ents, sum(log.(spec.eigs)/sum(spec.eigs)))
    end

    setrightglim!(M, b + 1)
    if leftglim(M) > rightglim(M) - 2
      setleftglim!(M, rightglim(M) - 2)
    end

    if order(M[b]) > 3
      @warn "more than 3 legs on M[$b]"
      @show inds(M[b])
    end

  end
  #println("final gen l/r lims :  $(leftglim(M)), $(rightglim(M))")

  # TODO maybe better get rid of primes in another spot ?
  noprime!(M)

  #@show sites
  return M, sites, ents
end

""" Just orthogonalize_gen! and throw away the entropies"""
function orthogonalize_gen!(M::gMPS, j::Int; normalize, method, verbose=false, cutoff=1e-14)
  
  M, _, _ = orthogonalize_gen_ents!(M,j ; normalize,method,verbose,cutoff)

  return M
end

# Allows overloading `orthogonalize!` based on the projected
# MPO type. By default just calls `orthogonalize!` on the MPS.
function orthogonalize_gen!(PH, M::gMPS, j::Int; kwargs...)
  return orthogonalize_gen!(M, j; kwargs...)
end

function orthogonalize_gen(ψ0::gMPS, args...; kwargs...)
  ψ = copy(ψ0)
  orthogonalize_gen!(ψ, args...; kwargs...)
  return ψ
end


""" Check that an MPS is in (generalized-symmetric) mixed canonical form at site b """
function check_gen_ortho(ψ::gMPS, verbose::Bool=false)

  isortho = true
  where_not_ortho = []
  isnormalized = true

  mpslen = length(ψ)
  bleft = leftglim(ψ)
  bright = rightglim(ψ)

  if verbose
      println("Checking MIXED gen/sym form, ortho center between $bleft and $bright")
  end


  right_env = ITensor(1.)
  for ii = mpslen:-1:bright
      Ai = ψ[ii]
      right_env =  right_env * Ai
      right_env = right_env * prime(Ai, commoninds(Ai,linkinds(ψ)))  
      @assert order(right_env) == 2
      if norm(array(right_env)- Diagonal(array(right_env))) > 0.1
          #println("[<R]non-diag@[$ii]")
          push!(where_not_ortho, ii)
          isortho = false
      end
      delta_norm = norm(array(right_env) - I(size(right_env)[1]))
      if delta_norm > 0.1
          #println("[<R]quite non-can@[$ii], $delta_norm")
          push!(where_not_ortho, [ii, delta_norm])
          isortho = false
      end
  end

  left_env = ITensor(1.)
  for ii =1:bleft
      Ai = ψ[ii]
      left_env =  left_env * Ai
      left_env = left_env * prime(Ai, commoninds(Ai,linkinds(ψ)))   #* delta(wLa, wLb) )
      @assert order(left_env) == 2
      delta_norm = norm(array(left_env) - I(size(left_env)[1])) /norm(array(left_env))

      if norm(array(left_env)- Diagonal(array(left_env))) > 0.1
          #println("[L>]non-diag@[$ii]")
          push!(where_not_ortho, ii)
          isortho = false
      end

      if delta_norm > 0.1
          #println("[L>]quite non-can@[$ii], $delta_norm")
          push!(where_not_ortho, [ii, delta_norm])
          isortho = false
      end
  end
  if verbose
      println("Done checking LEFT gen/sym form")
  end

  if verbose
      println("Done checking RIGHT gen/sym form")
  end

  if abs(norm_gen(ψ) - 1) > 1e-5
      println("overlap[mix] = $(norm_gen(ψ))")
      isnormalized = false
  end
  
  return isortho, isnormalized, where_not_ortho, ψ.gllim, ψ.grlim
end
