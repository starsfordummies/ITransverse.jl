
"""
Result of a truncated left-right contraction.

Fields:
- `L`: left output MPS
- `R`: right output MPS
- `sv`: singular value matrix (rows = bonds, cols = singular values normalised to sum 1)
- `norm_factor`: can be used to rescale `L` and `R` so that `overlap_noconj(L, R)` matches the
  pre-truncation overlap.

Supports destructuring as `(L, R, sv)` for backward compatibility:

    ll, rr, sv = tlrapply(...)         # existing callers unchanged
    result     = tlrapply(...)
    result.norm_factor                  # new field, nothing if not requested
"""
struct TruncLR{TSV,TNorm}
    L::MPS
    R::MPS
    sv::Matrix{TSV}
    norm_factor::TNorm
end

TruncLR(L, R, sv) = TruncLR(L, R, sv, 1.0)

Base.iterate(t::TruncLR, state=1) =
    state == 1 ? (t.L,  2) :
    state == 2 ? (t.R,  3) :
    state == 3 ? (t.sv, 4) :
    nothing

#Base.length(::TruncLR) = 3
ITensorMPS.maxlinkdim(t::TruncLR) = max(maxlinkdim(t.L), maxlinkdim(t.R))

function Base.getindex(t::TruncLR, i::Int)
    i == 1 && return t.L
    i == 2 && return t.R
    i == 3 && return t.sv
    i == 4 && return t.norm_factor
    throw(BoundsError(t, i))
end

overlap_noconj(lr::TruncLR, reverse_qn_ll::Bool=false; kwargs...) = overlap_noconj(lr.L, lr.R, reverse_qn_ll; kwargs...) 


function tlrcontract(::Algorithm"naiveRTM", psiL::MPS, OL::MPO, OR::MPO, psiR::MPS;
        preserve_mps_tags::Bool=false, # TODO not implemented yet
        kwargs...)
    OpsiR = applyn(OR, psiR; truncate=false)
    psiLO = applyns(OL, psiL; truncate=false)
    truncate_sweep(psiLO, OpsiR; kwargs...)
end

# Generic fallback
function tlrcontract(alg::Algorithm, psiL::MPS, OL::MPO, OR::MPO, psiR::MPS; kwargs...)
    OpsiR, sv = tapply(alg, OR, psiR; kwargs...)
    psiLO, _  = tapplys(alg, OL, psiL; kwargs...)
    return TruncLR(psiLO, OpsiR, sv, 1.0)
end

function tlrcontract(::Algorithm"notrunc", psiL::MPS, OL::MPO, OR::MPO, psiR::MPS; kwargs...)
    OpsiR = applyn(OR, psiR; truncate=false)
    psiLO = applyns(OL, psiL; truncate=false)
    sv = zeros(1,1)
    return TruncLR(psiLO, OpsiR, sv)
end

# (Cheating but maybe useful if) symmetric case: does not touch psiL, OL (assumed to be same as psiR, OR)
function tlrcontract(::Algorithm"RTMsym", psiL::MPS, OL::MPO, OR::MPO, psiR::MPS; kwargs...)
    OpsiR, sv = tcontract(Algorithm(:RTMsym), OR, psiR; kwargs...)
    return TruncLR(OpsiR, OpsiR, sv)
end


#### NEW LR apply
function tlrapply(alg, psiL::MPS, OL::MPO, OR::MPO, psiR::MPS; kwargs...)
    res = tlrcontract(alg, psiL, OL, OR, psiR; kwargs...)
    return TruncLR(noprime(res.L), noprime(res.R), res.sv, res.norm_factor)
end



### Apply only one, then truncate



function trcontract(::Algorithm"naiveRTM", ψL::MPS, AR::MPO, ψR::MPS;
        preserve_mps_tags::Bool=false, # TODO not implemented yet
        kwargs...)

    OpsiR = applyn(AR, ψR; truncate=false)
    truncate_sweep(ψL, OpsiR; kwargs...)
end

function trcontract(::Algorithm"notrunc", psiL::MPS, OR::MPO, psiR::MPS; kwargs...)
    OpsiR = applyn(OR, psiR; truncate=false)
    sv = zeros(1,1)
    return TruncLR(psiL, OpsiR, sv, 1.0)
end

# Generic fallback
function trcontract(alg::Algorithm, psiL::MPS, AR::MPO, psiR::MPS; kwargs...)
    OpsiR, sv = tapply(alg, AR, psiR; kwargs...)
    return TruncLR(psiL, OpsiR, sv)
end

"""
Applies MPO to left-right and truncates on the RTM |AR R><L AL| \\
Allows for different length MPS/MPO \\
Returns `TruncLR` (destructures as `(LEFT, RIGHT, SV)`)
"""
function tlcontract(alg, ψL::MPS, AL::MPO, ψR::MPS; kwargs...)
    res = trcontract(alg, ψR, swapprime(AL, 0=>1, "Site"), ψL; kwargs...)
    # trcontract treats its first arg as ψL and third as ψR; swap L↔R in result
    return TruncLR(res.R, res.L, res.sv, res.norm_factor)
end

"""
Applies MPO to the left and truncates on the RTM |R><L*AL| by building it explicitly \\
Returns `TruncLR` (destructures as `(LEFT, RIGHT, SV)`)
"""
function tlapply(alg, ψL::MPS, A::MPO, ψR::MPS; kwargs...)
    res = tlcontract(alg, ψL, A, ψR; kwargs...)
    return TruncLR(noprime(res.L), noprime(res.R), res.sv, res.norm_factor)
end

"""
Applies MPO to the right and truncates on the RTM |AR*R><L| by building it explicitly \\
Returns `TruncLR` (destructures as `(LEFT, RIGHT, SV)`)
"""
function trapply(alg, ψL::MPS, A::MPO, ψR::MPS; kwargs...)
    res = trcontract(alg, ψL, A, ψR; kwargs...)
    return TruncLR(noprime(res.L), noprime(res.R), res.sv, res.norm_factor)
end


# Generic wrappers 
tlapply(ψL::MPS, A::MPO, ψR::MPS; alg=Algorithm(:naiveRTM), kwargs...) = tlapply(Algorithm(alg), ψL, A, ψR; kwargs...)
trapply(ψL::MPS, A::MPO, ψR::MPS; alg=Algorithm(:naiveRTM), kwargs...) = trapply(Algorithm(alg), ψL, A, ψR; kwargs...)


"""
Applies AL to ψL from the left, AR to ψR from the right, and truncates on the RTM
|AR*R><L*AL| by building it explicitly. \\
Convention: when direction=:left, we PTR over left environments and, going right→left,
we SVD τ_R = tr_L(τ). \\
Returns `TruncLR` (destructures as `(LEFT, RIGHT, SV)`)
"""
tlrapply(ψL::MPS, AL::MPO, AR::MPO, ψR::MPS; alg=Algorithm(:naiveRTM), kwargs...) = tlrapply(Algorithm(alg), ψL, AL, AR, ψR; kwargs...)