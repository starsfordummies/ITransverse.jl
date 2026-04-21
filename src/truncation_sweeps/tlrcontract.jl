
"""
Result of a truncated left-right contraction.
"""
struct TruncLR{TSV}
    L::MPS
    R::MPS
    sv::Matrix{TSV}
    ov_before::ComplexF64 # overlap before truncation
    ov_after::ComplexF64   # overlap after truncation
end

TruncLR(L, R, sv, ovb::Number=1.0, ova::Number=1.0) = TruncLR(L, R, sv, ComplexF64(ovb), ComplexF64(ova))

""" Ratio ov_before / ov_after. Both fields must be non-nothing. """
norm_factor(t::TruncLR) = t.ov_before / t.ov_after

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
    i == 4 && return t.ov_before
    i == 5 && return t.ov_after
    throw(BoundsError(t, i))
end

overlap_noconj(lr::TruncLR, reverse_qn_ll::Bool=false; kwargs...) = overlap_noconj(lr.L, lr.R, reverse_qn_ll; kwargs...) 


function tlrcontract(::Algorithm"naiveRTM", psiL::MPS, OL::MPO, OR::MPO, psiR::MPS;
        preserve_mps_tags::Bool=false, # TODO not implemented yet
        compute_ov_before::Bool=false,
        kwargs...)
    
    psiLO = applyns(OL, psiL; truncate=false)
    OpsiR = applyn(OR, psiR;  truncate=false)

    ov_before = compute_ov_before ? overlap_noconj(psiLO, OpsiR) : 1.0

    res = truncate_sweep(psiLO, OpsiR; kwargs...)
    return TruncLR(res.L, res.R, res.sv, ov_before, res.ov_after)
end

# Generic fallback (no overlap tracking)
function tlrcontract(alg::Algorithm, psiL::MPS, OL::MPO, OR::MPO, psiR::MPS; kwargs...)
    OpsiR, sv = tapply(alg, OR, psiR; kwargs...)
    psiLO, _  = tapplys(alg, OL, psiL; kwargs...)
    return TruncLR(psiLO, OpsiR, sv)
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
    return TruncLR(noprime(res.L), noprime(res.R), res.sv, res.ov_before, res.ov_after)
end



### Apply only one, then truncate



function trcontract(::Algorithm"naiveRTM", ψL::MPS, AR::MPO, ψR::MPS;
        preserve_mps_tags::Bool=false, 
        compute_ov_before::Bool=false, # TODO not implemented yet
        kwargs...)

    OpsiR = applyn(AR, ψR; truncate=false)

    ov_before = compute_ov_before ? overlap_noconj(ψL, OpsiR) : 1.0
    res = truncate_sweep(ψL, OpsiR; kwargs...)
    return TruncLR(res.L, res.R, res.sv, ov_before, res.ov_after)
end

function trcontract(::Algorithm"notrunc", psiL::MPS, OR::MPO, psiR::MPS; kwargs...)
    OpsiR = applyn(OR, psiR; truncate=false)
    sv = zeros(1,1)
    return TruncLR(psiL, OpsiR, sv)
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
    return TruncLR(res.R, res.L, res.sv, res.ov_before, res.ov_after)
end

"""
Applies MPO to the left and truncates on the RTM |R><L*AL| by building it explicitly \\
Returns `TruncLR` (destructures as `(LEFT, RIGHT, SV)`)
"""
function tlapply(alg, ψL::MPS, A::MPO, ψR::MPS; kwargs...)
    res = tlcontract(alg, ψL, A, ψR; kwargs...)
    return TruncLR(noprime(res.L), noprime(res.R), res.sv, res.ov_before, res.ov_after)
end

"""
Applies MPO to the right and truncates on the RTM |AR*R><L| by building it explicitly \\
Returns `TruncLR` (destructures as `(LEFT, RIGHT, SV)`)
"""
function trapply(alg, ψL::MPS, A::MPO, ψR::MPS; kwargs...)
    res = trcontract(alg, ψL, A, ψR; kwargs...)
    return TruncLR(noprime(res.L), noprime(res.R), res.sv, res.ov_before, res.ov_after)
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