
function tlrcontract(::Algorithm"naiveRTM", psiL::MPS, OL::MPO, OR::MPO, psiR::MPS; 
        preserve_mps_tags::Bool=false, # TODO not implemented yet 
        kwargs...)
    OpsiR = applyn(OR, psiR; truncate=false)
    psiLO = applyns(OL, psiL; truncate=false)  
    truncate_sweep(psiLO, OpsiR; kwargs...)
end

function tlrcontract(::Algorithm"naive", psiL::MPS, OL::MPO, OR::MPO, psiR::MPS; kwargs...)
    OpsiR, sv = tapply(Algorithm("naive"), OR, psiR; kwargs...)
    psiLO, _ = tapplys(Algorithm("naive"), OL, psiL; kwargs...)
    return psiLO, OpsiR, sv  
end

function tlrcontract(::Algorithm"densitymatrix", psiL::MPS, OL::MPO, OR::MPO, psiR::MPS; kwargs...)
    OpsiR, sv = tapply(Algorithm("densitymatrix"), OR, psiR; kwargs...)
    psiLO, _ = tapplys(Algorithm("densitymatrix"), OL, psiL; kwargs...)
    return psiLO, OpsiR, sv  
end

function tlrcontract(::Algorithm"notrunc", psiL::MPS, OL::MPO, OR::MPO, psiR::MPS; kwargs...)
    OpsiR = applyn(OR, psiR; truncate=false)
    psiLO = applyns(OL, psiL; truncate=false)
    sv = zeros(1,1)
    return psiLO, OpsiR, sv  
end

# (Cheating but maybe useful if) symmetric case: does not touch psiL, OL (assumed to be same as psiR, OR)
function tlrcontract(::Algorithm"RTMsym", psiL::MPS, OL::MPO, OR::MPO, psiR::MPS; kwargs...)
    OpsiR, sv = tcontract(Algorithm(:RTMsym), OR, psiR; kwargs...)
    return OpsiR, OpsiR, sv  
end


#### NEW LR apply 
function tlrapply(alg, psiL::MPS, OL::MPO, OR::MPO, psiR::MPS; kwargs...)
    tpsiL, tpsiR, sv = tlrcontract(alg, psiL, OL, OR, psiR; kwargs...)
    return noprime(tpsiL), noprime(tpsiR), sv
end



### Apply only one, then truncate 



function trcontract(::Algorithm"naiveRTM",
        ψL::MPS,
        AR::MPO,
        ψR::MPS;
        preserve_mps_tags::Bool=false, # TODO not implemented yet 
        kwargs...)
        
    OpsiR = applyn(AR, ψR; truncate=false)
    truncate_sweep(ψL, OpsiR; kwargs...)
end

# Just for conveninence, does not touch left MPS, only acts on Right 
function trcontract(::Algorithm"densitymatrix", psiL::MPS, AR::MPO, psiR::MPS; kwargs...)
    OpsiR, sv = tapply(Algorithm("densitymatrix"), AR, psiR; kwargs...)
    return psiL, OpsiR, sv  
end
function trcontract(::Algorithm"naive", psiL::MPS, AR::MPO, psiR::MPS; kwargs...)
    OpsiR, sv = tapply(Algorithm("naive"), AR, psiR; kwargs...)
    return psiL, OpsiR, sv  
end


""" 
Applies MPO to left-right and truncates on the RTM |AR R><L AL| \\
Allows for different length MPS/MPO \\
Returns LEFT, RIGHT, SV
"""
#TODO check should be as simple as this 
function tlcontract(alg, ψL::MPS, AL::MPO, ψR::MPS; kwargs...)
    ψR, ψL, sv = trcontract(alg, ψR, swapprime(AL, 0=>1, "Site"), ψL; kwargs...)
    return ψL, ψR, sv 
end

""" 
Applies MPO to the left and truncates on the RTM |R><L*AL| by building it explicitly \\
Returns LEFT, RIGHT, SV
"""
function tlapply(alg, ψL::MPS, A::MPO, ψR::MPS; kwargs...) 
     tpsiL, tpsiR, sv = tlcontract(alg, ψL, A, ψR; kwargs...)
     return noprime(tpsiL), noprime(tpsiR), sv
end

""" 
Applies MPO to the right and truncates on the RTM |AR*R><L| by building it explicitly \\
Returns LEFT, RIGHT, SV
"""
function trapply(alg, ψL::MPS, A::MPO, ψR::MPS; kwargs...) 
     tpsiL, tpsiR, sv = trcontract(alg, ψL, A, ψR; kwargs...)
     return noprime(tpsiL), noprime(tpsiR), sv
end

tlapply(ψL::MPS, A::MPO, ψR::MPS; alg=Algorithm(:naiveRTM), kwargs...) = tlapply(Algorithm(alg), ψL, A, ψR; kwargs...)
trapply(ψL::MPS, A::MPO, ψR::MPS; alg=Algorithm(:naiveRTM), kwargs...) = trapply(Algorithm(alg), ψL, A, ψR; kwargs...)

""" 
Applies AL to ψL from the left, AR to the right ψL, and truncates on the RTM |AR*R><L*AL| by building it explicitly. \\
Convention: when direction=:left, we PTR over left environments and, going right->left, we SVD the τ_R = tr_L(τ)
Returns LEFT, RIGHT, SV
"""
tlrapply(ψL::MPS, AL::MPO, AR::MPO, ψR::MPS; alg=Algorithm(:naiveRTM), kwargs...) = tlrapply(Algorithm(alg), ψL, AL, AR, ψR; kwargs...)