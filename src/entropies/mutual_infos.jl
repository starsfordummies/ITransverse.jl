""" Assume folded tMPS with physical indices time = fw x back,
computes Renyi2 mutual info of IF seen as density matrix (operator) """
function renyi2_mutual_folded_bipartition_rho(psi::MPS; normalize::String="trrho")

    if normalize == "norm_psi"
        psi = normalize(psi)
    end

    W = MPO(copy(tensors(psi)))


    # Reopen inds
    sites_phys = [Index(2,"Site,n=$i") for i in eachindex(W)]
    for kk in eachindex(psi)
        comb = combiner(sites_phys[kk], sites_phys[kk]')
        W[kk] *= dag(replaceind(comb, combinedind(comb), siteind(psi,kk)))
    end

    if normalize == "trrho"
        # Trace over rho, 
        trrho = ITensors.OneITensor()
        for kk in eachindex(W)
            trrho *= (W[kk] * delta(sites_phys[kk], sites_phys[kk]'))
        end

        W = W/scalar(trrho)
    end


    mutualsA = []
    mutualsB = []

    @showprogress for cut =  0:length(W) 

        # SA computed from tr(trB(rho)^2)
        Ablock = ITensor(1.)
        for kk = 1:cut
            Ablock *= W[kk] 
            @assert ndims(Ablock) <= 4 "[A $(kk) inds? $(inds(Ablock))"
            Ablock *= W[kk]'
            Ablock *= delta(sites_phys[kk], sites_phys[kk]'')
            @assert ndims(Ablock) <= 2  "[A $(kk) inds? $(inds(Ablock))"
        end

        halfBblock = ITensor(1.)
        for kk = reverse(cut+1:length(W))
            halfBblock *= (W[kk] * delta(sites_phys[kk], sites_phys[kk]'))
            @assert ndims(halfBblock) <= 1  "[B $(kk) inds? $(inds(halfBblock))"

        end

        SA = (halfBblock * Ablock) 
        SA = scalar(SA * halfBblock')



        # Now compute SB = tr(trA(rho)^2)

        halfAblock = ITensor(1.)
        for kk = 1:cut
            halfAblock *= (W[kk] * delta(sites_phys[kk], sites_phys[kk]'))
        end

        Bblock = ITensor(1.)
        for kk = reverse(cut+1:length(W))
            Bblock *= W[kk] 
            Bblock *= W[kk]'
            Bblock *= delta(sites_phys[kk], sites_phys[kk]'')
        end

        SB = halfAblock * Bblock
        SB = scalar(SB * halfAblock')



        push!(mutualsA, SA)
        push!(mutualsB, SB)
    end

    @assert mutualsA[1] ≈ mutualsB[end]
    @assert mutualsA[end] ≈ mutualsB[1]

    return -log.(mutualsA), -log.(mutualsB), -log.(mutualsA[end]), -log.(mutualsA) -log.(mutualsB) .+log.(mutualsA[end])
end


""" Assume folded tMPS with physical indices time = fw x back,
computes Renyi2 mutual info of IF seen as density matrix (operator) """
function renyi2_mutual_folded_bipartition_rho_2(psi::MPS; normalize::String="trrho")

    if normalize == "norm_psi"
        psi = normalize(psi)
    end

    W = psi

    if normalize == "trrho"
        # Trace over rho, 
        trrho = ITensors.OneITensor()
        for kk in eachindex(W)
            trrho *= ITransverse.trace_combinedind(W[kk], siteind(W,kk)) 
        end

        W = W/scalar(trrho)
    end


    mutualsA = []
    mutualsB = []

    @showprogress for cut =  0:length(W) 

        # SA computed from tr(trB(rho)^2)
        Ablock = ITensor(1.)
        for kk = 1:cut
            Ablock *= W[kk]
            Ablock = ptranspose_contract(Ablock, W[kk]', siteind(W,kk), siteind(W,kk)')
            @assert ndims(Ablock) <= 2  "[A $(kk) inds? $(inds(Ablock))"
        end

        halfBblock = ITensor(1.)
        for kk = reverse(cut+1:length(W))
            halfBblock *= ITransverse.trace_combinedind(W[kk], siteind(W,kk))
            @assert ndims(halfBblock) <= 1  "[B $(kk) inds? $(inds(halfBblock))"

        end

        SA = (halfBblock * Ablock) 
        SA = scalar(SA * halfBblock')



        # Now compute SB = tr(trA(rho)^2)

        halfAblock = ITensor(1.)
        for kk = 1:cut
            halfAblock *= ITransverse.trace_combinedind(W[kk], siteind(W,kk))
        end

        Bblock = ITensor(1.)
        for kk = reverse(cut+1:length(W))
            Bblock *= W[kk]
            Bblock = ptranspose_contract(Bblock, W[kk]', siteind(W,kk), siteind(W,kk)')
        end

        SB = halfAblock * Bblock
        SB = scalar(SB * halfAblock')



        push!(mutualsA, SA)
        push!(mutualsB, SB)
    end

    @assert mutualsA[1] ≈ mutualsB[end]
    @assert mutualsA[end] ≈ mutualsB[1]

    return -log.(mutualsA), -log.(mutualsB), -log.(mutualsA[end])
end


""" Takes as input two folded tMPS (left-right) and computes Renyi2 mutual info

joining legs

For rhoA 
A: fL1-bL2, bL1-fL2, fR1-bR2, bR1-fR2
B:  fL1-fR1, bL1-bR1 (same for 2)

For rhoB it's the other way around B<->A
"""
function renyi2_mutual_folded_bipartition_rhoLrhoR(psiL::MPS, psiR::MPS; normalize::String="trrho", swap_prod::Bool=false)

    @assert length(psiL) == length(psiR)

    psiR = sim(linkinds, psiR)

    if normalize == "trrho"
        # Trace over rho, ie. connect all Lf with Lb and Rf with Rb
        trrhoL = ITensor(1)
        trrhoR = ITensor(1)

        for kk in eachindex(psiL)
            trrhoL *= ITransverse.trace_combinedind(psiL[kk], siteind(psiL,kk)) 
            trrhoR *= ITransverse.trace_combinedind(psiR[kk], siteind(psiR,kk)) 
        end

        psiL = psiL/scalar(trrhoL)
        psiR = psiR/scalar(trrhoR)


    else
        @warn "Invalid/no normalization? $(normalize)"
    end
    
    mutualsA = []
    mutualsB = []

    @showprogress for cut =  0:length(psiL) 

        # rho_A = Tr_B (rho_AB)

        ALRblock = ITensor(1)
        for kk = 1:cut
            ALRblock *= psiL[kk] 
            if swap_prod 
                ALRblock = ptranspose_contract(ALRblock, psiR[kk], siteind(psiR,kk), siteind(psiR,kk))  # or: ptranspose_contract()
            else
                ALRblock *= psiR[kk]  
            end
            @assert ndims(ALRblock) <= 2  "[A $(kk) inds? $(inds(ALRblock))"
        end

        BLblock = ITensor(1)
        BRblock = ITensor(1)
        for kk = reverse(cut+1:length(psiL))
            BLblock *= ITransverse.trace_combinedind(psiL[kk], siteind(psiL,kk)) 
            BRblock *= ITransverse.trace_combinedind(psiR[kk], siteind(psiR,kk)) 
            @assert ndims(BLblock) <= 1  "[B $(kk) inds? $(inds(BLblock))"
            @assert ndims(BRblock) <= 1  "[B $(kk) inds? $(inds(BRblock))"
        end
       
        SA = ALRblock * BLblock
        SA = SA * BRblock

        SA = scalar(SA)

        push!(mutualsA, SA)

        #
        # The other way around for SB 
        #

        ALblock = ITensor(1)
        ARblock = ITensor(1)

        for kk = 1:cut
            ALblock *= ITransverse.trace_combinedind(psiL[kk], siteind(psiL,kk)) 
            ARblock *= ITransverse.trace_combinedind(psiR[kk], siteind(psiR,kk)) 
            @assert ndims(ALblock) <= 1  "[A $(kk) inds? $(inds(ALblock))"
            @assert ndims(ARblock) <= 1  "[A $(kk) inds? $(inds(ARblock))"

        end

        BLRblock = ITensor(1)
        for kk = reverse(cut+1:length(psiL))
            BLRblock *= psiL[kk] 
            if swap_prod 
                BLRblock = ptranspose_contract(BLRblock, psiR[kk], siteind(psiR,kk), siteind(psiR,kk))  # or: ptranspose_contract()
            else
                BLRblock *= psiR[kk]  
            end
            @assert ndims(BLRblock) <= 2  "[A $(kk) inds? $(inds(BLRblock))"
        end

       
        SB = BLRblock * ALblock
        SB = SB*ARblock
       
        SB = scalar(SB)

        push!(mutualsB, SB)

    end

    @assert mutualsA[1] ≈ mutualsB[end]
    @assert mutualsA[end] ≈ mutualsB[1]

    return -log.(mutualsA), -log.(mutualsB), -log.(mutualsA[end])
end


 

""" Takes as input two folded tMPS (left-right) and computes Renyi2 mutual info

joining legs

For rhoA 
A: fL1-bL2, bL1-fL2, fR1-bR2, bR1-fR2
B:  fL1-fR1, bL1-bR1 (same for 2)

For rhoB it's the other way around B<->A
"""
function renyi2_mutual_folded_bipartition_LR(psiL::MPS, psiR::MPS; normalize::String="trrho")

    @assert length(psiL) == length(psiR)
    @assert siteinds(psiL) == siteinds(psiR)

    psiR = sim(linkinds, psiR)

    if normalize == "trrho"
        # Trace over rho, ie. connect all Lf with Lb and Rf with Rb
        trrhoL = ITensor(1)
        trrhoR = ITensor(1)

        for kk in eachindex(psiL)
            trrhoL *= ITransverse.trace_combinedind(psiL[kk], siteind(psiL,kk)) 
            trrhoR *= ITransverse.trace_combinedind(psiR[kk], siteind(psiR,kk)) 
        end

        psiL = psiL/scalar(trrhoL)
        psiR = psiR/scalar(trrhoR)


    elseif normalize == "trtau"
        trtau = ITensor(1)
        for kk in eachindex(psiL)
            trtau *= psiL[kk] 
            trtau *= psiR[kk]
        end

        psiL = psiL / sqrt(scalar(trtau))
        psiR = psiR / sqrt(scalar(trtau))

    else
        @warn "Invalid/no normalization? $(normalize)"
    end
    
    mutualsA = []
    mutualsB = []

    @showprogress for cut =  0:length(psiL) 

        # rho_A = Tr_B (rho_AB)

        Ablock = ITensor(1)
        for kk = 1:cut
            Ablock *= psiL[kk] 
            Ablock *= psiR[kk]
            @assert ndims(Ablock) <= 2  "[A $(kk) inds? $(inds(Ablock))"
        end

        Bblock = ITensor(1)
        for kk = reverse(cut+1:length(psiL))
            Bblock *= psiL[kk] 
            Bblock *= prime(linkinds, psiR)[kk]
            @assert ndims(Bblock) <= 2  "[B $(kk) inds? $(inds(Bblock))"
        end
       
        SA = Ablock * Bblock
        @assert ndims(SA) <= 2
        SA *= prime(Ablock)
        @assert ndims(SA) <= 2
        SA *= swapprime(Bblock, 1=>0)

        SA = scalar(SA)

        push!(mutualsA, SA)

        #
        # The other way around for SB 
        #

        ALRblock = ITensor(1)
        for kk = 1:cut
            ALRblock *= WL[kk] 
            ALRblock *= WR[kk]
            @assert ndims(ALRblock) <= 2  "[A $(kk) inds? $(inds(ALRblock))"
        end

        BLblock = ITensor(1)
        BRblock = ITensor(1)
        for kk = reverse(cut+1:length(WL))
            BLblock *= WL[kk] 
            BLblock *= prime(replaceinds(WL[kk], siteinds(WL,kk) => reverse(siteinds(WL,kk))), 2, "L")
            BRblock *= WR[kk] 
            BRblock *= prime(replaceinds(WR[kk], siteinds(WR,kk) => reverse(siteinds(WR,kk))), 2, "R")
            @assert ndims(BLblock) <= 2  "[B $(kk) inds? $(inds(BLblock))"
            @assert ndims(BRblock) <= 2  "[B $(kk) inds? $(inds(BRblock))"
        end

       
        SB = BLblock * ALRblock
        @assert ndims(SB) <= 2
        SB *= BRblock
        @assert ndims(SB) <= 2
        SB *= ALRblock''

        SB = scalar(SB)

        push!(mutualsB, SB)

    end

    @assert mutualsA[1] ≈ mutualsB[end]
    return mutualsA, mutualsB
end




""" Trace an ITensor over combined indices given by the combiner """
function trace_cind(a::ITensor, combiner::ITensor)
    (_, c1, c2) = inds(combiner)
    a = a * dag(combiner)
    a = a * delta(c1,c2)
    return a 
end


# Assume we work with combined fw-back indices
function trace_cind(A::ITensor, iA::Index)

        dA = dim(iA)
        sqdA = isqrt(dA)

        @assert hasind(A, iA)

        temp_i1 = Index(sqdA)
        temp_i2 = Index(sqdA)

        comb = combiner(temp_i1, temp_i2)
        comb = replaceind(comb, combinedind(comb), iA)
        
        trace_combinedind(A, comb)
end


function ptr_contract(A::ITensor, B::ITensor, iA::Index, iB::Index)

    @assert hasind(A, iA)
    @assert hasind(B, iB)
   
    dA = dim(iA)
    sqdA = isqrt(dA)

    @assert hasind(A, iA)

    temp_i1 = Index(sqdA)
    temp_i2 = Index(sqdA)


    c1 = combiner(temp_i1, temp_i2)
    c2 = combiner(temp_i2, temp_i1)

    combA = A * replaceind(c1, combinedind(c1), iA)
    combB = B * replaceind(c2, combinedind(c2), iB)

    #@show commoninds(combA, combB)

    AB = combA * combB

    return AB

end

""" Ugly chi^5 chunk builder for symmetric case """
function tm_chunk(psi::MPS, i1::Int, i2::Int, prev_chunk::ITensor=ITensor(1); flip_fb::Bool, contract_from_right::Bool=false)
    tm = prev_chunk
    ss = siteinds(psi)
    psi2 = prime(linkinds, psi)
    rrange = contract_from_right ? (i2:-1:i1) : (i1:i2) 
    for kk in rrange
        tm *= psi[kk]
        if flip_fb
            tm = ptr_contract(tm, psi2[kk], ss[kk], ss[kk])  # or: ptranspose_contract()
        else
            tm *= psi2[kk]
        end
        @assert ndims(tm) < 5 "?? $(inds(tm))"
    end

    return tm
end





""" Ugly chi^5 chunk builder for symmetric case """
function ptr_chunk(psi::MPS, n1::Int, n2::Int)
    chunk = ITensor(1)
    ss = siteinds(psi)

    for kk = n1:n2
        chunk *= trace_cind(psi[kk], ss[kk])
    end

    @assert ndims(chunk) < 3 "?? $(inds(chunk))"

    return chunk
end

function trrho_lr(psi::MPS, iA::Int, fA::Int, iB::Int=length(psi)+1, fB::Int=length(psi)+1)

    LL = length(psi)
    ss = siteinds(psi)

    psip = prime(linkinds, psi)

    trrho = ITensor(1.)

    for kk = 1:iA-1
        trrho *= psi[kk] 
        trrho *= psip[kk]
    end
    for kk = iA:fA 
        trrho *= trace_combinedind(psi[kk], ss[kk])
        trrho *= trace_combinedind(psip[kk], ss[kk])
    end
    for kk = fA+1:iB-1
        trrho *= psi[kk] 
        trrho *= psip[kk]
    end

    if iB <= LL
        for kk = iB:fB
            trrho *= trace_combinedind(psi[kk], ss[kk]) 
            trrho *= trace_combinedind(psip[kk], ss[kk])
        end
        for kk = fB+1:LL
            trrho *= psi[kk] 
            trrho *= psip[kk]
        end
    end
    
    return scalar(trrho)
end





""" Computes left-right RDM purities for symmetric case L=R """
function rho2_lr(psi::MPS, iA::Int, fA::Int, iB::Int, fB::Int)


    blockL = tm_chunk(psi::MPS, 1, iA-1; flip_fb=false)
    blockLL = tm_chunk(psi::MPS, iA, iB-1, blockL; flip_fb=false)

    @assert ndims(blockLL) < 3 "?? $(inds(blockLL))"

    blockR = tm_chunk(psi::MPS, fB+1, length(psi),; flip_fb=false, contract_from_right=true)
    blockRR = tm_chunk(psi::MPS, fA+1, fB, blockR; flip_fb=false, contract_from_right=true)

    @assert ndims(blockRR) < 3 "?? $(inds(blockRR))"

    # Normalization(s?): tr(rho) = 1 

    trrho_A = trrho_lr(psi, iA, fA)
    trrho_B = trrho_lr(psi, iB, fB)
    trrho_AB = trrho_lr(psi, iA, fA, iB, fB)




    tr_block = tm_chunk(psi::MPS, iA, fA; flip_fb=true)

    rhoLR = blockL
    rhoLR *= prime(blockL,2)

    rhoLR *= prime(tr_block)
    rhoLR *= swapprime(tr_block, 1 => 3)

    @assert ndims(rhoLR) < 5

    @show inds(blockRR)


    rhoLR_A = scalar((rhoLR * blockRR) *  prime(blockRR,2))/(trrho_A^2)


    mid_block = tm_chunk(psi::MPS, fA+1, iB-1; flip_fb=false)
    rhoLR = rhoLR * mid_block 
    rhoLR *= prime(mid_block,2)


    tr_block = tm_chunk(psi::MPS, iB, fB; flip_fb=true, contract_from_right=true)
    rhoLR *= prime(tr_block)
    rhoLR *= swapprime(tr_block, 1 => 3)
    @assert ndims(rhoLR)< 5

    rhoLR *= blockR 
    rhoLR *= prime(blockR,2)

    rhoLR = scalar(rhoLR)/(trrho_AB^2)

    rhoLR_B = blockR
    rhoLR_B *= prime(blockR,2)

    rhoLR_B *= prime(tr_block)
    rhoLR_B *= swapprime(tr_block, 1 => 3)

    rhoLR_B = scalar((rhoLR_B * blockLL) *  prime(blockLL,2))/(trrho_B^2)

    return rhoLR_A, rhoLR_B, rhoLR
end




function rho2_left(psi::MPS, iA::Int, fA::Int, iB::Int, fB::Int)

    LL = length(psi)

    # Normalization: tr(rho) = 1 
    tr_rho = scalar(ptr_chunk(psi::MPS, 1, LL))

    #AB 
    block = ptr_chunk(psi::MPS, 1, iA-1)
    rhoL = block * prime(block)

    rhoL = tm_chunk(psi::MPS, iA, fA, rhoL; flip_fb=true)
    

    @assert ndims(rhoL) < 3
    block = ptr_chunk(psi::MPS, fA+1, iB-1)
    rhoL *= block 
    rhoL *= prime(block)

    rhoL = tm_chunk(psi::MPS, iB, fB, rhoL; flip_fb=true)
    @assert ndims(rhoL)< 3

    block = ptr_chunk(psi::MPS, fB+1, length(psi))
    rhoL *= block 
    rhoL *= prime(block)

    rhoAB = scalar(rhoL)/(tr_rho^2)


    #A 
    block = ptr_chunk(psi::MPS, 1, iA-1)
    rhoL = block * prime(block)

    rhoL = tm_chunk(psi::MPS, iA, fA, rhoL; flip_fb=true)

    @assert ndims(rhoL) < 3
    block = ptr_chunk(psi::MPS, fA+1, LL)
    rhoL *= block 
    rhoL *= prime(block)

    rhoA = scalar(rhoL)/(tr_rho^2)

    #B 
    block = ptr_chunk(psi::MPS, 1, iB-1)
    rhoL = block * prime(block)

    rhoL = tm_chunk(psi::MPS, iB, fB, rhoL; flip_fb=true)

    @assert ndims(rhoL) < 3
    block = ptr_chunk(psi::MPS, fB+1, LL)
    rhoL *= block 
    rhoL *= prime(block)

    rhoB = scalar(rhoL)/(tr_rho^2)

    return rhoA, rhoB, rhoAB
end


function compute_rho2s(psi::MPS, length_intervals::Int)
    LL = length(psi)
    mid = div(LL+1,2)


    rhos_lr = []
    rhos_left = []
    for distance=0:div(LL-2*length_intervals,2)-1
        iA = mid-distance-length_intervals
        fA = mid-distance
        iB = mid+distance+1
        fB = mid+distance+length_intervals

        @info "computing for [$(iA)-$(fA)]-[$(iB)-$(fB)]"
        #r2s_LR = rho2_lr(psi::MPS, iA, fA, iB, fB)

        r2s_left = rho2_left(psi::MPS, iA, fA, iB, fB)


        #push!(rhos_lr, r2s_LR)
        push!(rhos_left, r2s_left)
    end

    return rhos_lr, rhos_left

end


