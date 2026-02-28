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
            trrhoL *= trace_combinedind(psiL[kk], siteind(psiL,kk)) 
            trrhoR *= trace_combinedind(psiR[kk], siteind(psiR,kk)) 
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
            BLblock *= trace_combinedind(psiL[kk], siteind(psiL,kk)) 
            BRblock *= trace_combinedind(psiR[kk], siteind(psiR,kk)) 
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
            ALblock *= trace_combinedind(psiL[kk], siteind(psiL,kk)) 
            ARblock *= trace_combinedind(psiR[kk], siteind(psiR,kk)) 
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
            trrhoL *= trace_combinedind(psiL[kk], siteind(psiL,kk)) 
            trrhoR *= trace_combinedind(psiR[kk], siteind(psiR,kk)) 
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
