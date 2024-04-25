# ! experimental and not used for now 

""" timeMPO structure for cheaply containing temporal MPO structures, which are 
(except for the first and last site) translationally invariant """
struct timeMPO
    p::pparams
    func_expH::Function
    fold_op::Vector{ComplexF64}
    eH::MPO
    folded::Bool
    WWc::ITensor
    tMPO::MPO

    function timeMPO(p::pparams, func_expH::Function, fold_op::Vector{ComplexF64}, Nsteps::Int)
    
        folded = true
        time_sites = siteinds("S=3/2", Nsteps)

        # Real time evolution
        space_sites = siteinds("S=1/2", 3; conserve_qns = false)
        eH = func_expH(space_sites, p)

        tMPO = build_folded_tMPO_new(eH, p.init_state, fold_op, time_sites)

        WWc = build_WWc(eH)[1]

        new(p, func_expH, fold_op, eH, folded, WWc, tMPO)
    end

end

function ITensors.MPO(tt::timeMPO, Nsteps::Int)
    return 0
end


""" * TODO does this work ? 
Extend by 1 the tMPO """
function extend_timeMPO!(inTM::timeMPO)
    mpolen = length(inTM.tMPO)

    if mpolen == 1  # need to fix both 
        first = inTM.WWc
        second = inTM.WWc 
    end
    
    # we only extend the bulk. Insert at the end  
    wlinkinds = linkinds(eH) #?
    lk = commonind(inTM.tMPO[mpolen-1],inTM.tMPO[mpolen])
    lkp = Index(4, "Link,rotl=$mpolen") 
    newW = inTM.Wc * delta(wlinkinds[1], lk) *  delta(wlinkinds[2], lkp)

    insert!(inTM.tMPO.data, mpolen, newW)
  
    inTM.tMPO[mpolen+1] = inTM.tMPO[mpolen+1] * delta(lk, lkp)

end
