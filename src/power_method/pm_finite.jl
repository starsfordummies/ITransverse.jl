""" Applies MPOs in MPO_list to a random MPS. 
TODO: all MPOs must share same physical sites... """
function pm_finite(MPO_list)

    cutoff = 1e-8
    maxbondim = 128

    truncp = TruncParams(cutoff, maxbondim)

    ts = [siteind(MPO_list[1], i) for i in eachindex(MPO_list[1])]
    #left = random_mps(siteinds(MPO_list[1], linkdims=1)) 
    rr = random_mps(ts, linkdims=1) 

    p = Progress(length(MPO_list); showspeed=true)  #barlen=40

    for Wj in MPO_list
        rr, vn_ent = pm_step(Wj, rr, truncp)
        next!(p; showvalues = [(:Info,"[Smax=$(maximum(vn_ent)), chi=$(maxlinkdim(rr))" )])

    end

end


function pm_step(in_mpo::MPO, rr::MPS, truncp::TruncParams)
    cutoff = truncp.cutoff
    maxdim = truncp.maxbondim

    rr = apply(in_mpo, rr; cutoff, maxdim)
    sjj = vn_entanglement_entropy(rr)

    return rr, sjj

end
