""" Applies MPOs in MPO_list to a random MPS. 
TODO: all MPOs must share same physical sites... """
function contract_finite(left_edge::MPS, MPO_list, right_edge::MPS)

    LL = length(MPO_list)
    @assert LL % 2 == 0

    cutoff = 1e-8
    maxbondim = 128

    truncp = TruncParams(cutoff, maxbondim)

    ts = [siteind(MPO_list[1], i) for i in eachindex(MPO_list[1])]
    #left = random_mps(siteinds(MPO_list[1], linkdims=1)) 
    ll = left_edge
    rr = right_edge

    p = Progress(div(LL,2) ; showspeed=true)  #barlen=40

    for jj = 1:div(LL,2)
        ll = applys(MPO_list[jj], rr; cutoff, maxdim=maxbondim)
        rr = apply(MPO_list[LL-jj+1], rr; cutoff, maxdim=maxbondim)

        next!(p; showvalues = [(:Info,"chi=$(maxlinkdim(rr))" )])

    end

    overlap_noconj(ll,rr)

end


function pm_step(in_mpo::MPO, rr::MPS, truncp::TruncParams)
    cutoff = truncp.cutoff
    maxdim = truncp.maxbondim

    rr = apply(in_mpo, rr; cutoff, maxdim)
    sjj = vn_entanglement_entropy(rr)

    return rr, sjj

end
