""" Join two MPOs fold-like, if `fold_op` != nothing we join the two sheets.
For this, we *remove* the *last* site tensors of both MPOS 
and replace them with the folding operator `fold_op`
"""
function combine_fold_mpos(W1::AbstractMPS, W2::AbstractMPS; fold_op=ComplexF64[1 0 ; 0 1], dag_W2::Bool=true)

    sites1_p =  siteinds(first, W1,plev=0)
    sites2_p =  siteinds(first, W2,plev=0)

    sites1_ps = siteinds(first, W1,plev=1)
    sites2_ps = siteinds(first, W2,plev=1)

    @assert length(sites1_p) == length(sites2_p)  "Only equal-length MPS/MPOs are supported for now"

    links1 = linkinds(W1)
    links2 = linkinds(W2)

    LL = !isnothing(fold_op) ? length(W1)-1 : length(W1)
    folding_tensor = !isnothing(fold_op) ? ITensor(fold_op, links1[end], links2[end]'') : nothing
 
    if dag_W2 
        W2 = dag(W2)
    end
 
    W12 = typeof(W1)(LL)
    comb_p = [ITransverse.ITenUtils.pcombiner(sites1_p[ii], sites2_p[ii]'', tags="Site,nn=$(ii)") for ii = 1:LL]
    comb_ps = comb_p
 
    comb_link = [combiner(links1[ii],links2[ii]'', tags="Link,ll=$(ii)") for ii = 1:LL-1]

    for ii = 1:LL
        W12[ii] = W1[ii]*(W2[ii]'')
        W12[ii] *= comb_p[ii]
    end

    if W12 isa MPO
        comb_ps = [combiner(sites1_ps[ii], sites2_ps[ii]'', tags="Site,nn=$(ii)") for ii = 1:LL]
        for ii = 1:LL
            replaceind!(comb_ps[ii], combinedind(comb_ps[ii]), combinedind(comb_p[ii])' )
            W12[ii] *= comb_ps[ii]
        end
    end


    W12[1] *= comb_link[1]
    for ii = 2:LL-1 
        W12[ii] *= comb_link[ii-1]
        W12[ii] *= comb_link[ii]
    end

    if !isnothing(folding_tensor)
        W12[end] *= comb_link[end]
        W12[end] *= folding_tensor 
    else
        W12[end] *= comb_link[end]
    end

    return W12, comb_ps, comb_p

end




""" Builds a folded MPS by joining two, returns the combined MPS and the combiners  """ 
function combine_mps_spacefold(W1::MPS, W2::MPS)
    #@assert siteinds(W1) == siteinds(W2)
    LL = length(W1)-1
    sites1 = siteinds(W1)
    sites2 = siteinds(W2)

    links1 = linkinds(W1)
    links2 = linkinds(W2)

    W12 = MPS(LL)
    comb_p = [ITransverse.ITenUtils.pcombiner(sites1[ii], sites2[ii]', tags="Site,nn=$(ii)") for ii = 1:LL]
 
    comb_link = [combiner(links1[ii],links2[ii]', tags="Link,ll=$(ii)") for ii = 1:LL-1]

    for ii = 1:LL
        W12[ii] = W1[ii]*(W2[ii]')
        W12[ii] *= comb_p[ii]
    end

    W12[1] *= comb_link[1]
    for ii = 2:LL-1 
        W12[ii] *= comb_link[ii-1]
        W12[ii] *= comb_link[ii]
    end
    W12[end] *= comb_link[end]

    return W12, comb_p

end



function reopen_inds!(WWm::MPS, combs)
    for (ii, c) in enumerate(combs)
        iinds = uniqueinds(inds(c), combinedind(c))
        WWm[ii] *= dag(c)
    end
    return WWm
end


""" Traces over combined indices of ITensors. On a vectorized DM it should basically give the norm of the state"""
function trace_mps_inds(WWm::MPS, combs)
    #WWmc = deepcopy(WWm)
    trace_mps_inds!(copy(WWm), combs)  #TODO copy() is enough or deepcopy() ? 
end

function trace_mps_inds!(WWm::MPS, combs)
    trace = ITensors.OneITensor()
    for (ii, c) in enumerate(combs)
        iinds = uniqueinds(inds(c), combinedind(c))
        WWm[ii] *= dag(c)
        WWm[ii] *= delta(iinds)
        trace *= WWm[ii]
    end

    return scalar(trace)
end


""" Trace an ITensor over combined indices given by the combiner """
function trace_combinedind(a::ITensor, combiner::ITensor)
    (_, c1, c2) = inds(combiner)
    a = a * dag(combiner)
    a = a * delta(c1,c2)
    return a 
end

