""" Given an MPO U, builds the folded version  UxUdag of it, optionally with `new_siteinds`
TODO: we can do it with combine_and_fold ?  """
function folded_UUt(Ut::MPO; new_siteinds=nothing)

    N = length(Ut)

    UUt = MPO(N)

    sites_u = firstsiteinds(Ut)
    links_u = linkinds(Ut)

    for jj = 1:N
        UUt[jj] = Ut[jj] * dag(prime(Ut[jj],2))
        s = sites_u[jj]
        cs = ITransverse.ITenUtils.pcombiner(s, dag(s)'', tags = tags(s); dir=ITensors.In)
        UUt[jj] = UUt[jj] * cs  #TODO CHECK THIS 
        UUt[jj] = UUt[jj] * dag(cs)'  #TODO CHECK THIS 
    end

    for jj=1:N-1
        l = links_u[jj]
        cl = combiner(l, dag(l)'', tags = "Link,l=$(jj)")
    
        UUt[jj] *= cl
        UUt[jj + 1] *= dag(cl)
       
    end

    if !isnothing(new_siteinds)
        replace_siteinds!(UUt, new_siteinds)
    end

    return UUt
end


function FoldITensor(a::Array; ind_fw=Index(size(a,1), "fw"), ind_back=Index(size(a,2), "back"))

    ft = if ndims(a) == 1 
        ia = Index(size(a,1))
        open_ind(ITensor(a, ia), ia, ind_fw, ind_back)
    elseif ndims(a) == 2
        # by default, first ind is fw, second is back 
        ITensor(a, ind_fw, ind_back)
    else
        @error "Not a good input tensor? $a"
        nothing
    end

    return ft
end

FoldITensor(a::ITensor; kwargs...) = FoldITensor(array(a); kwargs...)
 


""" Join two MPS/MPOs fold-like.
- If `fold_op` is not empty/nothing, we join the two sheets. For this, we **remove** the **last** site tensors of both MP*S 
and **replace** them with the folding operator `fold_op`
- If `fold_init_state` is not empty/nothing, we **remove** the **first** site tensors of both and replace them with the folded initial state
"""
function combine_and_fold(W1::AbstractMPS, W2::AbstractMPS; dag_W2::Bool=false,
    fold_op=nothing, fold_init_state=nothing,  new_siteinds=nothing)

    LL = length(W1)

    sites1_p =  siteinds(first, W1,plev=0)
    sites2_p =  siteinds(first, W2,plev=0)

    sites1_ps = siteinds(first, W1,plev=1)
    sites2_ps = siteinds(first, W2,plev=1)

    @assert LL == length(sites2_p)  "Only equal-length MPS/MPOs are supported for now: $(LL)-$(length(sites2_p))"

    links1 = linkinds(W1)
    links2 = linkinds(W2)


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


    for ii = 2:LL-1 
        W12[ii] *= comb_link[ii-1]
        W12[ii] *= comb_link[ii]
    end

    if !isnothing(fold_init_state)
        popfirst!(W12.data)
        W12[1] *= (FoldITensor(fold_init_state; ind_fw=links1[1], ind_back=links2[1]'') * comb_link[1])
    else
        W12[1] *= comb_link[1]
    end



    if !isnothing(fold_op)
        pop!(W12.data)
        W12[end] *= (FoldITensor(fold_op; ind_fw=links1[end], ind_back=links2[end]'') * comb_link[end])
    else
        W12[end] *= comb_link[end]
    end



    if !isnothing(new_siteinds)
        replace_siteinds!(W12, new_siteinds)
    end


    return W12, comb_ps, comb_p

end


function open_ind(a::ITensor, comb_ind::Index, i1::Index, i2::Index)
    c = combiner(i1,i2)
    replaceind!(c, combinedind(c), comb_ind)
    return a * dag(c)
end

function reopen_inds!(WWm::MPS, combs)
    for (ii, c) in enumerate(combs)
        #iinds = uniqueinds(inds(c), combinedind(c))
        WWm[ii] *= dag(c)
    end
    return MPO(WWm.data)
end


""" Reopen inds assuming they're folded, we make quite a few assumptions here... """
function reopen_inds!(folded_psi::MPS; different_fwback_inds::Bool=true)

    sqdim = isqrt(dim(siteind(folded_psi,1)))
    ss = siteinds(folded_psi)

    for i in eachindex(folded_psi)
        fw_ind, bk_ind = if different_fwback_inds
            Index(sqdim,"Site,n=$i,fw"), Index(sqdim,"Site,n=$i,back")'
        else
            temp = Index(sqdim,"Site,n=$i,unfold")
            temp, temp'
        end

        comb = combiner(fw_ind, bk_ind)
        kc = combinedind(comb)
        comb = replaceind(comb, kc, ss[i])
        folded_psi[i] *= dag(comb)
    end

    return MPO(folded_psi.data)
end

reopen_inds(folded_psi::MPS, combs) = reopen_inds!(copy(folded_psi), combs)
reopen_inds(folded_psi::MPS; kwargs...) = reopen_inds!(copy(folded_psi); kwargs...)



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
