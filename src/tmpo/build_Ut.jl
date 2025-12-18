""" Given an MPO U, builds the folded version  UxUdag of it, optionally with `new_siteinds` """
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

