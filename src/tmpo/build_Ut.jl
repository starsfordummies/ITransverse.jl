

timephysdim(b) = dim(b.rot_inds[:P])

function linkphysdim(b)
    @assert dim(b.rot_inds[:L]) == dim(b.rot_inds[:R]) 
    dim(b.rot_inds[:L])
end


""" WIP: from b to U(t) MPO """ 
function UtMPO(ss::Vector{<:Index}, b::T, imag::Bool=false) where T 
    Wl, Wc, Wr = get_Ws(b; imag)
    ri = b.rot_inds

    Nx = length(ss)

    links = [sim(ri[:R], tags="Link,l=$(ii)") for ii = 1:Nx-1]
    
    @show links 

    out_w = MPO(repeat(Wc,Nx))

    out_w[1] = replaceinds(Wl, ri[:R] => links[1], ri[:P] => ss[1], ri[:Ps] => ss[1]' )
    for ii in 2:length(ss)-1 
        replaceinds!(out_w[ii], ri[:L] => dag(links[ii-1]), ri[:R] => links[ii], ri[:P] => ss[ii],  ri[:Ps] => ss[ii]')
    end
    out_w[end] = replaceinds(Wr,  ri[:R] => dag(links[end]), ri[:P] => ss[end], ri[:Ps] => ss[end]' )

    return out_w
end



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

