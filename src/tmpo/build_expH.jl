
function build_expH(tp::tMPOParams; dt::Number=tp.dt)
    tp.expH_func(tp.mp, dt)
end

""" Exp(-τH) for imaginary time evolution """
function build_expHim(tp::tMPOParams)
    tp.expH_func(tp.mp, -im*tp.dt )
end

""" Quick building a FwtMPOBlocks or FoldtMPOBlocks struct for playing around """
function quick_b(; folded::Bool=true)

    tp = ising_tp()
    b = folded ? FoldtMPOBlocks(tp) : FwtMPOBlocks(tp)

    return b
end


function folded_UUt(Ut::MPO; new_siteinds=nothing)

    N = length(Ut)

    UUt = MPO(N)

    sites_u = firstsiteinds(Ut)
    links_u = linkinds(Ut)

    for jj = 1:N
        UUt[jj] = Ut[jj] * dag(prime(Ut[jj],2))
        s = sites_u[jj]
        cs = combiner(s, dag(s)'', tags = tags(s); dir=ITensors.In)
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


timephysdim(b) = dim(b.rot_inds[:P])
linkphysdim(b) = dim(b.rot_inds[:L])