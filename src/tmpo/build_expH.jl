
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


function folded_UUt(Ut::MPO)

    N = length(Ut)

    UUt = MPO(N)

    sites_u = firstsiteinds(Ut)
    links_u = linkinds(Ut)

    sites_uu = [Index(dim(sites_u[ii])^2,tags="Site,n=$(ii)") for ii = 1:N]
    links_uu = [Index(linkdim(Ut,ii)^2,tags="Link,l=$(ii)") for ii = 1:length(links_u)]

    WWl = Ut[1] * dag(prime(Ut[1],2))

    # Combine indices  
    CvR = combiner(links_u[1],links_u[1]''; tags="cwR")
    Cp = combiner(sites_u[1],sites_u[1]''; tags="cp")
    Cps = combiner(sites_u[1]',sites_u[1]'''; tags="cps")

    WWl = WWl * CvR * Cp * Cps

    UUt[1] = replaceinds(WWl, combinedind(CvR) => links_uu[1], combinedind(Cp) => sites_uu[1], combinedind(Cps) => sites_uu[1]') 

    for jj = 2:N-1

        WWc = Ut[jj] * dag(prime(Ut[jj],2))

        # Combine indices  
        CvL = combiner(links_u[jj-1],links_u[jj-1]''; tags="cwR")
        CvR = combiner(links_u[jj],links_u[jj]''; tags="cwR")
        Cp = combiner(sites_u[jj],sites_u[jj]''; tags="cp")
        Cps = combiner(sites_u[jj]',sites_u[jj]'''; tags="cps")

        WWc = (((WWc * CvR) * Cp) * Cps) * CvL

        UUt[jj] = replaceinds(WWc, combinedind(CvR) => links_uu[jj], combinedind(CvL) => links_uu[jj-1], combinedind(Cp) => sites_uu[jj], combinedind(Cps) => sites_uu[jj]') 
    end

    WWr = Ut[end] * dag(prime(Ut[end],2))

    # Combine indices  
    CvL = combiner(links_u[end],links_u[end]''; tags="cwL")
    Cp = combiner(sites_u[end],sites_u[end]''; tags="cp")
    Cps = combiner(sites_u[end]',sites_u[end]'''; tags="cps")

    WWr = WWr * CvL * Cp * Cps

    UUt[end] = replaceinds(WWr, combinedind(CvL) => links_uu[end], combinedind(Cp) => sites_uu[end], combinedind(Cps) => sites_uu[end]') 

    return UUt
end


timephysdim(b) = dim(b.rot_inds[:P])
linkphysdim(b) = dim(b.rot_inds[:L])