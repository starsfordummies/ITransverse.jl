using ITensors
using Plots
include("../power_method/compute_entropies.jl")



function build_rdmcut(psi, cut::Int, cut2::Int, center::Bool)
    print("Building rho traced over $cut - $cut2 legs")

    s = siteinds(psi)
    # OPEN LEGS between cut and cut2 ! 
    rho = ITensor(1.)

    for jj in eachindex(psi)
        rho = rho * psi[jj]
        rho = rho * dag(psi[jj]')
        if center
            if jj in cut:cut2
                # Trace it
                rho *= delta(s[jj],s[jj]' )
            end
    
        else
            if jj ∉ cut:cut2 # jj in 1:cut || jj in cut2:length(psi)
                rho *= delta(s[jj],s[jj]' )
            end
        end
    end

    open_inds = inds(rho,plev=1)

    if isempty(open_inds)
        print("contracted everything")
    else
        rho = rho * combiner(open_inds) * combiner(noprime(open_inds))
    end

    return rho 
end



function build_rdmcut_virtual(psi, cut::Int, cut2::Int)
    

    psi_ortho_5 = orthogonalize(psi,cut2-2) ## FIXME very hacky

    s = siteinds(psi_ortho_5)

    rho = ITensor(1.)

    for jj in cut:cut2
        rho = rho * psi_ortho_5[jj]
        rho = rho * dag(psi_ortho_5[jj]')
        rho *= delta(s[jj],s[jj]' )
    end
    
    return rho 

end


function main_check_rdmdiag()

sites = siteinds("S=1/2", 12; conserve_qns = false)


psi= randomMPS(sites, 40)

norm(psi)

orthogonalize!(psi,1)

s1= vn_entanglement_entropy(psi)

scatter(s1)

rho = outer(psi', psi)

rhoLeft = build_rdmcut(psi,1,7,true)
rhoRight = build_rdmcut(psi,8,12,true)

dLeft, _ = eigen(rhoLeft, inds(rhoLeft)[1], inds(rhoLeft)[2]);
dRight, _ = eigen(rhoRight, inds(rhoRight)[1],inds(rhoRight)[2]);

println(diag(dLeft.tensor))
println(diag(dRight.tensor))

# Try to calculate in clever way using canonical form 
psi_ortho = orthogonalize(psi,6)

rho1 = build_rdmcut(psi,3,8,true)  # open 1,2,9,10,11,12 (should be dim 2^6=64)
rho2 = build_rdmcut(psi,3,8,false) # open 3,4,5,6,7,8  (dim 2^6 also)


d1, _ = eigen(rho1, inds(rho1)[1], inds(rho1)[2]);
d2, _ = eigen(rho2, inds(rho2)[1],inds(rho2)[2]);


rho1 = build_rdmcut(psi,2,9,true)  # open 1,2,9,10,11,12 (should be dim 2^6=64)
rho2 = build_rdmcut(psi,2,9,false) # open 3,4,5,6,7,8  (dim 2^6 also)

d1, _ = eigen(rho1, inds(rho1)[1], inds(rho1)[2]);
d2, _ = eigen(rho2, inds(rho2)[1],inds(rho2)[2]);


rhovirt = build_rdmcut_virtual(psi,2,9)
irho = inds(rhovirt)
dv, _ = eigen(rhovirt, (irho[2],irho[4]),(irho[1],irho[3]));

diag(d1)
diag(d2)

diag(dv)


end
# looks good 