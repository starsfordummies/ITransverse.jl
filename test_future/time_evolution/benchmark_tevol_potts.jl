using ITensors
using ITensorMPS
using Plots
#using Polynomials
using LsqFit
# using LongRangeITensors
using ITransverse.ChainModels: build_H_potts, build_expH_potts_2o, build_expH_potts_2o_Jan, build_expH_potts_murg, build_expH_potts_symmetric_svd
using ITransverse.ITenUtils: vn_entanglement_entropy
# include("../models/potts.jl")


JJ = 1.0
ff = 1.

N = 10
sites_potts = siteinds("S=1", N)

max_chi = 300
SVD_cutoff = 1e-8


overlaps = Vector{Float64}()
overlaps_mu = Vector{Float64}()
overlaps_2_mu = Vector{Float64}()
overlaps_svd_mu = Vector{Float64}()


dts = Vector{Float64}()
Tmaxs = Vector{Float64}()


psi_prod = productMPS(ComplexF64, sites_potts, "↑") 
#psi_prod = productMPS(ComplexF64, sites_potts, "↑") 

psi_prod = randomMPS(ComplexF64, sites_potts, linkdims=10) 


Hpotts = build_H_potts(sites_potts, JJ, ff)

pl1 = plot()

function test1()
# for Tmax in [3.] # range(1,4,step=0.5)
#     for dt in [0.1] #  range(0.01, 0.2, step=0.01)
    
Tmax = 3.0
dt = 0.1 

        tsteps = trunc(Int,Tmax/dt)

        psi_tdvp = ITensorTDVP.tdvp(
                Hpotts,
                psi_prod,
                -im * dt; # 'real' time evolution according to U(τ) ≈ exp(τ * H) = exp(-im*dt * H)
                nsweeps = tsteps,
                maxdim = max_chi,
                cutoff = SVD_cutoff,
                normalize = true,
                outputlevel=1,
        )


        exphp = build_expH_potts_2o(sites_potts, JJ, ff, dt)

        psi_u2 = deepcopy(psi_prod)

        @time for (nt, t) in enumerate(range(dt, step = dt, length = tsteps))
            psi_u2 = apply(exphp, psi_u2; normalize = true, cutoff = SVD_cutoff, maxdim = max_chi)
            println("nt=$(nt),\tt=$(t),\tmaxbondim = $(maxlinkdim(psi_u2))")
        end


        # exphp_jan = build_expH_potts_2o_Jan(sites_potts, JJ, ff, dt)

        # psi_u2_j = deepcopy(psi_prod)

        # #@show siteinds(psi_u2_j)
        # #@show siteinds(exphp_jan)

        # @time for (nt, t) in enumerate(range(dt, step = dt, length = tsteps))
        #     psi_u2_j = apply(exphp_jan, psi_u2_j; normalize = true, cutoff = SVD_cutoff, maxdim = max_chi)
        #     println("nt=$(nt),\tt=$(t),\tmaxbondim = $(maxlinkdim(psi_u2_j))")
        # end

        exphp_murg = build_expH_potts_murg(sites_potts, JJ, ff, dt)

        psi_murg = deepcopy(psi_prod)

        @time for (nt, t) in enumerate(range(dt, step = dt, length = tsteps))
            psi_murg = apply(exphp_murg, psi_murg; normalize = true, cutoff = SVD_cutoff, maxdim = max_chi)
            println("nt=$(nt),\tt=$(t),\tmaxbondim = $(maxlinkdim(psi_murg))")
        end


        exphp_svd = build_expH_potts_symmetric_svd(sites_potts, JJ, ff, dt)

        psi_svd = deepcopy(psi_prod)

        @time for (nt, t) in enumerate(range(dt, step = dt, length = tsteps))
            psi_svd = apply(exphp_svd, psi_svd; normalize = true, cutoff = SVD_cutoff, maxdim = max_chi)
            println("nt=$(nt),\tt=$(t),\tmaxbondim = $(maxlinkdim(psi_svd))")
        end

        overlap_2 =  abs(inner(psi_tdvp, psi_u2))
        println("<TDVP(ψ) | Ut2(ψ)> = $(overlap_2)")

        overlap_murg =  abs(inner(psi_tdvp, psi_murg))
        println("<TDVP(ψ) | murg(ψ)> = $(overlap_murg)")

        overlap_svd =  abs(inner(psi_tdvp, psi_svd))
        println("<TDVP(ψ) | svd(ψ)> = $(overlap_murg)")

        # overlap_2j =  abs(inner(psi_tdvp, psi_u2_j))
        # println("<TDVP(ψ) | svd(ψ)> = $(overlap_2j)")

        overlap_2_murg =  abs(inner(psi_murg, psi_u2))
        println("<murg(ψ) | Ut_2(ψ)> = $(overlap_2_murg)")

        # overlap_2j_murg =  abs(inner(psi_murg, psi_u2_j))
        # println("<murg(ψ) | Ut_2j(ψ)> = $(overlap_2j_murg)")

        # overlap_2_2j =  abs(inner(psi_u2_j, psi_u2))
        # println("<Ut_2(ψ) | Ut_2_j(ψ)> = $(overlap_2_2j)")

        overlap_svd_murg =  abs(inner(psi_svd, psi_murg))
        println("<murg(ψ) | svd(ψ)> = $(overlap_svd_murg)")

        ev_tdvp =  expect(psi_tdvp, "Sz")
        ev2 =  expect(psi_u2, "Sz")

        # ev2j =  expect(psi_u2_j, "Sz")

        evmurg =  expect(psi_murg, "Sz")
        evsvd =  expect(psi_svd, "Sz")

        @show (ev_tdvp)
        @show (ev2)
        # @show (ev2j)
        @show(evmurg)
        @show(evsvd)

        plot!(pl1, ev_tdvp, label="tdvp")
        scatter!(pl1, ev2, label="o2", markersize=3)
        # scatter!(pl1, ev2j, label="o2j")
        scatter!(pl1, evmurg, label="murg",markersize=4)
        scatter!(pl1, evsvd, label="svd")

        # push!(Tmaxs, Tmax)
        # push!(dts, dt)
        # push!(overlaps, overlap)
        # push!(overlaps_mu, overlap_murg)
        # push!(overlaps_2_mu, overlap_2_murg)
        # push!(overlaps_svd_mu, overlap_svd_murg)


        

#    end
#end

# print(overlaps)
# print(overlaps_mu)
# print(overlaps_2_mu)

plot(pl1)
return ev_tdvp, evmurg, evsvd, ev2
end


function test_gs()

N = 60
sites_potts = siteinds("S=1", N)

max_chi = 300
SVD_cutoff = 1e-10

psi_prod = productMPS(ComplexF64, sites_potts, "↑") 


dt = -im*0.1 

Hpotts = build_H_potts(sites_potts, JJ, ff)

e0, psi_gs_dmrg = dmrg(Hpotts, psi_prod; nsweeps=10, maxdim = max_chi, cutoff = SVD_cutoff)

exphp = build_expH_potts_symmetric_svd(sites_potts, JJ, ff, dt)


psi_gs_H = deepcopy(psi_prod)
psi_gs_eH = deepcopy(psi_prod)

@time for jj = 1:50
    psi_gs_H = apply(Hpotts, psi_gs_H; normalize = true, cutoff = SVD_cutoff, maxdim = max_chi)
    psi_gs_eH = apply(exphp, psi_gs_eH; normalize = true, cutoff = SVD_cutoff, maxdim = max_chi)
    @show inner(psi_gs_H, psi_gs_eH)
    @show inner(psi_gs_eH, psi_gs_dmrg)
    @show maxlinkdim(psi_gs_H), maxlinkdim(psi_gs_eH), maxlinkdim(psi_gs_dmrg)
end

s1 = vn_entanglement_entropy(psi_gs_H)
s2 = vn_entanglement_entropy(psi_gs_eH)
s3 = vn_entanglement_entropy(psi_gs_dmrg)

scatter(s1)
scatter!(s2)
scatter!(s3)

func1(x, p) = p[1] .+ p[2]/3 .*log.(2*N/π * sin.(π*x/N))

xss =  0.5:1:59.0 
f1 = curve_fit(func1, xss, s2, [0.1, 0.2])
f2 = curve_fit(func1, xss, s3, [0.1, 0.2])

@show f1.param
@show f2.param 
end



function test_gs_ent()

    N = 60
    sites_potts = siteinds("S=1", N)
    
    max_chi = 300
    SVD_cutoff = 1e-10
    
    psi_prod = productMPS(ComplexF64, sites_potts, "↑") 
    
    
    dt = -im*0.1 
    
    Hpotts = build_H_potts(sites_potts, JJ, ff)
    
    e0, psi_gs_dmrg = dmrg(Hpotts, psi_prod; nsweeps=10, maxdim = max_chi, cutoff = SVD_cutoff)
    
    exphp = build_expH_potts_symmetric_svd(sites_potts, JJ, ff, dt)
    
    
    psi_gs_eH = deepcopy(psi_prod)
    
    anim = @animate for jj = 1:600
        psi_gs_eH = apply(exphp, psi_gs_eH; normalize = true, cutoff = SVD_cutoff, maxdim = max_chi)
        overl = inner(psi_gs_eH, psi_gs_dmrg)

        s2 = vn_entanglement_entropy(psi_gs_eH)
        s3 = vn_entanglement_entropy(psi_gs_dmrg)

        func1(x, p) = p[1] .+ p[2]/6 .*log.(2*N/π * sin.(π*x/N))
        
        xss =  0.5:1:59.0 
        f1 = curve_fit(func1, xss, s2, [0.1, 0.2])
        #f2 = curve_fit(func1, xss, s3, [0.1, 0.2])
        
        plot(s3, label="dmrg", legend=:bottom)
        scatter!(s2, label="$(jj) - c = $(f1.param[2])")
         
        #@show maxlinkdim(psi_gs_H), maxlinkdim(psi_gs_eH)
    end
    
    # scatter(s1)
    # scatter!(s2)
    # scatter!(s3)
    
    # func1(x, p) = p[1] .+ p[2]/3 .*log.(2*N/π * sin.(π*x/N))
    
    # xss =  0.5:1:59.0 
    # f1 = curve_fit(func1, xss, s2, [0.1, 0.2])
    # f2 = curve_fit(func1, xss, s3, [0.1, 0.2])
    
    # @show f1.param
    # @show f2.param 

    return anim 
    end
    
    
#a,b,c,d = test1()
 
#test_gs()
anim = test_gs_ent()

gif(anim,fps=20)
    