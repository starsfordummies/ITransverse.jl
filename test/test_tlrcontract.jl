using ITensors, ITensorMPS, ITransverse
using Test

""" Contracts and build/truncates over RTM """ 
function tlrcontract_old(
        ψL::MPS,
        AL::MPO,
        AR::MPO,
        ψR::MPS;
        cutoff = 1.0e-13,
        maxdim = max(maxlinkdim(AL) * maxlinkdim(ψL), maxlinkdim(AR) * maxlinkdim(ψR)),
        mindim = 1,
        kwargs...,
    )

    @assert length(ψL) == length(ψR)
    @assert length(AL) >= length(ψL)
    @assert length(AL) == length(AR)

    ### Indices prime contractions conventions
    # 
    # ψL--p'--AL--p--     --p'--AR--p--ψR 

    sR = firstsiteinds(AR, plev=1)
    sL = firstsiteinds(AL, plev=0)

    @assert noprime.(sR) == sL

    @assert firstsiteinds(AR, plev=0)[1:length(ψR)] == noprime.(firstsiteinds(AL, plev=1)[1:length(ψL)])

    N = length(AR)
    n = length(ψR)

    requested_maxdim = maxdim

    ψR_out = typeof(ψR)(N)
    ψL_out = typeof(ψL)(N)

    AL = swapprime(AL, 0 => 1, tags="Site")
    ALp = replaceprime(AL, 1 => 2)
    #sLp = firstsiteinds(ALp, plev=2)

    # Store the right environment tensors
    E = Vector{ITensor}(undef, N-1)

    E[1] = ψR[1] * AR[1] * AL[1] * ψL[1]

    for j in 2:n
        E[j] = E[j - 1] * ψR[j] * AR[j] * AL[j] * ψL[j]
    end
    for j in n+1:N-1 # only need N-1
        E[j] = E[j - 1] * AR[j] * AL[j]
    end

    # inds.(E)

    S_all = zeros(Float64, N-1, requested_maxdim)

    R, L = if N > n 
        AR[N], ALp[N]
    else
        ψR[N] * AR[N], ψL[N] * ALp[N]
    end

    r_renorm = nothing

    for j = reverse(n+1:N-1)

        # Determine smallest maxdim to use
        ciR = commoninds(R, E[j])
        ciL = commoninds(L, E[j])

        maxdim = min(dim(ciR), dim(ciL), requested_maxdim)

        rho = E[j] * L * R

        l = linkind(ψR, j)

        ts = isnothing(l) ? "" : tags(l)

        Ris = isnothing(r_renorm) ? IndexSet(sR[j+1]) : IndexSet(sR[j+1], r_renorm)

        @assert ndims(rho) < 5 "inds(rho) @site $j ? $(inds(rho))"

        #@show inds(rho)

        F = svd(rho, Ris; cutoff, maxdim, lefttags=ts, kwargs...)
        S, U, V = F.S, F.U, F.V
        r_renorm= F.u

        ψR_out[j+1] = U
        ψL_out[j+1] = V

        R = R * dag(U) * AR[j]
        L = L * dag(V) * ALp[j] 

        Svec = collect(S.tensor.storage.data)/sum(S)  
 
        S_all[j, 1:length(Svec)] .= Svec  
    
    end

    for j = reverse(1:n)

        # Determine smallest maxdim to use
        cipR = commoninds(ψR[j], E[j])
        ciAR = commoninds(AR[j], E[j])
        cipL = commoninds(ψL[j], E[j])
        ciAL = commoninds(ALp[j], E[j])
        prod_dimsR = dim(cipR) * dim(ciAR)
        prod_dimsL = dim(cipL) * dim(ciAL)

        maxdim = min(prod_dimsL, prod_dimsR, requested_maxdim)

        rho = E[j] * L * R

        l = linkind(ψR, j)

        ts = isnothing(l) ? "" : tags(l)

        Ris = isnothing(r_renorm) ? sR[j+1] : IndexSet(sR[j+1], r_renorm)

        @assert ndims(rho) < 5 "inds(rho) @site $j ? $(inds(rho))"

        F = svd(rho, Ris; cutoff, maxdim, lefttags=ts , kwargs...)
        S, U, V = F.S, F.U, F.V
        r_renorm = F.u

        ψR_out[j+1] = U
        ψL_out[j+1] = V

        R = R * dag(U) * ψR[j] * AR[j]
        L = L * dag(V) * ψL[j] * ALp[j]

        Svec = collect(S.tensor.storage.data)/sum(S)  
 
        S_all[j, 1:length(Svec)] .= Svec  
        #@show sum(Dvec)

    end

    ψR_out[1] = R
    ψL_out[1] = L

    return ψL_out, ψR_out, S_all 
end


ss = siteinds("S=1/2", 12)
ss2 = siteinds("S=1/2", 16)
ss2[1:length(ss)] = ss
ss
ss2
ψL = random_mps(ComplexF64, ss, linkdims=128) 
ψR = random_mps(ComplexF64, ss, linkdims=100) 

AL = random_mpo(ss2) + im*random_mpo(ss2)
AR = random_mpo(ss2) + im*random_mpo(ss2)

for kk = length(ss)+1:length(ss2)
    AR[kk] *= ITensor([1,0],ss2[kk])
    AL[kk] *= ITensor([1,0],ss2[kk]')
end

LO = applyns(AL, ψL; truncate=false)
OR = applyn(AR, ψR; truncate=false)

cutoff = 1e-20
maxdim=128

llt, rrt, sst = ITransverse.truncate_sweep(LO, OR; cutoff, maxdim, direction=:right)
llt_left, rrt_left, sst_left = ITransverse.truncate_sweep(LO, OR; cutoff, maxdim, direction=:left)

llt2, rrt2, sst2 = ITransverse.truncate_rsweep_rtm(LO, OR; cutoff, maxdim)

ll, rr, ss = tlrcontract_old(ψL, AL, AR, ψR; cutoff, maxdim)
 abs( gen_fidelity(llt,rrt) - gen_fidelity(ll,rr))/abs(gen_fidelity(llt,rrt))
#@test abs( gen_fidelity(llt,rrt) - gen_fidelity(ll,rr)) < 1e-5

llc, rrc, ssc = tlrapply(ψL, AL, AR, ψR; cutoff, maxdim)

#@code_warntype  tlrapply(ψL, AL, AR, ψR; cutoff, maxdim)

@test fidelity(ll, llc) > 0.9999
@test fidelity(ll, llt) > 0.9999
@test fidelity(llt, llt2) > 0.9999
@test fidelity(llc, llt2) > 0.95
@test fidelity(rrc, rrt2) > 0.95

@test fidelity(llc, llt) ≈ 1 
@test fidelity(rrc, rrt) ≈ 1 

fidelity(rr, rrc)
fidelity(rr, rrt)


ss = siteinds("S=1/2", 40)

ψL = random_mps(ComplexF64, ss, linkdims=100) 
ψR = random_mps(ComplexF64, ss, linkdims=120) 

AL = random_mpo(ss) + im*random_mpo(ss)
AR = random_mpo(ss) + im*random_mpo(ss)


cutoff = 1e-20
maxdim=256
direction = :right 
truncp = (; cutoff, maxdim, direction)
left, right, s = ITransverse.tlrapply(ITensors.Algorithm("RTM"), ψL, AL, AR, ψR; truncp...)
leftll, rightrr, s = ITransverse.tlrapply(ITensors.Algorithm("RTM"), ψL, AL, AR, ψR; merge(truncp, (;direction=:left))...)

leftn, rightn, sn = ITransverse.tlrapply(ITensors.Algorithm("naiveRTM"), ψL, AL, AR, ψR; direction=:left)
leftref, rightref, sref = ITransverse.tlrapply(ITensors.Algorithm("naiveRTM"), ψL, AL, AR, ψR; direction=:left)

@show fidelity(left,leftn) 
@test fidelity(left,leftn) > 0.99
@show fidelity(right, rightn) #> 0.99

@test (abs(gen_fidelity(left, right) - gen_fidelity(leftn, rightn)))/ abs(gen_fidelity(left, right)) < 0.1
@test gen_fidelity(leftref, rightref) ≈ gen_fidelity(leftn, rightn) 


cutoff = 1e-20
maxdim=256
truncp = (; cutoff, maxdim, direction)
left, right, s = ITransverse.trapply(ITensors.Algorithm("RTM"), ψL, AR, ψR; truncp...)
leftn, rightn, sn = ITransverse.trapply(ITensors.Algorithm("naiveRTM"), ψL, AR, ψR; direction=:right)
leftref, rightref, sref = ITransverse.trapply(ITensors.Algorithm("densitymatrix"), ψL, AR, ψR; direction=:left)

@test siteinds(right) == siteinds(rightn)
@test siteinds(right) == siteinds(rightref)
@test siteinds(leftref) == siteinds(rightref)