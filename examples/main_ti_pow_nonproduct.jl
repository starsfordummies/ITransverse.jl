###############################################################################
# Translation-invariant transverse problem with non-product boundaries + power method
#
# Same amplitude as `main_finite_nonproduct.jl`,
#
#   A(L) = <phi_f| U(dt)^Nt |psi_0>,
#
# but now |psi_0> and <phi_f| are *uniform* (translation-invariant) MPS, given by a single
# bulk tensor of bond dimension 5 and 7 respectively.  Then every column of the transverse
# network is the same tMPO T, and the L -> infinity limit is set by its dominant eigenvalue:
#
#   A(L) ~ <l| T^(L-2) |r>   ->   A(L+1)/A(L) -> lambda,   T|r> = lambda |r>,  <l|T = lambda <l|
#
# `powermethod_lr` gets |r> and <l| (non-symmetric: no conjugation anywhere).  The extra
# boundary sites (dim 5 at the bottom, dim 7 at the top) ride along through the truncation
# sweeps like any other site.
#
# Check: the power-method eigenvalue against the ratio of finite-L amplitudes.
###############################################################################

using ITensors, ITensorMPS
using ITransverse
using LinearAlgebra
using Random

Random.seed!(1234)

"""
Bulk tensor of a translation-invariant MPS: `identity x v` (a product state written with bond
dimension `dim(il)`) plus `eps` of noise, so the state is correlated but still gapped.
"""
function perturbed_product_tensor(sphys::Index, il::Index, ir::Index, v, eps::Real)
    A = ITensor(ComplexF64, sphys, il, ir)
    for a in 1:dim(il), sigma in 1:dim(sphys)
        A[sphys => sigma, il => a, ir => a] = v[sigma]
    end
    A += eps * random_itensor(ComplexF64, sphys, il, ir)
    return A / norm(A)
end


function main_ti_pow_nonproduct(; Nt=12, dt=0.1, chi0=5, chif=7, eps=0.2,
                                  maxdim=128, cutoff=1e-12, itermax=600)

    mp = IsingParams(1.0, 0.7, 0.0)
    sphys = addtags(sim(mp.phys_site), "Site")

    # bulk tensor of the initial TI MPS (bond dimension 5) and of the final one (bond
    # dimension 7).  A *purely* random bulk tensor gives a transverse transfer operator with
    # no gap to speak of - the power method then wanders and the finite-L ratios oscillate.
    # Take instead a correlated state: product state + noise, which keeps a clean dominant
    # eigenvalue while still using the full bond dimension.
    il, ir = Index(chi0, "il"), Index(chi0, "ir")
    A = perturbed_product_tensor(sphys, il, ir, ComplexF64[1, 0], eps)
    bl = boundary_tensor(A; phys=sphys, left=il, right=ir)

    jl, jr = Index(chif, "jl"), Index(chif, "jr")
    B = perturbed_product_tensor(sphys, jl, jr, ComplexF64[1, 1] / sqrt(2), eps)
    tr = boundary_tensor(B; phys=sphys, left=jl, right=jr)

    tp = tMPOParams(mp; dt, scheme=Murg(), nbeta=0, init_state=bl)
    b = FwtMPOBlocks(tp)
    time_sites = [Index(2, "Site,time,t=$i") for i in 1:Nt]

    # one bulk column: Nt time sites + one boundary site at each end
    T = fw_tMPO(b, time_sites; bl, tr)
    @assert length(T) == Nt + 2

    # any edge closure works as a power-method seed; take random ones
    v0, vf = randn(ComplexF64, chi0), randn(ComplexF64, chif)
    seed = fw_tMPS(b, time_sites; LR=:right,
                   bl=close_boundary(bl, v0; side=:right),
                   tr=close_boundary(tr, vf; side=:right))
    @assert length(seed) == Nt + 2
    @info "column" nsites=length(T) site_dims=dim.(siteinds(seed))

    truncp = (; cutoff, maxdim, alg="naiveRTM")
    pm_params = PMParams(; truncp, itermax, eps_converged=1e-10,
                           opt_method=:nosym, normalization="norm")

    # `powermethod_lr` is not exported yet, hence the qualified call
    ll, rr, info = ITransverse.powermethod_lr(seed, T, T, pm_params)

lambda = overlap_noconj(ll, apply_column(T, rr)) / overlap_noconj(ll, rr)
    @info "power method" lambda chi_l=maxlinkdim(ll) chi_r=maxlinkdim(rr) nsites=length(rr)

    ## ------------------------------------------------ finite-L cross-check
    # A(L) with genuine edge columns of the same TI boundary MPS
    blL = close_boundary(bl, v0; side=:left);  blR = close_boundary(bl, v0; side=:right)
    trL = close_boundary(tr, vf; side=:left);  trR = close_boundary(tr, vf; side=:right)
    lle = fw_tMPS(b, time_sites; LR=:left,  bl=blL, tr=trL)
    rre = fw_tMPS(b, time_sites; LR=:right, bl=blR, tr=trR)

    Ls = 6:2:26
    amps = ComplexF64[]
    psi = rre
    Lprev = 2
    for L in Ls
        for _ in 1:(L - Lprev)
            psi, _ = tapply(T, psi; alg="densitymatrix", cutoff, maxdim)
        end
        Lprev = L
        push!(amps, overlap_noconj(lle, psi))
    end
    ratios = [(amps[k+1] / amps[k])^(1 / (Ls[k+1] - Ls[k])) for k in 1:(length(amps)-1)]

    @info "finite-L ratios A(L+1)/A(L) -> lambda" ratios lambda
    @info "last ratio vs lambda" rel_err = abs(last(ratios) - lambda) / abs(lambda)

    return ll, rr, lambda, ratios, info
end

ll, rr, lambda, ratios, info = main_ti_pow_nonproduct()

# Note: `compute_expvals` / `expval_LR` do *not* work on these vectors, and say so.  They
# rebuild the operator column over the time sites and close its top with `fold_op` = the
# operator, so there is no room left for a non-product final state on top.  Build such a
# network by hand instead: `T_op = fw_tMPO(b, ts; bl, tr)` with the operator inserted in the
# bulk, then `overlap_noconj(ll, applyn(T_op, rr))`.  Building the vectors with `sided=true`
# is what lets the helpers detect the situation (see `TransverseMPS`, `n_boundary_top`).
