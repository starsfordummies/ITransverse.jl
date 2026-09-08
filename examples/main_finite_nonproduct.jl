###############################################################################
# Finite transverse contraction with non-product initial *and* final states
#
#   amplitude  A = <phi_f| U(dt)^Nt |psi_0>
#
# on L = 10 sites and Nt = 14 Trotter steps, with |psi_0> a random MPS of bond
# dimension 3 and <phi_f| a random MPS of bond dimension 4 (both site-dependent,
# i.e. *not* translation invariant).
#
# The transverse network has one column per lattice site.  A non-product boundary
# state is not a vector any more: it is one *column* of the boundary MPS, so it is
# appended to the temporal chain as an extra site.  With both ends non-product each
# column has Nt + 2 sites (see `boundary_tensor` and src/tmpo/boundary_states.jl).
#
#             <phi_f|      B1--B2--...--BL       <- top    (extra site, dim 4)
#                          |   |        |
#             U^Nt         W---W---...--W        <- Nt time sites
#                          |   |        |
#             |psi_0>      A1--A2--...--AL       <- bottom (extra site, dim 3)
#
#                          ^col 1       ^col L
#
# The result is checked against a direct real-space evolution of the same MPS.
###############################################################################

using ITensors, ITensorMPS
using ITransverse
using LinearAlgebra
using Random

Random.seed!(4321)

"""
Random MPS on sites `ss` with *uniform* bond dimension `chi`, returned both as a plain
`MPS` (for the real-space reference) and as the list of transverse boundary columns.

Uniform bonds are required: every column of the transverse network is contracted with its
neighbours through the boundary bond, so all columns must share one bond index `s`
(`bond_ind=s`), which then plays the role of a *site* index of the temporal chain.
"""
function random_boundary_mps(ss::Vector{<:Index}, chi::Int, tag::String)
    L = length(ss)
    il, ir = Index(chi, "il"), Index(chi, "ir")
    As = [random_itensor(ComplexF64, ss[j], il, ir) for j in 1:L]
    As = [A / norm(A) for A in As]
    vL, vR = randn(ComplexF64, chi), randn(ComplexF64, chi)

    # real-space MPS built out of the very same tensors
    bonds = [Index(chi, "Link,l=$j") for j in 1:(L-1)]
    data = ITensor[replaceind(As[1], ir => bonds[1]) * ITensor(vL, il)]
    for j in 2:(L-1)
        push!(data, replaceinds(As[j], (il, ir), (bonds[j-1], bonds[j])))
    end
    push!(data, replaceind(As[L], il => bonds[L-1]) * ITensor(vR, ir))
    psi = MPS(data)

    # transverse columns: rank-3 in the bulk, rank-2 at the two edges
    s = Index(chi, tag)
    cols = [boundary_tensor(As[j]; phys=ss[j], left=il, right=ir, bond_ind=s) for j in 1:L]
    colL = close_boundary(cols[1], vL; side=:left)
    colR = close_boundary(cols[L], vR; side=:right)
    return (; psi, cols, colL, colR, s)
end


function main_finite_nonproduct(; L=10, Nt=14, dt=0.1, chi0=3, chif=4,
                                  cutoff=1e-13, maxdim=256)

    mp = IsingParams(1.0, 0.7, 0.0)
    ss = [addtags(sim(mp.phys_site), "Site") for _ in 1:L]

    in0 = random_boundary_mps(ss, chi0, "bdry_in")
    fin = random_boundary_mps(ss, chif, "bdry_fin")

    ## ------------------------------------------------ real-space reference
    U = build_Ut(ss, Murg(), mp; dt)
    psi_t = in0.psi
    for _ in 1:Nt
        psi_t = apply(U, psi_t; alg="naive", cutoff=1e-15, maxdim=64)
    end
    # the builders close the top with the *bra* <phi_f| (dagger_tr=true by default)
    ref = inner(fin.psi, psi_t)

    ## ------------------------------------------------ transverse contraction
    # `init_state` in the params only fixes the bottom *default*; each column gets its own
    # `bl`/`tr` below, which is what makes a site-dependent boundary MPS possible.
    tp = tMPOParams(mp; dt, scheme=Murg(), nbeta=0, init_state=in0.cols[1])
    b = FwtMPOBlocks(tp)
    time_sites = [Index(2, "Site,time,t=$i") for i in 1:Nt]

    # edge columns -> tMPS, bulk columns -> tMPO.  `sided=true` records which side each
    # vector is and how many of its sites are boundary sites (see `TransverseMPS`).
    rr = fw_tMPS(b, time_sites; LR=:right, bl=in0.colR, tr=fin.colR)
    ll = fw_tMPS(b, time_sites; LR=:left,  bl=in0.colL, tr=fin.colL)

    @info "transverse column" nsites=length(rr) Nt n_bottom=n_boundary_bottom(rr) n_top=n_boundary_top(rr)
    @assert length(rr) == Nt + 2
    @assert first(siteinds(MPS(rr))) == in0.s
    @assert last(siteinds(MPS(rr)))  == fin.s

    # sweep from the right edge to the left one, column by column
    psi = MPS(rr)
    for j in (L-1):-1:2
        Tj = fw_tMPO(b, time_sites; bl=in0.cols[j], tr=fin.cols[j])
        psi, _ = tapply(Tj, psi; alg="densitymatrix", cutoff, maxdim)
    end
    amp = overlap_noconj(MPS(ll), psi)

    @info "amplitude" transverse=amp real_space=ref rel_err=abs(amp-ref)/abs(ref) chi_max=maxlinkdim(psi)

    return amp, ref
end

amp, ref = main_finite_nonproduct()
