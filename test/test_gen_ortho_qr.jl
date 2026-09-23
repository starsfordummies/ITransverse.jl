using Test
using ITensors, ITensorMPS
using LinearAlgebra
using ITransverse

# ── helpers ───────────────────────────────────────────────────────────────────

function left_ortho_err(M::MPS, i::Int)
    l_ind = (i == 1)         ? nothing : linkind(M, i-1)
    r_ind = (i == length(M)) ? nothing : linkind(M, i)
    s_ind = siteinds(M, i)[1]
    chi_L = isnothing(l_ind) ? 1 : dim(l_ind)
    d     = dim(s_ind)
    chi_R = isnothing(r_ind) ? 1 : dim(r_ind)
    A = if isnothing(l_ind)
        reshape(Array(M[i], s_ind, r_ind), 1, d, chi_R)
    elseif isnothing(r_ind)
        reshape(Array(M[i], l_ind, s_ind), chi_L, d, 1)
    else
        Array(M[i], l_ind, s_ind, r_ind)
    end
    Mat = reshape(A, chi_L * d, chi_R)
    norm(transpose(Mat) * Mat - I)
end

function right_ortho_err(M::MPS, i::Int)
    l_ind = (i == 1)         ? nothing : linkind(M, i-1)
    r_ind = (i == length(M)) ? nothing : linkind(M, i)
    s_ind = siteinds(M, i)[1]
    chi_L = isnothing(l_ind) ? 1 : dim(l_ind)
    d     = dim(s_ind)
    chi_R = isnothing(r_ind) ? 1 : dim(r_ind)
    A = if isnothing(l_ind)
        reshape(Array(M[i], s_ind, r_ind), 1, d, chi_R)
    elseif isnothing(r_ind)
        reshape(Array(M[i], l_ind, s_ind), chi_L, d, 1)
    else
        Array(M[i], l_ind, s_ind, r_ind)
    end
    Mat = reshape(A, chi_L, d * chi_R)
    norm(Mat * transpose(Mat) - I)
end

# ── fixtures ──────────────────────────────────────────────────────────────────

const N_SITES  = 18
const LINKDIM  = 26
const TOL      = 1e-8

function make_normalized_mps()
    ss  = siteinds("S=1/2", N_SITES)
    psi = random_mps(ComplexF64, ss; linkdims=LINKDIM)
    psi / sqrt(overlap_noconj(psi, psi))
end

# ── tests ─────────────────────────────────────────────────────────────────────

@testset "gen_orthogonalize" begin

    @testset "gauge orthogonality — center=$center" for center in [1, 2, N_SITES÷2, N_SITES-1, N_SITES]
        psi0 = make_normalized_mps()
        psi  = gen_orthogonalize(psi0, center)
      
        @test fidelity(psi0, psi) ≈ 1

        for i in 1:center-1
            @test left_ortho_err(psi, i) < TOL
        end
        for i in center+1:N_SITES
            @test right_ortho_err(psi, i) < TOL
        end
    end

    @testset "Renyi-2 matches manual — iA=$iA, iB=$iB" for (iA, iB) in [(10, 15), (11, 15)]
        psi  = make_normalized_mps()
        S2_manual = ITransverse.gen_renyi2_sym_interval_manual(psi, iA, iB)

        psig  = gen_orthogonalize(psi, iA+1)
        psigp = prime(linkinds, psig)
        rhoc  = ITensor(1)
        for kk in iA:iB
            rhoc *= psig[kk]
            rhoc *= psigp[kk]
        end
        F        = ITransverse.symm_oeig(rhoc, (linkind(psig, iA-1), linkind(psig, iB)); cutoff=1e-13)
        S2_diag  = only(renyi_entropies(storage(F.D).data).S2)

        @test S2_manual ≈ S2_diag
    end

    @testset "overlap preserved under gen_orthogonalize" begin
        psi  = make_normalized_mps()
        psig = gen_orthogonalize(psi, N_SITES÷2)
        @test overlap_noconj(psi,  psi)  ≈ overlap_noconj(psig, psig)
    end

    @testset "QN-conserving MPS ($qn)" for qn in (:sz, :szparity)
        N = 10
        ss = qn == :sz ? siteinds("S=1/2", N; conserve_qns=true) :
                         siteinds("S=1/2", N; conserve_szparity=true)
        init = [isodd(n) ? "Up" : "Dn" for n in 1:N]
        psi = random_mps(ComplexF64, ss, init; linkdims=12)
        psi = psi / sqrt(overlap_noconj(psi, psi))
        @test hasqns(psi)

        psig = gen_orthogonalize(psi, N)
        @test hasqns(psig)
        @test overlap_noconj(psig, psig) ≈ overlap_noconj(psi, psi)
        @test fidelity(psi, psig) ≈ 1

        # bilinear left-orthogonality (checked on the dense version of the tensors)
        psigd = MPS([dense(psig[j]) for j in 1:N])
        for i in 1:N-1
            @test left_ortho_err(psigd, i) < TOL
        end

        # RTM spectra match the same state without QNs
        r_qn = diagonalize_rtm_symmetric(psi; sort_by_largest=false)
        # dense copy: same tensors with the QN structure stripped
        psid = MPS([dense(psi[j]) for j in 1:N])
        r_d  = diagonalize_rtm_symmetric(psid; sort_by_largest=false)
        for b in 1:N-1
            a = sort(r_qn[b], by=abs, rev=true); d = sort(r_d[b], by=abs, rev=true)
            k = min(length(a), length(d), 6)
            @test a[1:k] ≈ d[1:k]  atol=1e-9
        end
        @test gensym_renyi_entropies(psi).S2 ≈ gensym_renyi_entropies(psid).S2
        @test gensym_renyi_entropies(psid).S2 ≈ gensym_renyi_entropies(psid; gen_can_method=:oeig).S2
    end
end

