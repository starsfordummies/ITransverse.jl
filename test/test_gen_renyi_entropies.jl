using Test
using Random
using LinearAlgebra
using ITensors, ITensorMPS
using ITransverse

Random.seed!(20260911)

# every eigenvalue in `a` has a partner in `b` (and vice versa), relative to max|λ|
function spectra_match(a, b; rtol = 1e-10)
    sc = max(maximum(abs, a), maximum(abs, b))
    all(minimum(abs.(x .- b)) <= rtol * sc for x in a) &&
        all(minimum(abs.(x .- a)) <= rtol * sc for x in b)
end

# dense RTM spectra of Tr_B |phi><psi| at every cut (small chains only)
function rtm_eigs_bruteforce(psi, phi; cutoff = 1e-10)
    s = siteinds(psi); N = length(psi); d = dim(s[1])
    vpsi = Array(contract(psi), s...)
    vphi = Array(contract(replace_siteinds(sim(linkinds, phi), s)), s...)
    ov = sum(vpsi .* vphi)
    map(1:N-1) do k
        λ = eigvals(reshape(vphi, d^k, :) * transpose(reshape(vpsi, d^k, :))) ./ ov
        filter(x -> abs(x) > cutoff * maximum(abs, λ), λ)
    end
end

function random_gauge(psi)                         # scramble every link with a random GL(χ)
    psi = copy(psi)
    for k in 1:length(psi)-1
        l = linkind(psi, k); l2 = sim(l)
        G = randn(eltype(psi[k]), dim(l), dim(l)) + 3I
        psi[k] = psi[k] * ITensor(G, l, l2)
        psi[k+1] = psi[k+1] * ITensor(inv(G), l2, l)
    end
    return psi
end

@testset "diagonalize_rtm_lr: full spectrum vs brute force" begin
    s = siteinds(4, 7)
    for elt in (ComplexF64, Float64)
        psi = random_mps(elt, s; linkdims = 15)
        phi = random_mps(elt, s; linkdims = 36)
        eigs = diagonalize_rtm_lr(psi, phi)
        ref  = rtm_eigs_bruteforce(psi, phi)
        @test length(eigs) == length(s) - 1
        @test all(length.(eigs) .== length.(ref))
        @test all(spectra_match(e, r) for (e, r) in zip(eigs, ref))
        # normalized by the overlap = Tr τ at every cut
        @test all(isapprox(sum(e), 1; atol = 1e-10) for e in eigs)
        if elt == Float64      # real RTM: spectrum closed under conjugation
            @test all(spectra_match(e, conj.(e)) for e in eigs)
        end
    end
end

@testset "gen_renyi_entropies S2 vs replica contraction (gen_renyi2)" begin
    s = siteinds(4, 40)
    psi = random_mps(ComplexF64, s; linkdims = 15)
    phi = random_mps(ComplexF64, s; linkdims = 36)
    ents = gen_renyi_entropies(psi, phi)
    @test ents.S2 ≈ gen_renyi2(psi, phi) rtol = 1e-10
    # the bigger link at a cut only adds structural zeros: rank ≤ min(χ_ψ, χ_φ)
    @test maximum(length, diagonalize_rtm_lr(psi, phi)) <= 15
end

@testset "gauge invariance" begin
    s = siteinds(4, 12)
    psi = random_mps(ComplexF64, s; linkdims = 10)
    phi = random_mps(ComplexF64, s; linkdims = 14)
    e1 = diagonalize_rtm_lr(psi, phi)
    e2 = diagonalize_rtm_lr(random_gauge(psi), random_gauge(phi))
    @test all(spectra_match(a, b; rtol = 1e-9) for (a, b) in zip(e1, e2))
end

@testset "symmetric case matches the generalized-canonical route" begin
    s = siteinds(4, 16)
    psi = random_mps(ComplexF64, s; linkdims = 20)
    e_lr  = diagonalize_rtm_lr(psi, psi)
    e_sym = diagonalize_rtm_symmetric(psi; direction = :left, gen_can_method = :oeig)
    @test all(spectra_match(a, b; rtol = 1e-8) for (a, b) in zip(e_lr, e_sym))
    @test gen_renyi_entropies(psi, psi).S2 ≈ gensym_renyi_entropies(psi; gen_can_method = :oeig).S2 rtol = 1e-8
    # complex-orthogonal QR is a worse-conditioned gauge: ~3 digits less
    e_qr = diagonalize_rtm_symmetric(psi; direction = :left, gen_can_method = :qr)
    @test all(spectra_match(a, b; rtol = 1e-6) for (a, b) in zip(e_lr, e_qr))
    @test gen_renyi_entropies(psi, psi).S2 ≈ gensym_renyi_entropies(psi; gen_can_method = :qr).S2 rtol = 1e-6
    # rank-deficient bonds (bond 1 has rank 4 < χ): no spurious null eigenvalues, so S0 agrees
    @test gensym_renyi_entropies(psi; gen_can_method = :qr).S0 ≈ gensym_renyi_entropies(psi; gen_can_method = :oeig).S0
end

@testset "real inputs with negative eigenvalues give complex entropies, not a DomainError" begin
    s = siteinds(4, 8)
    psi = random_mps(Float64, s; linkdims = 6)
    phi = random_mps(Float64, s; linkdims = 6)
    eigs = diagonalize_rtm_lr(psi, phi)
    @test any(any(x -> isreal(x) && real(x) < 0, e) for e in eigs)
    ents = gen_renyi_entropies(psi, phi)
    @test eltype(ents.S1) <: Complex
    @test ents.S2 ≈ gen_renyi2(psi, phi) rtol = 1e-10
end

@testset "QN (Z2) states agree with their dense versions" begin
    s = siteinds("S=1/2", 12; conserve_szparity = true)
    st = [isodd(j) ? "Up" : "Dn" for j in 1:12]
    psi = random_mps(ComplexF64, s, st; linkdims = 8)
    phi = random_mps(ComplexF64, s, st; linkdims = 12)
    e_qn = diagonalize_rtm_lr(psi, phi)
    e_d  = diagonalize_rtm_lr(dense(psi), dense(phi))
    @test all(spectra_match(a, b; rtol = 1e-9) for (a, b) in zip(e_qn, e_d))
end

@testset "rtm_eigvals: QN block route == dense route on the same matrix" begin
    # the bond-space matrix M is block diagonal with trivial flux, so `ITensors.eigen`
    # decomposes it sector by sector; it must agree with densifying and calling `eigvals`
    for nb in (2, 4)
        i = Index([QN("N", q) => 8 for q in 1:nb]; dir = ITensors.In)
        M = random_itensor(ComplexF64, i, dag(i)')
        @test hasqns(M)
        λ_block = sort(ITransverse.rtm_eigvals(M, i); by = abs, rev = true)
        λ_dense = sort(eigvals(Array(M, i, dag(i)')); by = abs, rev = true)
        @test λ_block ≈ λ_dense atol = 1e-12 * maximum(abs, λ_dense)
        # and the dense branch of the same helper
        Md = dense(M)
        @test sort(ITransverse.rtm_eigvals(Md, ind(Md, 1)); by = abs, rev = true) ≈ λ_dense atol =
            1e-12 * maximum(abs, λ_dense)
    end
end
