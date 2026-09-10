"""
Benchmark: `truncate_sweep` ("naiveRTM", gauge-transformation sweep) vs
`truncate_sweep_rtm` ("RTM", explicit reduced transfer matrix + SVD)

For random complex MPS pairs (psi, phi) we truncate both members down to
`maxdim` and measure how well the generalized overlap <psi|phi> (no conjugation)
is preserved:

    err = |ov_after - ov_before| / |ov_before|

Averages are taken over `nsamples` random pairs, for a few bond dimensions and
both sweep directions. Wall times are reported as well.

Two ensembles are used:
  * :random     -> psi, phi independently random (overlap is essentially noise)
  * :correlated -> psi, phi share a common random component, so the RTM has a
                   decaying spectrum, which is the regime the algorithms are
                   actually used in.

Run with:  julia --project=. examples/bench_rtm_vs_naive.jl
"""

using ITensors, ITensorMPS
using ITransverse
using LinearAlgebra
using Printf
using Random
using Statistics

# ---------------------------------------------------------------- sample gen

"""Random complex MPS pair. `ensemble = :random` or `:correlated`."""
function sample_pair(sites, chi::Int, ensemble::Symbol)
    if ensemble == :random
        psi = random_mps(ComplexF64, sites; linkdims=chi)
        phi = random_mps(ComplexF64, sites; linkdims=chi)
    elseif ensemble == :correlated
        # common component + independent noise -> non-flat RTM spectrum
        base = random_mps(ComplexF64, sites; linkdims=chi ÷ 2)
        n1 = random_mps(ComplexF64, sites; linkdims=chi ÷ 2)
        n2 = random_mps(ComplexF64, sites; linkdims=chi ÷ 2)
        psi = add(base, 0.5 * n1; maxdim=chi, cutoff=0.0)
        phi = add(base, 0.5 * n2; maxdim=chi, cutoff=0.0)
    else
        error("unknown ensemble $ensemble")
    end
    psi, phi = normalize(psi), normalize(phi)
    # complex data throughout: `add`/`normalize` must not have dropped the imaginary part
    @assert all(t -> eltype(t) <: Complex, psi) && all(t -> eltype(t) <: Complex, phi)
    @assert maximum(t -> maximum(abs, imag(Array(t, inds(t)))), psi) > 0
    return psi, phi
end

# ------------------------------------------------------------------- runners

"""Relative overlap error + elapsed time for the naiveRTM sweep."""
function run_naive(psi, phi, ov_before; cutoff, maxdim, direction)
    t = @elapsed res = truncate_sweep(copy(psi), copy(phi); cutoff, maxdim, direction)
    ov_after = overlap_noconj(res.L, res.R)
    return abs(ov_after - ov_before) / abs(ov_before), t, maxlinkdim(res.L)
end

"""Relative overlap error + elapsed time for the explicit-RTM sweep."""
function run_rtm(psi, phi, ov_before; cutoff, maxdim, direction)
    t = @elapsed L, R, _ = truncate_sweep_rtm(psi, phi; cutoff, maxdim, direction)
    ov_after = overlap_noconj(L, R)
    return abs(ov_after - ov_before) / abs(ov_before), t, maxlinkdim(L)
end

# --------------------------------------------------------------------- bench

function benchmark(;
        nsites::Int = 20,
        chi::Int = 32,
        maxdims = (4, 8, 16),
        directions = (:left, :right),
        ensembles = (:correlated, :random),
        nsamples::Int = 10,
        cutoff::Float64 = 1e-14,
        seed::Int = 1234,
    )

    Random.seed!(seed)
    sites = siteinds("S=1/2", nsites)

    @printf("N=%d, chi_in=%d, %d samples/point, cutoff=%.0e\n\n", nsites, chi, nsamples, cutoff)
    @printf("%-12s %-6s %6s | %-22s | %-22s | %s\n",
            "ensemble", "dir", "maxdim",
            "naiveRTM err (med/mean)", "RTM err (med/mean)", "t_naive/t_RTM [ms]")
    println("-"^105)

    wins = Dict(:naive => 0, :rtm => 0, :tie => 0)

    for ensemble in ensembles, direction in directions, maxdim in maxdims
        errs_n, errs_r, ts_n, ts_r = Float64[], Float64[], Float64[], Float64[]

        for _ in 1:nsamples
            psi, phi = sample_pair(sites, chi, ensemble)
            ov_before = overlap_noconj(psi, phi)
            abs(ov_before) < 1e-14 && continue   # degenerate sample, skip

            en, tn, _ = run_naive(psi, phi, ov_before; cutoff, maxdim, direction)
            er, tr, _ = run_rtm(psi, phi, ov_before; cutoff, maxdim, direction)

            push!(errs_n, en); push!(ts_n, tn)
            push!(errs_r, er); push!(ts_r, tr)

            if en < er * 0.99
                wins[:naive] += 1
            elseif er < en * 0.99
                wins[:rtm] += 1
            else
                wins[:tie] += 1
            end
        end

        isempty(errs_n) && continue

        @printf("%-12s %-6s %6d | %10.3e %10.3e | %10.3e %10.3e | %7.1f / %7.1f\n",
                ensemble, direction, maxdim,
                median(errs_n), mean(errs_n),
                median(errs_r), mean(errs_r),
                1e3 * median(ts_n), 1e3 * median(ts_r))
    end

    total = sum(values(wins))
    println("\nper-sample winner (>1% relative difference in error):")
    @printf("  naiveRTM: %d/%d   RTM: %d/%d   tie: %d/%d\n",
            wins[:naive], total, wins[:rtm], total, wins[:tie], total)

    return wins
end

# warmup (compilation) then the real run
benchmark(; nsites=6, chi=8, maxdims=(4,), nsamples=1)
println()
benchmark()
