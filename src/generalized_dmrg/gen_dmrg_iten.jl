using ITensors
using TimerOutputs, Printf
import ITensors: _dmrg_sweeps, check_hascommoninds
using KrylovKit: eigsolve

include("gen_projectmpo.jl")
include("gen_replacebond_2.jl")
#include("_gen_replacebond.jl")


default_maxdim() = typemax(Int)
default_mindim() = 1
default_cutoff() = 1e-8
default_noise() = false


function dmrg_gen(H::MPO, psi0::MPS, sweeps::Sweeps; kwargs...)
        
    check_hascommoninds(siteinds, H, psi0)
    check_hascommoninds(siteinds, H, psi0')


    # Permute the indices to have a better memory layout
    # and minimize permutations
    H = permute(H, (linkind, siteinds, linkind))
    PH = ProjMPO(H)

    return dmrg_gen(PH, psi0, sweeps; kwargs...)
end


"""
the fat of the "generalized DMRG" algo
"""
function dmrg_gen(
PH,
psi0::MPS,
sweeps::Sweeps;
which_decomp=nothing,
svd_alg=nothing,
observer=NoObserver(),
outputlevel=1,
# eigsolve kwargs
eigsolve_tol=1e-14,
eigsolve_krylovdim=3,
eigsolve_maxiter=1,
eigsolve_verbosity=0,
eigsolve_which_eigenvalue=:LM,  # [LR: largest real part [or LM: largest magnitude?]
ishermitian=false,
)

if length(psi0) == 1
    error(
    "`dmrg` currently does not support system sizes of 1. You can diagonalize the MPO tensor directly with tools like `LinearAlgebra.eigen`, `KrylovKit.eigsolve`, etc.",
    )
end

#psi = copy(psi0)
N = length(psi0)


# ORTHOGONALIZE [ GEN ]
println("Generalized DMRG - bringing first to generalized RIGHT canonical form")
psi, _ = gen_canonical_right(psi0)
setleftlim!(psi, 0)
setrightlim!(psi,2)

check_gencan_right_sym(psi)

#println(siteinds(PH.H))
#println(siteinds(psi))
PH = position_gen!(PH, psi, 1)

energy = 0.0

for sw in 1:nsweep(sweeps)
    sw_time = @elapsed begin
        maxtruncerr = 0.0

        for (b, ha) in sweepnext(N)

            @timeit_debug timer "dmrg: position!" begin
                PH = position_gen!(PH, psi, b)
            end

            @timeit_debug timer "dmrg: psi[b]*psi[b+1]" begin
                phi = psi[b] * psi[b + 1]
            end

            #@show(eigsolve_which_eigenvalue, ishermitian, eigsolve_krylovdim, eigsolve_maxiter)
            @timeit_debug timer "dmrg: eigsolve" begin
                vals, vecs, infoKrylov = eigsolve(
                    PH,
                    phi,
                    1,
                    eigsolve_which_eigenvalue;
                    ishermitian,
                    tol=eigsolve_tol,
                    krylovdim=eigsolve_krylovdim,
                    maxiter=eigsolve_maxiter,
                )
            end
            #@show infoKrylov

            energy = vals[1]
            phi::ITensor = vecs[1]
        
            ortho = ha == 1 ? "left" : "right"

            drho = nothing
        

            @timeit_debug timer "dmrg: replacebond!" begin
                spec = replacebond_gen!(
                    PH,
                    psi,
                    b,
                    phi;
                    maxdim=maxdim(sweeps, sw),
                    mindim=mindim(sweeps, sw),
                    cutoff=cutoff(sweeps, sw),
                    eigen_perturbation=drho,
                    ortho,
                    normalize=true,
                    which_decomp,
                    svd_alg,
                )
            end

            # now we updated b,b+1, so it should be 
            # Left until b , Right from b+2  if Left sweep
            # Left until b-1, Right from b+1 if Right sweep

            # TODO debug: is this useful?
            # try
            #     check_gencan_mixed_sym(psi, b, ha)
            # catch e
            #     @infiltrate
            #     rethrow(e)
            # end
            

            # TODO fix spec ..
            maxtruncerr = max(maxtruncerr, spec.truncerr)

            if outputlevel >= 2
                @printf("Sweep %d, half %d, bond (%d,%d) energy=%s\n", sw, ha, b, b + 1, energy)
                @printf(
                    "  Truncated using cutoff=%.1E maxdim=%d mindim=%d\n",
                    cutoff(sweeps, sw),
                    maxdim(sweeps, sw),
                    mindim(sweeps, sw)
                )
                # TODO fix spec 
                # @printf(
                #     "  Trunc. err=%.2E, bond dimension %d\n", spec.truncerr, dim(linkind(psi, b))
                # )
                flush(stdout)
            end

            #half_sweep 
            if (b == length(psi)-1 && ha == 1)
                # this maybe becomes unnecessary if we normalize along the way
                #psi[end] *= 1/sqrt(overlap_noconj(psi,psi))
                check_gencan_left_sym(psi)
            end

            sweep_is_done = (b == 1 && ha == 2)
            if sweep_is_done
                # TODO normalize 1st site ? unnecessary if we normalize along the way ?
                #psi[1] *= 1/sqrt(overlap_noconj(psi,psi))
                check_gencan_right_sym(psi)
                
            end


            measure!(
            observer;
            energy,
            psi,
            projected_operator=PH,
            bond=b,
            sweep=sw,
            half_sweep=ha,
            spec,
            outputlevel,
            sweep_is_done,
            )
            
        end # sweepnext

    end # sweep_time

    if outputlevel >= 1
    @printf(
        "[gen]After sweep %d energy=%s  maxlinkdim=%d maxerr=%.2E time=%.3f\n",
        sw,
        energy,
        maxlinkdim(psi),
        maxtruncerr,
        sw_time
    )
    flush(stdout)
    end
    
    isdone = checkdone!(observer; energy, psi, sweep=sw, outputlevel)
    isdone && break

end # nsweeps

return (energy, psi)

end





function dmrg_gen(
    x1::MPO,
    psi0::MPS;
    nsweeps::Int64,
    maxdim=default_maxdim(),
    mindim=default_mindim(),
    cutoff=default_cutoff(),
    noise=default_noise(),
    kwargs...,
    )

    return dmrg_gen(x1, psi0, _dmrg_sweeps(; nsweeps, maxdim, mindim, cutoff, noise); kwargs...)
end