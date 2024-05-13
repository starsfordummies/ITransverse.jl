
function dmrg_gen_onesite_slow(H::MPO, psi0::MPS, sweeps::Sweeps; kwargs...)
        
    check_hascommoninds(siteinds, H, psi0)
    check_hascommoninds(siteinds, H, psi0')


    # Permute the indices to have a better memory layout
    # and minimize permutations
    H = permute(H, (linkind, siteinds, linkind))
    # ProjMPO(H::MPO) = ProjMPO(0, length(H) + 1, 2, H, Vector{ITensor}(undef, length(H)))
    PH = ProjMPO(0, length(H)+1, 1, H, Vector{ITensor}(undef, length(H)))

    return dmrg_gen_onesite_slow(PH, psi0, sweeps; kwargs...)
end


"""
the fat of the "generalized DMRG" algo
"""
function dmrg_gen_onesite_slow(
    PH,
    psi0::MPS,
    sweeps::Sweeps;
    which_decomp=nothing,
    #svd_alg=nothing,
    observer=NoObserver(),
    outputlevel=1,
    normalize=false,
    # eigsolve kwargs
    eigsolve_tol=1e-14,
    eigsolve_krylovdim=3,
    eigsolve_maxiter=1,
    eigsolve_verbosity=0,
    eigsolve_which_eigenvalue=:SR,  # [LR: largest real part [or LM: largest magnitude?]
    ishermitian=false,
    )

    if length(psi0) == 1
        error(
        "`dmrg` currently does not support system sizes of 1. You can diagonalize the MPO tensor directly with tools like `LinearAlgebra.eigen`, `KrylovKit.eigsolve`, etc.",
        )
    end

    psi = copy(psi0)
    N = length(psi)


    @show norm_gen(psi)

    # GEN-ORTHOGONALIZE 
    psi = orthogonalize!(PH, psi, length(psi))
    if !isgenortho(psi) || genorthocenter(psi) != 1
        psi = orthogonalize_gen!(PH, psi, 1, normalize=normalize, method=which_decomp)
    end

    @show norm_gen(psi)

    @assert isgenortho(psi) && genorthocenter(psi) == 1


    #println(siteinds(PH.H))
    #println(siteinds(psi))
    PH = position_gen!(PH, psi, 1)


    energy = 0.0

    phi_prev = ITensor()

    for sw in 1:nsweep(sweeps)
        sw_time = @elapsed begin
            maxtruncerr = 0.0

            for (b, ha) in sweepnext(N, ncenter=1)
                #println("$b  $ha")
            
            
                #println("updating $b/$(length(psi))")

                # skip last to the right, first to the left 
                if b == N && ha == 1
                    continue
                elseif b == 1 && ha == 2
                    # end of sweep - re-canonicalize everything ? 
                    orthogonalize_gen!(PH, psi, length(psi), normalize=normalize, method=which_decomp)
                    orthogonalize_gen!(PH, psi, 1, normalize=normalize, method=which_decomp)
                    #hack to recompute full envs 
                    PH.rpos = length(psi)+1
                    PH = position_gen!(PH, psi, 1)
                    continue
                end


                @timeit_debug timer "dmrg: position!" begin
                    PH = position_gen_wprev!(PH, psi, b, phi_prev, 0.0001)
                end


                @timeit_debug timer "dmrg: psi[b]" begin
                    phi = psi[b] 
                end

                phi_prev = copy(phi)


                #println(inds(phi))
                #println(inds(lproj(PH)))
                #println(inds(rproj(PH)))


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
                ## Right now there is a conversion problem in CUDA.jl where `UnifiedMemory` Arrays are being converted 
                ## into `DeviceMemory`. This conversion line is here temporarily to fix that problem when it arises
                ## Adapt is only called when using CUDA backend. CPU will work as implemented previously.
                phi::ITensor = if NDTensors.iscu(phi) && NDTensors.iscu(vecs[1])
                    adapt(set_eltype(unwrap_type(phi), eltype(vecs[1])), vecs[1])
                else
                    vecs[1]
                end
                #phi::ITensor = vecs[1]
        
                ortho = ha == 1 ? "left" : "right"
        
                drho = nothing
                if noise(sweeps, sw) > 0
                    @timeit_debug timer "dmrg: noiseterm" begin
                    # Use noise term when determining new MPS basis.
                    # This is used to preserve the element type of the MPS.
                    elt = real(scalartype(psi))
                    drho = elt(noise(sweeps, sw)) * noiseterm(PH, phi, ortho)
                    end
                end
        
                @debug_check begin
                    checkflux(phi)
                end

                spec=nothing
                #@show inds(phi)
                psi[b] = phi
                if ortho == "left"
                    orthogonalize_gen!(psi, b+1, method=which_decomp, normalize=normalize)    
                    #normalize &&  (psi[b+1] /= norm_gen(psi[b+1]))
                elseif ortho == "right"
                    orthogonalize_gen!(psi, b-1, method=which_decomp, normalize=normalize)
                    #normalize &&  (psi[b-1] /= norm_gen(psi[b-1]))
                    # for debugging
                    #if normalize
                    #    println("normalizing from $(norm_gen(psi))")
                    #    (psi[b-1] /= norm_gen(psi[b-1]))
                    #    println("normalizing to $(norm_gen(psi))")
                    #end
                    
                end
                @show b, norm_gen(psi)
                maxtruncerr = 0.

                # @timeit_debug timer "dmrg: replacebond!" begin
                #     spec = gen_replacebond_onesite!(
                #     psi,
                #     b,
                #     phi;
                #     maxdim=maxdim(sweeps, sw),
                #     mindim=mindim(sweeps, sw),
                #     cutoff=cutoff(sweeps, sw),
                #     eigen_perturbation=drho,
                #     ortho,
                #     normalize=true,
                #     which_decomp="gen_one",
                #     svd_alg,
                #     )
                # end
                #maxtruncerr = max(maxtruncerr, spec.truncerr)

                @debug_check begin
                    checkflux(psi)
                    checkflux(PH)
                end
        
                if outputlevel >= 2
                    @printf("Sweep %d, half %d, bond (%d,%d) energy=%s\n", sw, ha, b, b + 1, energy)
                    @printf(
                    "  Truncated using cutoff=%.1E maxdim=%d mindim=%d\n",
                    cutoff(sweeps, sw),
                    maxdim(sweeps, sw),
                    mindim(sweeps, sw)
                    )
                    @printf(
                    "  Trunc. err=%.2E, bond dimension %d\n", spec.truncerr, dim(linkind(psi, b))
                    )
                    flush(stdout)
                end
        
                sweep_is_done = (b == 1 && ha == 2)
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
                end

            end

            if outputlevel >= 1
                @printf(
                "After sweep %d energy=%s  maxlinkdim=%d maxerr=%.2E time=%.3f\n",
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
        end
    return (energy, psi)
end
    


function dmrg_gen_onesite_slow(
    x1::MPO,
    psi0::MPS;
    nsweeps::Int64,
    maxdim=default_maxdim(),
    mindim=default_mindim(),
    cutoff=default_cutoff(),
    noise=default_noise(),
    kwargs...,
    )

    return dmrg_gen_onesite_slow(x1, psi0, _dmrg_sweeps(; nsweeps, maxdim, mindim, cutoff, noise); kwargs...)
end
