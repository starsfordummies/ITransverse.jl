"""
    Generalized two-site DMRG 
"""
function dmrg_gen(H::MPO, psi0::gMPS, sweeps::Sweeps; kwargs...)
  check_hascommoninds(siteinds, H, psi0)
  check_hascommoninds(siteinds, H, psi0')
  # Permute the indices to have a better memory layout
  # and minimize permutations
  H = permute(H, (linkind, siteinds, linkind))
  PH = ProjMPO(H)
  return dmrg_gen(PH, psi0, sweeps; kwargs...)
end



function dmrg_gen(
  PH,
  psi0::gMPS,
  sweeps::Sweeps;
  which_decomp=nothing,
  #svd_alg=nothing,
  observer=NoObserver(),
  outputlevel=1,
  write_when_maxdim_exceeds=nothing,
  write_path=tempdir(),
  # eigsolve kwargs
  eigsolve_tol=1e-14,
  eigsolve_krylovdim=3,
  eigsolve_maxiter=1,
  eigsolve_verbosity=0,
  eigsolve_which_eigenvalue=:SR,
  ishermitian=false,
  normalize=false,
)
  if length(psi0) == 1
    error(
      "`dmrg` currently does not support system sizes of 1. You can diagonalize the MPO tensor directly with tools like `LinearAlgebra.eigen`, `KrylovKit.eigsolve`, etc.",
    )
  end

  @debug_check begin
    # Debug level checks
    # Enable with ITensors.enable_debug_checks()
    checkflux(psi0)
    checkflux(PH)
  end

  psi = copy(psi0)

  psi =  orthogonalize!(PH, psi, 1)
  N = length(psi)
  if !isgenortho(psi) || genorthocenter(psi) != 1
    @info "Bring to gen left-ortho form"
    psi = orthogonalize_gen!(PH, psi, 1, normalize=normalize, method="gen_one") # ! or: which_method
  end
  @assert isgenortho(psi) && genorthocenter(psi) == 1
  check_gen_ortho(psi)

  # if !isnothing(write_when_maxdim_exceeds)
  #   if (maxlinkdim(psi) > write_when_maxdim_exceeds) ||
  #     (maxdim(sweeps, 1) > write_when_maxdim_exceeds)
  #     PH = disk(PH; path=write_path)
  #   end
  # end
  PH = position_gen!(PH, psi, 1)
  energy = 0.0

  @info "Start sweeping"

  for sw in 1:nsweep(sweeps)
    sw_time = @elapsed begin
      maxtruncerr = 0.0

      if !isnothing(write_when_maxdim_exceeds) &&
        maxdim(sweeps, sw) > write_when_maxdim_exceeds
        if outputlevel >= 2
          println(
            "\nWriting environment tensors do disk (write_when_maxdim_exceeds = $write_when_maxdim_exceeds and maxdim(sweeps, sw) = $(maxdim(sweeps, sw))).\nFiles located at path=$write_path\n",
          )
        end
        PH = disk(PH; path=write_path)
      end

      for (b, ha) in sweepnext(N)
        @debug_check begin
          checkflux(psi)
          checkflux(PH)
        end

        #@show b, ha

        @timeit_debug timer "dmrg: position!" begin
          PH = position_gen!(PH, psi, b)
        end

        @debug_check begin
          checkflux(psi)
          checkflux(PH)
        end

        @timeit_debug timer "dmrg: psi[b]*psi[b+1]" begin
          phi = psi[b] * psi[b + 1]
        end

        #@show inds(phi)

        # The original: KrylovKit.eigsolve
        @timeit_debug timer "dmrg: eigsolve" begin
          vals, vecs = eigsolve(
            PH,
            phi,
            1,
            eigsolve_which_eigenvalue;
            ishermitian,
            tol=eigsolve_tol,
            krylovdim=eigsolve_krylovdim,
            maxiter=eigsolve_maxiter,
            verbosity=eigsolve_verbosity
          )
        end

        # Much slower but hey - does it help ? 
        FF =  eigsolve_ortho(PH)

        energy = vals[1]
        energy2 = FF.D[1]

        eigenvec1 = FF.Vt * onehot(inds(FF.Vt, "eigen")[1] => 1)

        if abs(energy - energy2 ) > 1e-5
          @warn "en(Lanczos) = $(energy) != $(energy2) = en(Ortho)"
        end


        ## Right now there is a conversion problem in CUDA.jl where `UnifiedMemory` Arrays are being converted 
        ## into `DeviceMemory`. This conversion line is here temporarily to fix that problem when it arises
        ## Adapt is only called when using CUDA backend. CPU will work as implemented previously.
        phi::ITensor = if NDTensors.iscu(phi) && NDTensors.iscu(vecs[1])
          adapt(set_eltype(unwrap_type(phi), eltype(vecs[1])), vecs[1])
        else
          vecs[1]
        end

        # with ortho forms 
        phi = eigenvec1

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
            #svd_alg,
          )
        end
        maxtruncerr = max(maxtruncerr, spec.truncerr)

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
          phi,
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


function dmrg_gen(
  x1,
  psi0::gMPS;
  nsweeps,
  maxdim=default_maxdim(),
  mindim=default_mindim(),
  cutoff=default_cutoff(),
  noise=default_noise(),
  kwargs...,
 )
  return dmrg_gen(x1, psi0, _dmrg_sweeps(; nsweeps, maxdim, mindim, cutoff, noise); kwargs...)
end
