using ITensors
using KrylovKit: eigsolve
using LinearAlgebra
using Plots
#using MKL

include("fuse_inds.jl")

using ITransverse

ITensors.Strided.disable_threads()
ITensors.disable_threaded_blocksparse()


function main_tmpo(n; blas_num_threads=Sys.CPU_THREADS, fuse=true, binary=true)
 
  
    BLAS.set_num_threads(blas_num_threads)
  

    JXX = 1.0  
    hz = 1.2
    dt = 0.1

    nbeta = 2

    plus_state = Vector{ComplexF64}([1/sqrt(2),1/sqrt(2)])

    init_state = plus_state



    SVD_cutoff = 1e-10
    maxbondim = 100
    itermax = 200
    verbose=false
    ds2_converged = 1e-4

    pm_params = ppm_params(itermax, SVD_cutoff, maxbondim, verbose, ds2_converged)



    Tstart= n-4

    Ntime_steps = Tstart
    Nsteps = Ntime_steps +2*nbeta
    time_sites =  addtags(siteinds("S=1/2", Nsteps; conserve_qns = false), "time")

    s = time_sites

    ψ0 = productMPS(time_sites,"+");

    H = build_ising_tMPO_regul_beta(build_expH_ising_murg, JXX, hz, dt, nbeta, time_sites, init_state)

    println("Length H = $(length(H))")


    edmrg, ψdmrg = dmrg(H, ψ0; nsweeps=30, eigsolve_which_eigenvalue=:LR, ishermitian=false)

    edmrg_alt1 = inner(ψdmrg',H,ψdmrg)
    #edmrg_alt2 = inner(dag(ψdmrg)',H,ψdmrg)
  
    ψpm, ds2s  = powermethod_sym(ψ0, H, pm_params)

    epm = inner(dag(ψpm)',H,ψpm)

    if n > 12
      @warn "System size of $n is likely too large for exact diagonalization."
      vals, vecs, info = [0], [0], 0
    else

      if fuse
        if binary
          println("Fuse the indices using a binary tree")
          T = fusion_tree_binary(s)
          H_full = @time fuse_inds_binary(H, T)
          ψ0_full = @time fuse_inds_binary(ψ0, T)
        else
          println("Fuse the indices using an unbalances tree")
          T = fusion_tree(s)
          H_full = @time fuse_inds(H, T)
          ψ0_full = @time fuse_inds(ψ0, T)
        end
      else
        println("Don't fuse the indices")
        @disable_warn_order begin
          H_full = @time contract(H)
          ψ0_full = @time contract(ψ0)
        end
      end
    
      vals, vecs, info = @time eigsolve(
        H_full, ψ0_full, 1, :LR; ishermitian=false, tol=1e-6, krylovdim=30, eager=true
      )
    
    end
    @show (edmrg)
    @show (edmrg_alt1)
 
    @show epm
    #@show epm_alt

    @show vals[1]

    
    return edmrg, edmrg_alt1, epm, vals[1]

  end
  
  # list = []
  # for nn = 6:1:14
  # push!(list, main_tmpo(nn))
  # end


  main_tmpo(60)