using ITensors
using KrylovKit: eigsolve
using LinearAlgebra
using Plots
using JLD2
#using MKL

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
    ds2_converged = 1e-7


    pm_params = ppm_params(itermax, SVD_cutoff, maxbondim, verbose, ds2_converged, false)



        Tstart= n-4

    Ntime_steps = Tstart
    Nsteps = Ntime_steps +2*nbeta
    time_sites =  addtags(siteinds("S=1/2", Nsteps; conserve_qns = false), "time")

    s = time_sites

    ψ0 = productMPS(time_sites,"+");

    H = build_ising_tMPO_regul_beta(build_expH_ising_murg, JXX, hz, dt, nbeta, time_sites, init_state)

    println("Length H = $(length(H))")

    egendmrg, ψgendmrg = dmrg_gen(H, ψ0; nsweeps=20,
    eigsolve_which_eigenvalue=:LM, ishermitian=false, eigsolve_maxiter=4, eigsolve_krylovdim=5)


    #egendmrg_alt1 = inner(ψgendmrg,H,ψgendmrg)/inner(ψgendmrg,ψgendmrg)
    egendmrg_ev = inner(dag(ψgendmrg)',H,ψgendmrg)/inner(dag(ψgendmrg),ψgendmrg)

    edmrg_lm, ψdmrg_lm = dmrg(H, ψ0; nsweeps=50, maxdim=200, cutoff=1e-14,
     eigsolve_which_eigenvalue=:LM, ishermitian=false, eigsolve_maxiter=4, eigsolve_krylovdim=5)
    edmrg, ψdmrg = dmrg(H, ψ0; nsweeps=30, eigsolve_which_eigenvalue=:LR, ishermitian=false)

    edmrg_ev = inner(dag(ψdmrg)',H,ψdmrg)/inner(dag(ψdmrg),ψdmrg)
    edmrg_lm_ev = inner(dag(ψdmrg_lm)',H,ψdmrg_lm)/inner(dag(ψdmrg_lm),ψdmrg_lm)

  
    ψpm, ds2s  = powermethod_sym(ψ0, H, pm_params)

    epm = inner(dag(ψpm)',H,ψpm)/inner(dag(ψpm),ψpm)
    #epm_alt = inner(ψpm',H,ψpm)

    if n > 12
      @warn "System size of $n is likely too large for exact diagonalization."
      vals, vecs, info = 0,0,0
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
        H_full, ψ0_full, 4, :LM; ishermitian=false, tol=1e-6, krylovdim=30, eager=true
      )
    end

    @show (edmrg)
    @show (edmrg_ev)
    @show (edmrg_lm)
    @show (edmrg_lm_ev)
    @show egendmrg
    #@show (egendmrg_alt1)
    @show (egendmrg_ev)
    @show epm
    #@show epm_alt
    @show vals[1]


    return edmrg, edmrg_ev, edmrg_lm, edmrg_lm_ev, egendmrg, egendmrg_ev, epm, vals[1], vals, ψpm, ψdmrg_lm, ψgendmrg
    
    
  end
  
  function main_tmpo(tstart::Int, tend::Int)

    ts = tstart:tend
    ens = []
    for t in ts
      push!(ens, main_tmpo(t))
    end
    return ts, ens
  end

  function compare_plot()
    plotlyjs()

    methods = ("edmrg", "edmrg_ev", "edmrg_lm", "edmrg_lm_ev", "egendmrg", "egendmrg_ev", "epm", "exact")

    ts, ens = main_tmpo(8, 40)
    pr = plot()
    for ii in eachindex(ens[1])
      scatter!(pr, [real(e[ii]) for e in ens], label=methods[ii])
    end

    ppi = plot()
    for ii in eachindex(ens[1])
      scatter!(ppi, [imag(e[ii]) for e in ens], label=methods[ii])
    end

    l = @layout [a b]
    plot(pr, ppi, layout=l)
  end


  function compare_print()

    methods = ("edmrg", "edmrg_ev", "edmrg_lm", "edmrg_lm_ev", "egendmrg", "egendmrg_ev", "epm", "exact")

    ts, ens = main_tmpo(6, 12)

    jldsave("out_compare.jld2"; methods, ts, ens)

    return ens
  
  end


  