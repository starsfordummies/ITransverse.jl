using ITensors, ITensorMPS, JLD2

using ITransverse

function main(resume_filename = "none")

JXX = 1.0  
hz = 0.5
dt = 0.1

nbeta = 3

init_state = Vector{ComplexF64}([1,0])

SVD_cutoff = 1e-12
maxbondim = 400
itermax = 800

params = Dict("JXX" => JXX , "hz" => hz, "dt" => dt, "nbeta" => nbeta, "init_state" => init_state)
pm_params = Dict(:itermax => itermax, :SVD_cutoff=> SVD_cutoff, :maxbondim => maxbondim, :verbose => false)



out_filename = "out_ents_ising_" * Dates.format(now(), "yymmdd_HHMM") * ".jld2"


ll_murgs = []
ds2s = []


Tstart = 30
Tend = 50
Tstep = 1

if isfile(resume_filename)
    temp = jldopen(resume_filename)
    Tstart = temp["curr_T"]
    close(temp)
    println("Trying to resume from $resume_filename , starting from $Tstart")
    out_filename = resume_filename
end

Ntime_steps = Tstart
Nsteps = Ntime_steps +2*nbeta
time_sites =  addtags(siteinds("S=1/2", Nsteps; conserve_qns = false), "time")

ll_murg_s = [] 

for ts = Tstart:Tstep:Tend
    
    if isfile(out_filename)
        println("Appending nt=$ts results to $out_filename")
        temp = jldopen(out_filename)
        ll_murgs = temp["ll_murgs"]
        ds2s = temp["ds2s"]
        close(temp)
    end



    Ntime_steps = ts

    Nsteps = Ntime_steps +2*nbeta


    println("Optimizing for $ts timesteps + $nbeta imag steps ")
    println("Initial state $init_state  => quench @ J=$JXX , h=$hz ")

    time_sites =  addtags(siteinds("S=1/2", Nsteps; conserve_szparity=true), "time")
    #addtags(time_sites, "time")

    if ts == Tstart
        state = [isodd(n) ? "Up" : "Dn" for n=1:length(time_sites)]
        start_mps = productMPS(time_sites, state) #productMPS(time_sites,"+");
    else
        start_mps = extend_mps_v(ll_murg_s, time_sites)
    end

    mpo_L = build_ising_tMPO_regul_beta(build_expH_ising_murg, JXX, hz, dt, nbeta, time_sites, init_state)

    #println(inds(start_mps))
    #println(inds(mpo_L))
    
    ll_murg_s, ds2s_murg_s  = powermethod_sym(start_mps, mpo_L, pm_params)

    push!(ll_murgs, ll_murg_s)
    push!(ds2s, ds2s_murg_s)

    curr_T = ts
    jldsave(out_filename; nbeta, dt, ll_murgs, ds2s, params, pm_params, curr_T)

end

end


main("new")