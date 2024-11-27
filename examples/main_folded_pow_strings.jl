#using Revise
using ITensors, JLD2
using Plots

using ITransverse
#using ITransverse.ITenUtils

ITensors.enable_debug_checks()


function main_folded_pm()

    #tp = ising_tp()
    tp = tMPOParams(0.1, build_expH_ising_murg, 
    ModelParams("S=1/2", 1.0, 0.4, 0.0), 0, [1,0], [1,0,0,1])
    tp_proj = tMPOParams(0.1, build_expH_ising_murg, 
    ModelParams("S=1/2", 1.0, 0.4, 0.0), 0, [1,0], [1,0,0,0])
    cutoff = 1e-20
    maxbondim = 120
    length_string = 10
    itermax = 10
    total_size =2*itermax+length_string
    eps_converged=1e-6

    truncp = TruncParams(cutoff, maxbondim)

    pm_params = PMParams(truncp, itermax, eps_converged, true, "RTM_LR")

    sigX = ComplexF64[0,1,1,0]
    P_up = ComplexF64[1,0,0,0]

    evs = [] 

    rvecs = []
    ds2s = []


    tp = tMPOParams(tp; nbeta=4)
    #tp = tMPOParams(tp;bl =[1,0,0,0])
    tp_proj =  tMPOParams(tp_proj; nbeta=4)

    tpim = tMPOParams(tp; dt=-im*tp.dt)
    tpim_proj = tMPOParams(tp_proj; dt=-im*tp_proj.dt)


    b = FoldtMPOBlocks(tp)
    b_im = FoldtMPOBlocks(tpim)
    b_proj_im = FoldtMPOBlocks(tpim_proj)

    infos = Dict("tp" => tp, "pm_params" => pm_params)
    ts= 20:1:20

    alltimes = ts.* tp.dt

    for Nsteps in ts

        time_sites = siteinds(4, Nsteps)

        
        init_mps = folded_right_tMPS(b, time_sites)

        mpo_P_up = folded_tMPO(b, b_im, time_sites, P_up)
        mpo_1 = folded_tMPO(b, b_im, time_sites)


        rr, ll, ds2_pm  = powermethod(init_mps, mpo_1, mpo_P_up, pm_params) 

        ev = compute_expvals(ll, rr, ["Pz"], b)
        push!(rvecs, rr)
        push!(evs, ev)
        push!(ds2s, ds2_pm)
        itermax = 1
        pm_params = PMParams(truncp, itermax, eps_converged, true, "RTM_LR")
        for i in 1:length_string
            println("i = ", i)
            
            rr, ll, ds2_pm  = powermethod(init_mps, mpo_P_up, mpo_P_up, pm_params) 
            ev = compute_expvals(ll, rr, ["Pz"], b)
            push!(rvecs, rr)
            push!(evs, ev)
            push!(ds2s, ds2_pm)
        end

    end

    return rvecs, evs, ds2s, ts, infos
end



rvecs, evs, ds2s, alltimes, infos = main_folded_pm()



# evs contains my restults, I want to plot them
x_values = 1:2:length(evs) * 2 - 1  # Create x-axis values starting at 1, 3, 5, ...
keys_evs = collect(keys(evs))
values_evs = collect(values(evs))
plot(x_values, values_evs, title="Results", xlabel="Index", ylabel="Value", legend=false)

# Display the plot
display(plot)
