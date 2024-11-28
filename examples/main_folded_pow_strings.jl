#using Revise
using ITensors, JLD2
using Plots
using LaTeXStrings
using ITransverse
#using ITransverse.ITenUtils

ITensors.enable_debug_checks()


function main_folded_pm(itermax::Int,length_string::Int,ts::Int=10,Jxx::Float64=1.0,hz::Float64=1.0,hx::Float64=0.0)

    #If I understand correctly the Ising Hamiltonian is defined as Jxx XX + hz Z +hx X

    #tp = ising_tp()
    tp = tMPOParams(0.1, build_expH_ising_murg, 
    ModelParams("S=1/2", Jxx, hz, hx), 0, [1,0], [1,0,0,1])
    tp_proj = tMPOParams(0.1, build_expH_ising_murg, 
    ModelParams("S=1/2", Jxx, hz, hx), 0, [1,0], [1,0,0,0])
    cutoff = 1e-20
    maxbondim = 120
    #length_string = 10
    #itermax = 10
    total_size =2*itermax+length_string
    eps_converged=1e-6

    truncp = TruncParams(cutoff, maxbondim)

    pm_params = PMParams(truncp, itermax, eps_converged, true, "RTM_LR")

    sigX = ComplexF64[0,1,1,0]
    P_up = ComplexF64[1,0,0,0]

    ev = [] 




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

    #for Nsteps in ts

        time_sites = siteinds(4, ts)

        
        init_mps = folded_right_tMPS(b, time_sites)

        mpo_P_up = folded_tMPO(b, b_im, time_sites, P_up)
        mpo_1 = folded_tMPO(b, b_im, time_sites)

        #Here I am using the power method to find the the left and right environments after itermax which I interpret as the spatial size of the system 
        println("itermax no string= ", itermax)
        rr, ll, ds2_pm  = powermethod(init_mps, mpo_1, mpo_P_up, pm_params) 
        
        #ev = compute_expvals(ll, rr, ["Pz"], b)
        # and collect the results 
        #push!(rvecs, rr)
        #push!(evs, ev)
        #push!(ds2s, ds2_pm)
        #Now I will go step by step and absorb the operator P_up into the left and right, thus computing the expectation value of a larger string
        itermax = (length_string-1)/2
        println("itermax with string= ", itermax)
        pm_params = PMParams(truncp, itermax, eps_converged, true, "RTM_LR")
        #println("i = ", i)
            #I will use the left environment of the previous step as the initial state, hoping that this is fine.
        rr, ll, ds2_pm  =  powermethod(ll, mpo_P_up, mpo_P_up, pm_params) 
            #Notice that I expect that I absorb twice mpo_P_up, so I should get the expectation value of P_up \otimes P_up
            
        
        ev = compute_expvals(ll, rr, ["Pz"], b)
    #end

    return ev
end
for ts in 10:40
    println("ts = ", ts)
evs=[]
total_lenght = 51
string_length = 1
Jxx=1.
hx=0.0
hz=1.0  
for string_length in 1:2:total_lenght
    itermax = Int((total_lenght-string_length)/2)
    println("itermax = ", itermax)
    println("string_length = ", string_length)
    val_ev = main_folded_pm(itermax, string_length)
    print(val_ev)
    push!(evs, val_ev)
end


# evs contains my restults, I want to plot them
x_values = 1:2:length(evs) * 2 - 1  # Create x-axis values starting at 1, 3, 5, ...
values_evs = [ev["Pz"] for ev in evs if haskey(ev, "Pz")]

plot!(x_values, log.(-real(values_evs).+1.),title="L = $total_lenght, Jxx=$Jxx, hz=$hz,hx=$hx" , xlabel=L"l_s=\text{lenght  string}", ylabel=L"\log(1-\langle Pz^{\otimes l_s} \rangle)", legend=true, label="ts = $ts",seriestype=:scatter, markershape=:circle, markercolor=:blue, markerstrokecolor=:blue, markerstrokewidth=2, markersize=5)
plot!(x_values, log.(-real(values_evs).+1.), seriestype=:line, linecolor=:black)

# Display the plot
display(plot)
end