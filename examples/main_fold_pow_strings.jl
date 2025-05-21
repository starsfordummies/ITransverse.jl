#using Revise
using ITensors, JLD2
using Plots
#using LaTeXStrings
using ITransverse
using ProgressMeter
#using ITransverse.ITenUtils

ITensors.enable_debug_checks()


function main_folded_pm(itermax::Int,length_string::Int; ts::Int=10,Jxx::Float64=1.0,hz::Float64=0.9,hx::Float64=0.0)

    #If I understand correctly the Ising Hamiltonian is defined as Jxx XX + hz Z +hx X

    #tp = ising_tp()
    tp = tMPOParams(0.1, build_expH_ising_murg, 
    IsingParams(Jxx, hz, hx), 0, [1,0], [1,0,0,1])
    tp_proj = tMPOParams(0.1, build_expH_ising_murg, 
    IsingParams(Jxx, hz, hx), 0, [1,0], [1,0,0,0])
    cutoff = 1e-12
    maxbondim = 120
    #length_string = 10
    #itermax = 10
    total_size =2*itermax+length_string
    eps_converged=1e-6

    truncp = TruncParams(cutoff, maxbondim)

    pm_params = PMParams(truncp, itermax, eps_converged, true, "RDM")

    sigX = ComplexF64[0,1,1,0]
    P_up = ComplexF64[1,0,0,0]

    ev = [] 



    tp = tMPOParams(tp; nbeta=4)
    #tp = tMPOParams(tp;bl =[1,0,0,0])
    tp_proj =  tMPOParams(tp_proj; nbeta=4)

    b = FoldtMPOBlocks(tp)
    
    infos = Dict("tp" => tp, "pm_params" => pm_params)

    alltimes = ts.* tp.dt

    time_sites = siteinds(4, ts)

    
    init_mps = folded_right_tMPS(b, time_sites)

    mpo_P_up = folded_tMPO(b, time_sites; fold_op=P_up)
    mpo_1 = folded_tMPO(b, time_sites)

    #Here I am using the power method to find the the left and right environments after itermax which I interpret as the spatial size of the system 
    println("itermax no string= ", itermax)
    rr, ll, ds2_pm  = powermethod(init_mps, mpo_1, mpo_P_up, pm_params) 


    itermax = Int((length_string-1)/2)
    println("itermax with string= ", itermax)
    if itermax > 0
        pm_params = PMParams(truncp, itermax, eps_converged, true, "RDM")
    #println("i = ", i)
        #I will use the left environment of the previous step as the initial state, hoping that this is fine.
        rr, ds2_pm  =  powermethod_sym(ll, mpo_P_up, pm_params) 
        #Notice that I expect that I absorb twice mpo_P_up, so I should get the expectation value of P_up \otimes P_up
    end
    
    
    ev = compute_expvals(rr, rr, ["Pz"], b)
    #end

    return ev
end




values_evs = []

ppp  = plot()

for ts in 30:30
    println("ts = ", ts)
    evs=[]
    total_lenght = 51
    string_length = 1
    Jxx=1.
    hx=0.0
    hz=0.7
    for string_length in 1:2:total_lenght
        itermax = Int((total_lenght-string_length)/2)
        println("itermax = ", itermax)
        println("string_length = ", string_length)
        val_ev = main_folded_pm(itermax, string_length; ts, hz)
        print(val_ev)
        push!(evs, val_ev)
    end

# evs contains my restults, I want to plot them
x_values = 1:2:length(evs) * 2 - 1  # Create x-axis values starting at 1, 3, 5, ...
values_evs = [ev["Pz"] for ev in evs if haskey(ev, "Pz")]

plot!(ppp, x_values, log.(-abs(values_evs).+1.),title="L = $total_lenght, Jxx=$Jxx, hz=$hz,hx=$hx" , xlabel=L"l_s=\text{lenght  string}", ylabel=L"\log(1-\langle Pz^{\otimes l_s} \rangle)", legend=true, label="ts = $ts",seriestype=:scatter, markershape=:circle, markercolor=:blue, markerstrokecolor=:blue, markerstrokewidth=2, markersize=5)
plot!(ppp, x_values, log.(-abs(values_evs).+1.), seriestype=:line, linecolor=:black)

end
# Display the plot
plot(ppp)





