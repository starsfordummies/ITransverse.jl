using ITensors, ITensorMPS, ITransverse

ss = siteinds("S=1/2", 20)

psi0 = MPS(ss, "Up")

mp = IsingParams(1, 0.5, -1.05)
tp = ising_tp(integrable=false)
tp = tMPOParams(tp; mp=mp)
psi_full =  ITransverse.tebd(psi0, tp, 100, TruncParams(1e-12, 1024))

amply = inner(psi_full, psi0)

amplis_chi = []
for chi = 10:10:120
    psi_t = ITransverse.tebd(psi0, tp, 100, TruncParams(1e-12, chi))
    ampli = inner(psi_t, psi0)
    push!(amplis_chi, ampli)
end

#plot(10:10:120, abs.(abs.(amplis_chi) .- abs(amply) )./abs(amply))
#plot(1 ./log.(10:10:120), abs.(abs.(amplis_chi) .- abs(amply) )./abs(amply), yscale=:log10, marker=:o)


