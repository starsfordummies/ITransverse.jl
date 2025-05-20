using ITensors, JLD2
using ITensorMPS
using ITransverse

JXX = 1.0
hz = 1.0
gx = 0.0
#H= JXX - 2.0 * 0.525 Z + 2 * 0.25 X

dt = 0.1

nbeta = 1

init_state = plus_state
#init_state = zero_state

mp = IsingParams(JXX, hz, gx)

@info ("Initial state $(init_state)  => quench @ $(mp) ")

tp = tMPOParams(dt,  ITransverse.ChainModels.build_expH_ising_murg_new, mp, nbeta, init_state)


b = FwtMPOBlocks(tp)

tpim = tMPOParams(tp; dt=-im*tp.dt)

b_im = FwtMPOBlocks(tpim)


Nsteps = 10+2 

time_sites = addtags(siteinds("S=1/2", Nsteps; conserve_qns=false), "time")

tmpo, start_mps = fw_tMPOn(b, b_im, time_sites; tr=init_state)

vL = Index(5) 
vR = Index(5)
vP = Index(2)

ten = random_itensor(vL, vP, vR)

si = siteinds(tmpo)
li = linkinds(tmpo)

tmpo.data[1] = ten * delta(vP, li[1]) * delta(vL, vR')
tmpo.data[end] = dag(ten) * delta(vP, li[end]) * delta(vR, vL')

new_si = [siteind(tmpo,i) for i in 1:length(tmpo)]

new_init_psi = random_mps(new_si)

tpsi = deepcopy(tmpo)

tpsi[1] =  noprime(tmpo[1]*ITensor([1,0,0,0,0], siteind(tmpo,1)))
for jj = 2:length(tmpo)-1
    tpsi[jj] = noprime(tmpo[jj]*ITensor([1,0], siteind(tmpo,jj)))
end
tpsi[end] =  noprime(tmpo[end]*ITensor([1,0,0,0,0], siteind(tmpo,length(tmpo))))
tpsi = MPS(tpsi.data)

apply(tmpo, tpsi)
applyn(tmpo, tpsi)