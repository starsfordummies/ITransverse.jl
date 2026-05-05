using ITensors, ITensorMPS, ITransverse
using Test 

ts = siteinds(3, 80)
psi0 = random_mps(ts)
pm_params = PMParams()


bsym = FwtMPOBlocks(SymSVD(), PottsParams(1, 0.9))
fwmpo = fw_tMPO(bsym, ts)
psi_pmsym, infos = powermethod_sym(psi0, fwmpo, pm_params)


bns = FwtMPOBlocks(Murg(), PottsParams(1, 0.9))
fwmpo = fw_tMPO(bns, ts)

psi_fakesym, infos = powermethod_sym(psi0, fwmpo, pm_params)

left, right, infos = ITransverse.powermethod_lr(psi0, fwmpo, fwmpo, pm_params)

fidelity(psi_pmsym, psi_fakesym)

fidelity(psi_pmsym, right)

fidelity(psi_pmsym, left)

ssym = gensym_renyi_entropies(psi_pmsym)
ssymf = gensym_renyi_entropies(psi_fakesym)
s2 = gen_renyi2(left, right)