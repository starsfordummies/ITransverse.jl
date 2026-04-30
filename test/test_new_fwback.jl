using ITensors, ITensorMPS, ITransverse
using ITensors.Adapt
using ITransverse: to_itensor
using Test

""" Unfolded tMPO with 
- `nbetai` initial steps of imaginary time evolution 
- `nfw` steps of forward time evolution 
-  (optionally) a `mid_op` operator insertion 
- `nback` steps of backwards time evolution
- `nbetaf` steps of imaginary time evolution
"""

@testset "new fwmpo builder" begin

JXX = 1.0
hz = 0.7
gx = 0.0
#H= JXX - 2.0 * 0.525 Z + 2 * 0.25 X


dt = 0.1

# init_state = plus_state
init_state = up_state

mp = IsingParams(JXX, hz, gx)

tp = tMPOParams(dt, Murg(), mp, 0, init_state)
b = FwtMPOBlocks(tp)

ss = siteinds("S=1/2", 16)

tmpo_new = fwback_tMPO(b, ss, 2, 6, 6, 2, mid_op = [1,0,0,-1], tr=b.tp.bl)
tmpo_old = ITransverse.fwback_tMPO_old(b, ss, 2, 6, 6, 2, mid_op = [1,0,0,-1], tr=b.tp.bl)

@test tmpo_new ≈ tmpo_old

end
