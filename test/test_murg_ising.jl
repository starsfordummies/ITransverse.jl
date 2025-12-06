using ITensors, ITensorMPS
using ITransverse
using Test


# function contract(t1::ITensor, t2::ITensor, )

ss = siteinds("S=1/2", 3)

JJ = 0.7
eH = ITransverse.ChainModels.build_expXX_murg(ss, JJ)


gate1 = eH[1] * eH[end] * delta( linkind(eH,1),linkind(eH,2) )

gate2 = exp(JJ*im*(op(ss[1],"X")*op(ss[3],"X")))

@test gate1 ≈ gate2

