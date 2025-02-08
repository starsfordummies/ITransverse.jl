using ITensors
using ITransverse.ITenUtils: matrix_svd
using Test

a = random_itensor(Index(10), Index(10))

f = svd(a, ind(a,1), maxdim=7, cutoff=1e-3)

mf = matrix_svd(a, maxdim=7,cutoff=1e-3)

@test f.U.tensor ≈ mf.U.tensor
@test diag(f.S)≈ diag(mf.S)
@test f.V.tensor ≈ mf.V.tensor

