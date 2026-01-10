using LinearAlgebra
using ITensors
using ITransverse
using ITransverse.ITenUtils
using ITransverse.ITenUtils: truncated_svd
using Test

@testset "Testing symmetric SVD/EIG" begin


a = 1. + 0.1im
b = 1e-3 + 1e-4im
c = 1e-5
d = 1e-8
d = 1e-12
e = 1e-14
f = 1e-30
m = [1 0 0 0 0 0 0 0 0 0  ;
     0 a 0 0 0 0 0 0 0 0 ; 
     0 0 2 2 2 0 0 0 0 0; 
     0 0 2 2 3 0 0 0 0 0; 
     0 0 2 3 4 0 0 0 0 0;
     0 0 0 0 0 b 0 0 0 0; 
     0 0 0 0 0 0 c 0 0 0 ;
     0 0 0 0 0 0 0 d 0 0 ;
     0 0 0 0 0 0 0 0 e 0 ;
     0 0 0 0 0 0 0 0 0 f 
     ]

cutoff = 1e-12

@test issymmetric(m)

f,spec = truncated_svd(m; cutoff)
@test norm(m - f.U * Diagonal(f.S) * f.V') < sqrt(cutoff) 

f,spec = symm_svd(m; cutoff)
@test norm(m - f.U * Diagonal(f.S) * transpose(f.U)) < sqrt(cutoff) 


f, spec = mytrunc_eig(m; cutoff)
# !this test can betricky with truncation if inverses do not well behave..
@test norm(m - f.vectors * Diagonal(f.values) * pinv(f.vectors)) < sqrt(cutoff)

f, spec = symm_oeig(m; cutoff)
@test norm(m - f.vectors * Diagonal(f.values) * transpose(f.vectors)) < sqrt(cutoff)


@info "Testing truncated_svd VS ITensors truncated svd()"

m = randITensor_decayspec(40)
m = 1234*m
cutoff = 1e-8

fmy, spec =  truncated_svd(matrix(m); cutoff)
fiten = svd(m, ind(m,1); cutoff)
@test fmy.S ≈ diag(fiten.S)
@test fmy.U ≈ matrix(fiten.U)
@test fmy.Vt ≈ Matrix(fiten.V, fiten.v, ind(m,2))




ms = randsymITensor(40)
ms /= norm(ms)
u,s,ut = ITransverse.ITenUtils.symm_svd(ms, ind(ms,1))
@test norm(u * s * ut - ms) < 1e-10

u,s,ut = ITransverse.ITenUtils.symm_svd(ms, ind(ms,1), cutoff=1e-3)
@test norm(u * s * ut - ms) < sqrt(1e-3)
end