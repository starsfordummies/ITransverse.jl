using ITensors
using LinearAlgebra
using ITransverse.ITenUtils
using Test

m1 = rand(ComplexF64, 8, 13)

f = svd(m1)

# svd(m) returns u,s,v so that m = u*s*vdagger 
u,s,v = svd(m1)

m1 ≈ u * Diagonal(s) * v'

# The SVD() object is built via vdagger though !!!
@test f == SVD(u,s,v') 
#@test f == SVD(u,s,v)  # False! 



# Now ITensors 

i1 = Index(9)
i2 = Index(15)
t1 = randomITensor(i1,i2)

f = svd(t1,i1)
u,s,vd = svd(t1,i1)

# ITensors works directly with conjugates (V = Vdagger) 
# the V never appears here 
# No need to conjugate here! 
t1 ≈ u * s * vd

u,s,v, spec, index_u, index_v = svd(t1, i1)

f == ITensors.TruncSVD(u,s,v, spec, index_u, index_v)

f.U == u
u
f.U 

@test matrix(u) ≈ matrix(f.U)
@test matrix(v) ≈ matrix(f.V)

m2 = rand(ComplexF64, 8, 8)
s1 = m2 + transpose(m2)

fs, spec = symm_svd(s1)

@test fs.U * Diagonal(fs.S) * fs.V' ≈ s1

ts1 = randsymITensor(10)
is1 = ind(ts1,1)

fst = symm_svd(ts1, is1)

fst.U * fst.S * fst.V ≈ ts1

u,s,uT = symm_svd(ts1, is1)

@test u * s * uT ≈ ts1