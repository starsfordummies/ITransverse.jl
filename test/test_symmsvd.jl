using ITensors
using LinearAlgebra
using ITransverse
using Test


""" Older version for checks """
function symm_svd_ref(a::ITensor, linds, rinds = uniqueinds(a, linds) ; kwargs...)
   

    cL = combiner(linds)
    cR = combiner(rinds)

    ac = a * cL * cR

    @assert ndims(ac) == 2 "check your inds? $(inds(ac))"

    iL = combinedind(cL)
    iR = combinedind(cR)

    ac = symmetrize(ac)

    # u * s * vd ≈ a 
    u,s,vd, spec = svd(ac, iL; kwargs...)
   
    index_u = commonind(u,s)
    index_v = commonind(vd,s)

    #@show matrix(u)
    #@show matrix(vd)
    #@show u * s * vd ≈ ac

    z = noprime(dag(u) * (vd' * delta(iL, iR')))

    #@show inds(z)
    # Z could still be block-diagonal. 
    # What is the safest way to invert it ? With SVD it doens't work so well,
    # maybe with eigenvalue decomp since it's symmetric? 
    # zvals, zvecs = eigen(z, index_u, index_v)
    # @info zvecs * zvals * dag(zvecs)' ≈ z
    # sq_z = zvecs * sqrt.(zvals) * dag(zvecs)

    # Best way is probably still to rely on Schur decomposition from Julia's matrix utils !? 
    sq_z = sqrt(z) # ITensor(sqrt(matrix(z)), inds(z))

    # TODO for GPU aware code : check if z is diagonal -> do on GPU
    # otherwise, bring back to CPU, do it here and bring back to GPU

    #@show matrix(z)
    #@show matrix(sq_z)

    u *= sq_z 
    uT = u * delta(iL, iR) 

    u *= delta(index_u, index_v)
    u *= dag(cL)
    uT *= dag(cR)
  
    return ITensors.TruncSVD(u,s,uT, spec, index_u, index_v)
end

@testset "symmetric SVD" begin
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
t1 = random_itensor(i1,i2)

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

ts1 = randsymITensor(400)
is1 = ind(ts1,1)

fst = symm_svd(ts1, is1)

fst.U * fst.S * fst.V ≈ ts1

u,s,uT = symm_svd(ts1, is1)

@test u * s * uT ≈ ts1


u2,s2,uT2 = symm_svd_ref(ts1, is1)

@test u2 * s2 * uT2 ≈ ts1

array(u) ≈ array(u2)
array(s) ≈ array(s2)
array(uT) ≈ transpose(array(uT2))

end
