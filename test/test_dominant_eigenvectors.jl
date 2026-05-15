using ITensors
using ITransverse 
using Test 

@testset "dominant eigenvectors" begin 
ii = Index(8,"left")
jj = Index(8,"right")

tt = random_itensor(ComplexF64, ii, jj)
vals, vecs =  dominant_eigenvectors(tt, jj)

valsE, vecsE = eigen(matrix(tt), sortby = x -> -abs(x))

@test vals[1] ≈ valsE[1]
array(vecs[1]) # ≈ vecsE[:,1] 
vecsE[:,1]

array(vecs[1]) ≈ vecsE[:,1]

v1 = array(vecs[1]) ./ array(vecs[1])[1]
v2 = vecsE[:,1] ./ vecsE[1,1]

@test v1 ≈ v2

# Left eigenvectors 
vals, vecs =  dominant_eigenvectors(tt, ii)

valsE, vecsE = eigen(transpose(matrix(tt)), sortby = x -> -abs(x))


@test vals[1] ≈ valsE[1]
array(vecs[1]) # ≈ vecsE[:,1] 
vecsE[:,1]

array(vecs[1]) ≈ vecsE[:,1]

v1 = array(vecs[1]) ./ array(vecs[1])[1]
v2 = vecsE[:,1] ./ vecsE[1,1]

@test v1 ≈ v2

end