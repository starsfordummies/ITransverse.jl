# using ITensors
# using TakagiFactorization

# include("utils.jl")
# include("truncations.jl")


"""
Performs SVD decomposition of a SYMMETRIC tensor using Autonne-Takagi \\
assumes that NO degeneracy in SVs is present, so the intermediate matrix z is truly diagonal \\
Intermediate steps are done with the ITensors
"""
function symmetric_svd_iten(a::ITensor; svd_cutoff::Float64=1e-12, chi_max::Int=100, check_symmetry::Bool=false)
 
    # check symmetry ? 
    if check_symmetry

        @assert order(a) == 2 
        @assert dims(a)[1] == dims(a)[2]

        #(i,j) = inds(a)

        a1 = array(a) # Array(a, i, j)
        a2 = transpose(array(a)) # Array(a, j, i)
        if a1 ≈ a2
            println("Symmetric")
        else
            println("Warning: not symmetric!")
        end
    end

    u, s, vd = ITensors.svd(a, ind(a,1), cutoff=svd_cutoff, maxdim=chi_max)

    z = dag(u) * delta(inds(a))
    z = z * vd

    if abs(norm(z) - norm(diag(z)))/norm(z) > 0.1
        println("warning: z likely not diag")
    end

    zdiag = diagITensor(diag(z).storage.data, inds(z))
    sq_z = sqrt.(zdiag)

    uz = u * sq_z 

    # check that the SVD decomposition is accurate 
    uzt = (uz * delta(inds(a)) * delta(inds(s)))
    a_rec = uz * s * uzt 

    if  !isapprox(a_rec, a)
        println("warning, AT/SVD decomp maybe not accurate, $(norm(a_rec-a)) [rel=$(norm(a_rec-a)/norm(a)) ")
    end

    return uz, s, uzt 
end


function symmetric_svd_ndten(a::ITensor; svd_cutoff::Float64=1e-12, check_symmetry::Bool=false)
 
    i = ind(a,1)
    
    a_nd = NDTensors.Tensor(a, inds(a))
    # check symmetry ? 
    if check_symmetry
        @assert a_nd ≈ transposee(a_nd)
    end

    u, s, v = svd(a_nd, cutoff=svd_cutoff)


    z = transpose(conj(u)) *  v
    #equiv ? z = transpose(v) * conj(u)

    if abs(norm(z) - norm(diag(z))) > 0.1
        println("warning: z likely not diag")
    end

    zdiag =  NDTensors.diagm(diag(z))
    sq_z = sqrt.(zdiag)

    uz = u * sq_z 

    # check that the SVD decomposition is accurate 
    a_rec = uz * (s * transpose(uz))
    if  !isapprox(a_rec, a_nd)
        println("warning, AT/SVD decomp maybe not accurate, $(norm(a_rec-a)) [rel=$(norm(a_rec-a)/norm(a)) ")
    end


    iu = Index(size(s)[1],tags="u")
    iv = Index(size(s)[1],tags="v")

    uz = ITensor(uz, i, iu)
    s = diagITensor(s.storage.data, iu, iv)

    return uz, s
end




""" Symmetric SVD decomposition using TakagiFactorization library """
function symmetric_svd_takagi_iten(a::ITensor; svd_cutoff::Float64=1e-12, chi_max::Int=100, check_symmetry::Bool=true)
 
    @assert order(a) == 2 
    @assert dims(a)[1] == dims(a)[2]

    (i,j) = inds(a)

    arr_a = Array(a, i, j)
 
    # check symmetry ? 
    if check_symmetry
        a1 = Array(a, i, j)
        a2 = Array(a, j, i)
        if a1 ≈ a2
            println("Symmetric")
        else
            println("Warning: not symmetric!")
        end
    end


    # symmetrize forcibly 
    a_sym = 0.5 * ( arr_a + transpose(arr_a) )

    s, uz = takagi_factor(a_sym, sort=-1)

    uzt = transpose(uz)
    @assert a_sym ≈ uzt * s * uz

    return uzt, s, uz 

    # TODO can I truncate here ? 
    # normsq_svd = norm(s)^2
    # s = s[s/normsq_svd .> svd_cutoff]

    # k = min(length(s), chi_max)

    # # TODO return sum of truncated SVs to estimate error
    # #trunc_err = sum(s[k:length(s)].^2)/normsq_svd
    # trunc_err = norm(s[k:length(s)])^2/normsq_svd


    # # a_rec = uz * s * uzt # (uz * delta(i,j) * delta(inds(s)))
    # # if  !isapprox(a_rec, a)
    # #     println("warning, AT/SVD decomp maybe not accurate, $(norm(a_rec-a)) [rel=$(norm(a_rec-a)/norm(a)) ")
    # # end

    # return uzt[:,1:k], s[1:k], uz[1:k,:], trunc_err
end


""" Symmetric SVD decomposition using TakagiFactorization library """
function symmetric_svd_takagi_arr(a::Matrix{ComplexF64}; svd_cutoff::Float64=1e-12, chi_max::Int=100)
 

    s, uz = takagi_factor(a, sort=-1)

    uzt = transpose(uz)
    println(norm(a - uzt * s * uz), "   ", norm(a) )
    println(diag(s))
    @assert a ≈ uzt * s * uz


    normsq_svd = norm(s)^2
    s = s[s/normsq_svd .> svd_cutoff]

    k = min(length(s), chi_max)

    # TODO return sum of truncated SVs to estimate error
    #trunc_err = sum(s[k:length(s)].^2)/normsq_svd
    trunc_err = norm(s[k:length(s)])^2/normsq_svd


    # a_rec = uz * s * uzt # (uz * delta(i,j) * delta(inds(s)))
    # if  !isapprox(a_rec, a)
    #     println("warning, AT/SVD decomp maybe not accurate, $(norm(a_rec-a)) [rel=$(norm(a_rec-a)/norm(a)) ")
    # end

    return uzt[:,1:k], s[1:k], uz[1:k,:], trunc_err
end




""" 
Performs SVD decomposition of a SYMMETRIC tensor using Autonne-Takagi \\ 
assumes that NO degeneracy in SVs is present, so the intermediate matrix z is truly diagonal \\
Returns Uz, S so that  a ≈ Uz * S * Uzᵀ
Intermediate steps are done with arrays - seems *much* faster than symmetric_svd ! 
TODO: implement degenerate case
""" 


function symmetric_svd_arr(a::ITensor; 
    svd_cutoff::Real=1e-12, chi_max::Int=100, 
    check_precision::Bool=false) 

    @assert order(a) == 2 
    @assert dims(a)[1] == dims(a)[2]

    i= ind(a,1)

    symmetric_svd_arr(array(a), i; svd_cutoff=svd_cutoff, chi_max=chi_max, check_precision=check_precision) 
end


function symmetric_svd_arr(arr_a::Matrix{T}, i::Index{Int};   
    svd_cutoff::Real=1e-12, chi_max::Int=100, check_precision::Bool=false) where T<:Union{ComplexF64,Float64}
    
#function symmetric_svd_arr(arr_a::Array{T,2}, i::Index{Int};   
#    svd_cutoff::Real=1e-12, chi_max::Int=100, check_precision::Bool=false) where T<:Union{ComplexF64,Float64}

    #check symmetry
    @assert isapprox(arr_a, transpose(arr_a) ) "Not symmetric? $arr_a"

    # symmetrize forcibly 
    #a_sym = (0.5 * ( arr_a + transpose(arr_a) )) 
    #u,s,vd, trunc_err = svdtrunc(a_sym, svd_cutoff=svd_cutoff, chi_max=chi_max)

    #arr_a = (0.5 * ( arr_a + transpose(arr_a) )) 
    u,s,vd, trunc_err = svdtrunc(arr_a, svd_cutoff=svd_cutoff, chi_max=chi_max)


    z = transpose(conj(u)) * transpose(vd)
    #sq_z = sqrt.(Diagonal(diag(z)))
    sq_z = sqrt(z)
    uz = u * Diagonal(transpose(sq_z))
    a_rec = uz * Diagonal(s) * transpose(uz)

    if check_precision
        if norm(a_rec-arr_a)/norm(arr_a) > 1e-6
            println("warning, AT/SVD decomp maybe not accurate, $(norm(a_rec-a)) vs[$(norm(a))]")
        end
    end

    iu = Index(size(s)[1],tags="u")
    iv = Index(size(s)[1],tags="v")


    #uz = ITensor(uz, i, iu)
    #s = diagITensor(s, iu, iv)
    #return uz, s 

    return  ITensor(uz, i, iu), diagITensor(s, iu, iv)
end


""" Performs Symmetric eigenvalue decomposition (Takagi-style) for a complex symmetric matrix, A = O D O^T 
"""
function symmetric_eig_arr(a::ITensor; 
    svd_cutoff::Real=1e-12, chi_max::Int=100, 
    check_precision::Bool=false) 

    @assert order(a) == 2 
    @assert dims(a)[1] == dims(a)[2]

    i= ind(a,1)

    symmetric_eig_arr(array(a), i; svd_cutoff=svd_cutoff, chi_max=chi_max, check_precision=check_precision) 
end

""" Performs Symmetric eigenvalue decomposition (Takagi-style) for a complex symmetric matrix, A = O D O^T 
"""
function symmetric_eig_arr(arr_a::Matrix{T}, i::Index{Int};   
    svd_cutoff::Real=1e-12, chi_max::Int=100, check_precision::Bool=false) where T<:Union{ComplexF64,Float64}
    
    @assert isapprox(arr_a, transpose(arr_a) ) "Not symmetric? $arr_a"

    # symmetrize forcibly 
    #a_sym = (0.5 * ( arr_a + transpose(arr_a) )) 
    #u,s,vd, trunc_err = svdtrunc(a_sym, svd_cutoff=svd_cutoff, chi_max=chi_max)

    #arr_a = (0.5 * ( arr_a + transpose(arr_a) )) 
    u,s,vd, trunc_err = svdtrunc(arr_a, svd_cutoff=svd_cutoff, chi_max=chi_max)


    z = transpose(conj(u)) * transpose(vd)
    #sq_z = sqrt.(Diagonal(diag(z)))
    sq_z = sqrt(z)
    uz = u * Diagonal(transpose(sq_z))
    a_rec = uz * Diagonal(s) * transpose(uz)

    if check_precision
        if norm(a_rec-arr_a)/norm(arr_a) > 1e-6
            println("warning, AT/SVD decomp maybe not accurate, $(norm(a_rec-a)) vs[$(norm(a))]")
        end
    end

    iu = Index(size(s)[1],tags="u")
    iv = Index(size(s)[1],tags="v")


    #uz = ITensor(uz, i, iu)
    #s = diagITensor(s, iu, iv)
    #return uz, s 

    return  ITensor(uz, i, iu), diagITensor(s, iu, iv)
end


