# relic from the past but eh 

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
