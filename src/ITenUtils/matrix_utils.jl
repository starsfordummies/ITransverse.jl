
""" TODO Check
Builds a random matrix with a decaying singular value spectrum
"""
function randmat_decayspec(n::Int)
    @assert n > 1
    h1 = rand(ComplexF64,n,n)
    h1 = h1 + transpose(conj(h1))
    u = exp(im*h1)

    h2 = rand(ComplexF64,n,n)
    h2 = h2 + transpose(conj(h2))
    v = exp(im*h2)

    sv = rand(1)
    for jj = 2:n
        push!(sv, exp(-jj)*rand())
    end

    normalize!(sv)

    mat = u * Diagonal(sv) * v

    return mat
end

""" Checks if a matrix is diagonal within a given cutoff
"""
function check_diag_matrix(d::AbstractMatrix, cutoff::Float64=1e-6)
    delta_diag = norm(d - Diagonal(d))/norm(d)
    if delta_diag > cutoff
        println("Warning, matrix non diagonal: $delta_diag")
        return false
    end
    return true
end


""" Check if a matrix is identity within a given cutoff 
"""
function check_id_matrix(m::Matrix, cutoff::Float64=1e-8)
    if size(m,1) == size(m,2)
        delta_diag = norm(m - I(size(m,1)))/norm(m)
        if delta_diag > cutoff
            @warn("Not identity: off by(norm) $delta_diag")
            if norm(m./m[1,1] - I(size(m,1)))/norm(m) < cutoff
                @info("But proportional to identity, factor $(m[1,1])")
            end
            return false
        end
        return true
    else
        @error ("Not even square? $(size(d))")
        return false
    end
end


""" Symmetrizes a matrix to improve numerical stability (throws an error if it's not too symetric to begin with)
    TODO This fails for GPU matrices?!  """
    function symmetrize(a::Matrix, tol::Float64=1e-6)
        if size(a,1) != size(a,2)
            @error("Not square")
        end
        if norm(a - transpose(a))/norm(a) > tol
            @error("Not symmetric? norm(a-aT)=$(norm(a - transpose(a)))")
            #sleep(2)
        end
        return 0.5*(a + transpose(a))
    end
    

    