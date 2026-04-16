""" If we're SVD-ing a rank-2 ITensor, we can do it allocating much less memory and without needing to specify indices"""
function matrix_svd(
    A::ITensor;
    lefttags=nothing,
    righttags=nothing,
    svd_kwargs...
  )

  @assert ndims(A) == 2 

    lefttags = NDTensors.replace_nothing(lefttags, ts"Link,u")
    righttags = NDTensors.replace_nothing(righttags, ts"Link,v")


    USVT = svd(tensor(A); svd_kwargs...)

    if isnothing(USVT)
        return nothing
    end

    UT, ST, VT, spec = USVT
    U = itensor(UT)
    S = itensor(ST)
    V = itensor(VT)

    #@show first(S), sum(S)

    #@info diag(S)
    
    u = commonind(S, U)
    v = commonind(S, V) 

    U = settags(U, lefttags, u)
    S = settags(S, lefttags, u)
    S = settags(S, righttags, v)
    V = settags(V, righttags, v)

    u = settags(u, lefttags)
    v = settags(v, righttags)


    return ITensors.TruncSVD(U, S, V, spec, u, v)
end
