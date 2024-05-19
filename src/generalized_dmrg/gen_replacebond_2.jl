
# struct Exposed{Unwrapped,Object}
#     object::Object
#   end
  
#   expose(object) = Exposed{unwrap_type(object),typeof(object)}(object)
  
#   unexpose(E::Exposed) = E.object


# From ITensors/src/tensor_operations/matrix_decomposition.jl
struct TruncEigen
    D::ITensor
    V::ITensor
    Vt::ITensor
    spec::Spectrum
    l::Index
    r::Index
  end


# iteration for destructuring into components `D, V, spec, l, r = E`
iterate(E::TruncEigen) = (E.D, Val(:V))
iterate(E::TruncEigen, ::Val{:V}) = (E.V, Val(:spec))
iterate(E::TruncEigen, ::Val{:spec}) = (E.spec, Val(:l))
iterate(E::TruncEigen, ::Val{:l}) = (E.l, Val(:r))
iterate(E::TruncEigen, ::Val{:r}) = (E.r, Val(:done))
iterate(E::TruncEigen, ::Val{:done}) = nothing




default_maxdim() = typemax(Int)
default_mindim() = 1
default_cutoff() = 1e-8
default_noise() = false

default_maxdim(a) = minimum(size(a))
default_mindim(a) = true
default_cutoff(a) = zero(eltype(a))
default_svd_alg(a) = default_svd_alg(unwrap_type(a), a)
default_svd_alg(::Type{<:AbstractArray}, a) = "divide_and_conquer"
default_use_absolute_cutoff(a) = false
default_use_relative_cutoff(a) = true

replace_nothing(::Nothing, replacement) = replacement
replace_nothing(value, replacement) = value



using .NDTensors.Unwrap

    # TODO HACK HACK 
    #@enum Arrow In = -1 Out = 1 Neither = 0

# Allows overloading `replacebond!` based on the projected
# MPO type. By default just calls `replacebond!` on the MPS.
function replacebond_gen!(PH, M::MPS, b::Int, phi::ITensor; kwargs...)
    return replacebond_gen!(M, b, phi; kwargs...)
 end

  function replacebond_gen!(
    M::MPS,
    b::Int,
    phi::ITensor;
    normalize=nothing,
    swapsites=nothing,
    ortho=nothing,
    # Decomposition kwargs
    which_decomp=nothing,
    mindim=nothing,
    maxdim=nothing,
    cutoff=nothing,
    eigen_perturbation=nothing,
    svd_alg=nothing,
  )
    normalize = NDTensors.replace_nothing(normalize, false)
    swapsites = NDTensors.replace_nothing(swapsites, false)
    ortho = NDTensors.replace_nothing(ortho, "left")
  
    indsMb = inds(M[b])
    if swapsites
      sb = siteind(M, b)
      sbp1 = siteind(M, b + 1)
      indsMb = replaceind(indsMb, sb, sbp1)
    end
    L, R, spec = factorize_gen(
      phi,
      indsMb;
      mindim,
      maxdim,
      cutoff,
      ortho,
      which_decomp,
      eigen_perturbation,
      svd_alg,
      tags=tags(linkind(M, b)),
    )

    #println("Checking mix ortho @ $b, $ortho ")

    M[b] = L
    M[b + 1] = R


    if ortho == "left"
      leftlim(M) == b - 1 && setleftlim!(M, leftlim(M) + 1)
      rightlim(M) == b + 1 && setrightlim!(M, rightlim(M) + 1)
      normalize && (M[b + 1] ./= sqrt(scalar(M[b + 1]*M[b + 1])))

       #@show norm(M[b+1])
       #@show scalar(M[b+1] * M[b+1])
    
    elseif ortho == "right"
      leftlim(M) == b && setleftlim!(M, leftlim(M) - 1)
      rightlim(M) == b + 2 && setrightlim!(M, rightlim(M) - 1)
      normalize && (M[b] ./= sqrt(scalar(M[b] * M[b])))
    else
      error(
        "In replacebond!, got ortho = $ortho, only currently supports `left` and `right`."
      )
    end

    return spec
  end

  

function factorize_gen(
    A::ITensor,
    Linds...;
    mindim=nothing,
    maxdim=nothing,
    cutoff=nothing,
    ortho=nothing,
    tags=nothing,
    plev=nothing,
    which_decomp=nothing,
    # eigen
    eigen_perturbation=nothing,
    # svd
    svd_alg=nothing,
    use_absolute_cutoff=nothing,
    use_relative_cutoff=nothing,
    min_blockdim=nothing,
    (singular_values!)=nothing,
    dir=nothing,
  )
    # if !isnothing(eigen_perturbation)
    #   if !(isnothing(which_decomp) || which_decomp == "eigen")
    #     error("""when passing a non-trivial eigen_perturbation to `factorize`,
    #              the which_decomp keyword argument must be either "automatic" or
    #              "eigen" """)
    #   end
    #   which_decomp = "eigen"
    # end
    ortho = NDTensors.replace_nothing(ortho, "left")
    tags = NDTensors.replace_nothing(tags, ts"Link,fact")
    plev = NDTensors.replace_nothing(plev, 0)
  
    # Determines when to use eigen vs. svd (eigen is less precise,
    # so eigen should only be used if a larger cutoff is requested)
    automatic_cutoff = 1e-12
    Lis = commoninds(A, indices(Linds...))
    Ris = uniqueinds(A, Lis)
    dL, dR = dim(Lis), dim(Ris)
    # maxdim is forced to be at most the max given SVD
    if isnothing(maxdim)
      maxdim = min(dL, dR)
    end
    maxdim = min(maxdim, min(dL, dR))
    might_truncate = !isnothing(cutoff) || maxdim < min(dL, dR)
  
  
    if which_decomp == "eigen"
    L, R, spec = factorize_eigen_gen(
      A, Linds...; mindim, maxdim, cutoff, tags, ortho, eigen_perturbation
    )
    elseif which_decomp == "svd"
    L, R, spec = factorize_svd_gen(
      A, Linds...; mindim, maxdim, cutoff, tags, ortho, eigen_perturbation
    )
    else
        @warn No good factorization selected! 
    end

    #println(L)
    #println(R)

    # Set the tags and prime level
    l = commonind(L, R)
    l̃ = setprime(settags(l, tags), plev)
    L = replaceind(L, l, l̃)
    R = replaceind(R, l, l̃)
    l = l̃
  
    return L, R, spec, l
  end
  


# factorize_svd: ITensors.jl/src/tensor_operations/matrix_decomposition.jl:592
function factorize_svd_gen(
    A::ITensor,
    Linds...;
    ortho="left",
    eigen_perturbation=nothing,
    mindim=nothing,
    maxdim=nothing,
    cutoff=nothing,
    tags=nothing,
  )
    # the other way round now ? 
    if ortho == "right"
      Lis = commoninds(A, indices(Linds...))
    elseif ortho == "left"
      Lis = uniqueinds(A, indices(Linds...))
    else
      error("In factorize using eigen decomposition, ortho keyword
      $ortho not supported. Supported options are left or right.")
    end

    println(ortho, Lis)
    # Lis are the indices we do *not* contract on A2 
    #Lis = inds(A,"Link")
    simLis = sim(Lis)
    A2 = A * replaceinds(A, Lis, simLis)
  
    F = eigen_gen(A2, Lis, simLis; ishermitian=false, mindim, maxdim, cutoff, tags)

    # TODO Fix this .. 
    #D, _, spec = F
    D = F.D 
    #@show(D)
    sqD = D.^(0.5)
    isqD = sqD.^(-1)

    #@show sqD, isqD

    spec = F.spec


    Z = (F.Vt * noprime(F.Vt))
    isqZ = diag(Z).^(-0.5)
    O = F.Vt * diag_itensor(isqZ.storage.data, inds(Z))

    L = A * O * isqD
    R = sqD * O 


    if ortho == "right"
      L, R = R, L
    end


    @show ortho
    @show Lis
    @show inds(A)
    @show inds(A2)
    @show inds(O)
    @show inds(sqD)
    @show inds(L)
    @show inds(R)


    return L, R, spec
  end

  

  function factorize_eigen_gen(
    A::ITensor,
    Linds...;
    ortho="left",
    eigen_perturbation=nothing,
    mindim=nothing,
    maxdim=nothing,
    cutoff=nothing,
    tags=nothing,
  )
    if ortho == "left"
      Lis = commoninds(A, indices(Linds...))
    elseif ortho == "right"
      Lis = uniqueinds(A, indices(Linds...))
    else
      error("In factorize using eigen decomposition, ortho keyword
      $ortho not supported. Supported options are left or right.")
    end
    simLis = sim(Lis)
    A2 = A * replaceinds(A, Lis, simLis)
   
    F = eigen_gen(A2, Lis, simLis; ishermitian=false, mindim, maxdim, cutoff, tags)

    # TODO Fix this .. 
    #D, _, spec = F
    D = F.D 
    #@show(diag(D))

    spec = F.spec


    Z = (F.Vt * noprime(F.Vt))
    isqZ = diag(Z).^(-0.5)
    O = F.Vt * diag_itensor(isqZ.storage.data, inds(Z))
   

    L = O
    R = O * A

    #A_rec = L * R 

    if ortho == "right"
      L, R = R, L
    end
    return L, R, spec
  end


  function eigen_gen(
    A::ITensor,
    Linds,
    Rinds;
    mindim=nothing,
    maxdim=nothing,
    cutoff=nothing,
    use_absolute_cutoff=nothing,
    use_relative_cutoff=nothing,
    ishermitian=nothing,
    tags=nothing,
    lefttags=nothing,
    righttags=nothing,
    plev=nothing,
    leftplev=nothing,
    rightplev=nothing,
  )
    ishermitian = NDTensors.replace_nothing(ishermitian, false)
    tags = NDTensors.replace_nothing(tags, ts"Link,eigen")
    lefttags = NDTensors.replace_nothing(lefttags, tags)
    righttags = NDTensors.replace_nothing(righttags, tags)
    plev = NDTensors.replace_nothing(plev, 0)
    leftplev = NDTensors.replace_nothing(leftplev, plev)
    rightplev = NDTensors.replace_nothing(rightplev, plev)
  
    N = ndims(A)
    NL = length(Linds)
    NR = length(Rinds)
    NL != NR && error("Must have equal number of left and right indices")
    N != NL + NR &&
      error("Number of left and right indices must add up to total number of indices")
  
    if lefttags == righttags && leftplev == rightplev
      leftplev = rightplev + 1
    end
  
    # Linds, Rinds may not have the correct directions
    Lis = indices(Linds)
    Ris = indices(Rinds)
  
    # Ensure the indices have the correct directions,
    # QNs, etc.
    # First grab the indices in A, then permute them
    # correctly.
    Lis = permute(commoninds(A, Lis), Lis)
    Ris = permute(commoninds(A, Ris), Ris)
  
    for (l, r) in zip(Lis, Ris)
      if space(l) != space(r)
        error("In eigen, indices must come in pairs with equal spaces.")
      end
      if hasqns(A)
        if dir(l) == dir(r)
          error("In eigen, indices must come in pairs with opposite directions")
        end
      end
    end

    #TODO HACK 
    #CL = combiner(Lis...; dir=Out, tags="CMB,left")
    #CR = combiner(dag(Ris)...; dir=Out, tags="CMB,right")
  
    CL = combiner(Lis...; tags="CMB,left")
    CR = combiner(dag(Ris)...; tags="CMB,right")
  
    AC = A * dag(CR) * CL
  
    cL = combinedind(CL)
    cR = dag(combinedind(CR))
    if inds(AC) != (cL, cR)
      AC = permute(AC, cL, cR)
    end
  
    #AT = ishermitian ? Hermitian(tensor(AC)) : tensor(AC)
    AT = AC.tensor 
  
    DT, VT, spec = eigen_gen(AT; mindim, maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff)
    D, VC = itensor(DT), itensor(VT)
  
    V = VC * dag(CR)
  
    # Set right index tags
    l = uniqueind(D, V)
    r = commonind(D, V)
    l̃ = setprime(settags(l, lefttags), leftplev)
    r̃ = setprime(settags(l̃, righttags), rightplev)
  
    replaceinds!(D, (l, r), (l̃, r̃))
    replaceind!(V, r, r̃)
  
    l, r = l̃, r̃
  
    # The right eigenvectors, after being applied to A
    Vt = replaceinds(V, (Ris..., r), (Lis..., l))

  
    return TruncEigen(D, V, Vt, spec, l, r)
  end
  



function eigen_gen(
    T::NDTensors.DenseTensor{ElT,2,IndsT};
    mindim=nothing,
    maxdim=nothing,
    cutoff=nothing,
    use_absolute_cutoff=nothing,
    use_relative_cutoff=nothing,
  ) where {ElT<:Union{Real,Complex},IndsT}
    matrixT = matrix(T)
    if any(!isfinite, matrixT)
      throw(
        ArgumentError(
          "Trying to perform the eigendecomposition of a matrix containing NaNs or Infs"
        ),
      )
    end
  
    DM, VM = eigen(expose(matrixT))
  
    # Sort by largest to smallest eigenvalues
    p = sortperm(DM; by=abs, rev = true)
    DM = DM[p]
    VM = VM[:,p]
  
    if any(!isnothing, (maxdim, cutoff))
      println("Truncating $maxdim $cutoff")
      DM, truncerr, _ = truncate_gen!!(
        DM; mindim, maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff
      )
      dD = length(DM)
      if dD < size(VM, 2)
        VM = VM[:, 1:dD]
      end
    else
      dD = length(DM)
      truncerr = 0.0
end


    spec = Spectrum(abs.(DM), truncerr)
  
    i1, i2 = inds(T)
  
    # Make the new indices to go onto D and V
    l = typeof(i1)(dD)
    r = dag(sim(l))
    Dinds = (l, r)
    Vinds = (dag(i2), r)
    D = complex(NDTensors.tensor(NDTensors.Diag(DM), Dinds))
    V = complex(NDTensors.tensor(NDTensors.Dense(vec(VM)), Vinds))
    return D, V, spec
  end
  



### From ....
function truncate_gen!!(P::AbstractArray; kwargs...)
    return truncate_gen!!(unwrap_type(P), P; kwargs...)
  end
  
  # CPU version.
  function truncate_gen!!(::Type{<:Array}, P::AbstractArray; kwargs...)
    truncerr, docut = truncate_gen!(P; kwargs...)
    return P, truncerr, docut
  end

  function truncate_gen!( 
    P::AbstractVector;
    mindim=nothing,
    maxdim=nothing,
    cutoff=nothing,
    use_absolute_cutoff=nothing,
    use_relative_cutoff=nothing,
  )
    mindim = replace_nothing(mindim, default_mindim(P))
    maxdim = replace_nothing(maxdim, length(P))
    cutoff = replace_nothing(cutoff, typemin(Float64))
    use_absolute_cutoff = replace_nothing(use_absolute_cutoff, default_use_absolute_cutoff(P))
    use_relative_cutoff = replace_nothing(use_relative_cutoff, default_use_relative_cutoff(P))
  
    origm = length(P)
    docut = zero(eltype(P))
  
    # #if P[1] <= 0.0
    # #  P[1] = 0.0
    # #  resize!(P, 1)
    # #  return 0.0, 0.0
    # #end
  
    # if origm == 1
    #   docut = abs(P[1]) / 2
    #   return zero(eltype(P)), docut
    # end
  
    # s = sign(P[1])
    # s < 0 && (P .*= s)
  
    # #Zero out any negative weight
    # for n in origm:-1:1
    #   (P[n] >= zero(eltype(P))) && break
    #   P[n] = zero(eltype(P))
    # end
  
    # n = origm
    # truncerr = zero(eltype(P))
    # while n > maxdim
    #   truncerr += P[n]
    #   n -= 1
    # end
  
    # if use_absolute_cutoff
    #   #Test if individual prob. weights fall below cutoff
    #   #rather than using *sum* of discarded weights
    #   while P[n] <= cutoff && n > mindim
    #     truncerr += P[n]
    #     n -= 1
    #   end
    # else
    #   scale = one(eltype(P))
    #   if use_relative_cutoff
    #     scale = sum(P)
    #     (scale == zero(eltype(P))) && (scale = one(eltype(P)))
    #   end
  
    #   #Continue truncating until *sum* of discarded probability 
    #   #weight reaches cutoff reached (or m==mindim)
    #   while (truncerr + P[n] <= cutoff * scale) && (n > mindim)
    #     truncerr += P[n]
    #     n -= 1
    #   end
  
    #   truncerr /= scale
    # end
  
    # if n < 1
    #   n = 1
    # end
  

    # TODO HACK 
    # n = minimum([origm, 100])
    # truncerr = 1e-10

    n, truncerr  = mytruncate_eig(P, cutoff , maxdim)

    if n < origm
      docut = (P[n] + P[n + 1]) / 2
      # if abs(P[n] - P[n + 1]) < eltype(P)(1e-3) * P[n]
      #   docut += eltype(P)(1e-3) * P[n]
      # end
    end
  
    # s < 0 && (P .*= s)


    resize!(P, n)
    return truncerr, docut
  end







# Index stuff (?) copied from ITensors source without changes


_indices() = ()
_indices(x::Index) = (x,)

# Tuples
_indices(x1::Tuple, x2::Tuple) = (x1..., x2...)
_indices(x1::Index, x2::Tuple) = (x1, x2...)
_indices(x1::Tuple, x2::Index) = (x1..., x2)
_indices(x1::Index, x2::Index) = (x1, x2)

# Vectors
_indices(x1::Vector, x2::Vector) = narrow_eltype(vcat(x1, x2); default_empty_eltype=Index)

# Mix vectors and tuples/elements
_indices(x1::Vector, x2) = _indices(x1, [x2])
_indices(x1, x2::Vector) = _indices([x1], x2)
_indices(x1::Vector, x2::Tuple) = _indices(x1, [x2...])
_indices(x1::Tuple, x2::Vector) = _indices([x1...], x2)

indices(x::Vector{Index{S}}) where {S} = x
indices(x::Vector{Index}) = narrow_eltype(x; default_empty_eltype=Index)
indices(x::Tuple) = reduce(_indices, x; init=())
indices(x::Vector) = reduce(_indices, x; init=Index[])
indices(x...) = indices(x)



leftlim(m::MPS) = m.llim

rightlim(m::MPS) = m.rlim


function setleftlim!(m::MPS, new_ll::Integer)
    return m.llim = new_ll
  end
  
  function setrightlim!(m::MPS, new_rl::Integer)
    return m.rlim = new_rl
  end