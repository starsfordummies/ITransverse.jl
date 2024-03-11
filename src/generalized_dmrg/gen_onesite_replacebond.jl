include("./gen_replacebond_2.jl")
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
function replacebond_onesite_gen!(PH, M::MPS, b::Int, phi::ITensor; kwargs...)
    return replacebond_onesite_gen!(M, b, phi; kwargs...)
 end

  function replacebond_onesite_gen!(
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


    rhoi = phi*phi
    leftind = linkind(M,b-1)
    rightind = linkind(M,b)

    indslink = (leftind,rightind)

    # L, R, spec = factorize_onesite_gen(
    #   phi,
    #   indslink;
    #   mindim,
    #   maxdim,
    #   cutoff,
    #   ortho,
    #   which_decomp,
    #   eigen_perturbation,
    #   svd_alg,
    #   tags=tags(linkind(M, b)),
    # )

    # just put it all here for now 
      
    # the other way round now ? 
    if ortho == "left"
      Lis = rightind
      currid = b
    elseif ortho == "right"
      Lis = leftind
      currid = b-1
    else
      error("In factorize using eigen decomposition, ortho keyword
      $ortho not supported. Supported options are left or right.")
    end

    # Lis are the indices we do *not* contract on A2 
    simLis = sim(Lis)
    A = phi
    A2 = A * replaceind(A, Lis, simLis)

  # do not truncate
  mindim=nothing
  maxdim=nothing
  #println(tags)

  ttags = tags(linkind(M, currid))
  pplev = 0

  F = eigen_gen(A2, Lis, simLis; ishermitian=false, mindim, maxdim, cutoff, tags=ttags)

  # TODO Fix this .. 
  #D, _, spec = F
  D = F.D 
  #@show(D)
  sqD = D.^(0.5)
  isqD = sqD.^(-1)

  #isqD = diagITensor(, inds(sqD))
  #isqD = diagITensor(vec([d for d in pinv(diag(sqD).storage)]), inds(sqD))
  isqD = diagITensor(vector(diag(pinv(sqD.tensor))), inds(sqD))

  #@show sqD, isqD

  spec = F.spec


  Z = (F.Vt * noprime(F.Vt))
  isqZ = diag(Z).^(-0.5)
  O = F.Vt * diagITensor(isqZ.storage.data, inds(Z))

  if ortho == "left"
    L = A * O * isqD
    R = sqD * O * M[b+1]

      # Set the tags and prime level
      l = commonind(L, R)
      l̃ = setprime(settags(l, ttags), pplev)
      L = replaceind(L, l, l̃)
      R = replaceind(R, l, l̃)


    println("Before")
    println(inds(M[b]))
    println(inds(M[b+1]))

    M[b] = (L)
    M[b + 1] = (R)
    
    println("After")
    println(inds(M[b]))
    println(inds(M[b+1]))

  elseif ortho == "right"
    R = A * O * isqD
    L = sqD * O * M[b-1]

      # Set the tags and prime level
      l = commonind(L, R)
      l̃ = setprime(settags(l, ttags), pplev)
      L = replaceind(L, l, l̃)
      R = replaceind(R, l, l̃)

 
    M[currid] = (L)
    M[currid+1] = (R)
  end


    #println("Checking mix ortho @ $b, $ortho ")




    if ortho == "left"
      leftlim(M) == b - 1 && setleftlim!(M, leftlim(M) + 1)
      rightlim(M) == b + 1 && setrightlim!(M, rightlim(M) + 1)
      normalize && (M[b + 1] ./= sqrt(scalar(M[b + 1]*M[b + 1])))

       #@show norm(M[b+1])
       #@show scalar(M[b+1] * M[b+1])
    
    elseif ortho == "right"
      leftlim(M) == b && setleftlim!(M, leftlim(M) - 1)
      rightlim(M) == b + 2 && setrightlim!(M, rightlim(M) - 1)
      normalize && (M[b-1] ./= sqrt(scalar(M[b-1] * M[b-1])))
    else
      error(
        "In replacebond!, got ortho = $ortho, only currently supports `left` and `right`."
      )
    end

    return spec
  end

  








function factorize_onesite_gen(
    A::ITensor,
    link_inds;
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
  
    ortho = NDTensors.replace_nothing(ortho, "left")
    tags = NDTensors.replace_nothing(tags, ts"Link,fact")
    plev = NDTensors.replace_nothing(plev, 0)
  
    # Determines when to use eigen vs. svd (eigen is less precise,
    # so eigen should only be used if a larger cutoff is requested)
    automatic_cutoff = 1e-12
    Lis = link_inds[1]
    Ris = link_inds[2]

    if isnothing(Lis)
      dL = 1
    else
    dL = dim(Lis)
    end
    if isnothing(Ris)
      dR = 1
    else
      dR = dim(Ris)
    end

    # maxdim is forced to be at most the max given SVD
    if isnothing(maxdim)
      maxdim = min(dL, dR)
    end
    maxdim = min(maxdim, min(dL, dR))
    might_truncate = !isnothing(cutoff) || maxdim < min(dL, dR)
  
  
    L, R, spec = factorize_onesite_svd_gen(
      A, link_inds...; mindim, maxdim, cutoff, tags, ortho, eigen_perturbation
    )
    

    # Set the tags and prime level
    l = commonind(L, R)
    l̃ = setprime(settags(l, tags), plev)
    L = replaceind(L, l, l̃)
    R = replaceind(R, l, l̃)
    l = l̃
  
    return L, R, spec, l
  end
  


# factorize_svd: ITensors.jl/src/tensor_operations/matrix_decomposition.jl:592
function factorize_onesite_svd_gen(
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
  if ortho == "left"
    Lis = Linds[2]
  elseif ortho == "right"
    Lis = Linds[1]
  else
    error("In factorize using eigen decomposition, ortho keyword
    $ortho not supported. Supported options are left or right.")
  end

  # Lis are the indices we do *not* contract on A2 
  #Lis = inds(A,"Link")
  simLis = sim(Lis)
  A2 = A * replaceind(A, Lis, simLis)

  # do not truncate
  mindim=nothing
  maxdim=nothing
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
  O = F.Vt * diagITensor(isqZ.storage.data, inds(Z))

  L = A * O * isqD
  R = sqD * O 


  if ortho == "right"
    L, R = R, L
  end
  return L, R, spec
end



