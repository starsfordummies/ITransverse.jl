
"""
    replacebond_gen!(M::MPS, b::Int, phi::ITensor; kwargs...)

Factorize the ITensor `phi` and replace the ITensors
`b` and `b+1` of MPS `M` with the factors. Choose
the orthogonality with `ortho="left"/"right"`.
"""
function replacebond_gen!(
  M::gMPS,
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
  #svd_alg=nothing,
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
  #@show( b, linkind(M, b) )
  #@show linkinds(M)
  L, R, spec = factorize_gen(
    phi,
    indsMb;
    mindim,
    maxdim,
    cutoff,
    ortho,
    #which_decomp,
    eigen_perturbation,
    #svd_alg,
    tags=tags(linkind(M, b)),
    which_decomp,
  )
  M[b] = L
  M[b + 1] = R

  #@show inds(M[b])
  #@show inds(M[b+1])
  
  if ortho == "left"
    leftglim(M) == b - 1 && setleftglim!(M, leftglim(M) + 1)
    rightglim(M) == b + 1 && setrightglim!(M, rightglim(M) + 1)
    normalize && (M[b + 1] ./= norm_gen(M[b + 1]))
  elseif ortho == "right"
    leftglim(M) == b && setleftglim!(M, leftglim(M) - 1)
    rightglim(M) == b + 2 && setrightglim!(M, rightglim(M) - 1)
    normalize && (M[b] ./= norm_gen(M[b]))
  else
    error(
      "In replacebond_gen!, got ortho = $ortho, only currently supports `left` and `right`."
    )
  end
  return spec
end

"""
    replacebond_gen(M::MPS, b::Int, phi::ITensor; kwargs...)

Like `replacebond_gen!`, but returns the new MPS.
"""
function replacebond_gen(M0::gMPS, b::Int, phi::ITensor; kwargs...)
  M = copy(M0)
  replacebond_gen!(M, b, phi; kwargs...)
  return M
end

# Allows overloading `replacebond_gen!` based on the projected
# MPO type. By default just calls `replacebond_gen!` on the MPS.
function replacebond_gen!(PH, M::gMPS, b::Int, phi::ITensor; kwargs...)
  return replacebond_gen!(M, b, phi; kwargs...)
end
