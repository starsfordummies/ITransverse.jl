function LinearAlgebra.eigen(
  T::DenseTensor{ElT,2,IndsT};
  mindim=nothing,
  maxdim=nothing,
  cutoff=nothing,
  use_absolute_cutoff=nothing,
  use_relative_cutoff=nothing,
) where {ElT<:Complex,IndsT}
  matrixT = matrix(T)
  if any(!isfinite, matrixT)
    throw(
      ArgumentError(
        "Trying to perform the eigendecomposition of a matrix containing NaNs or Infs $matrixT"
      ),
    )
  end

  #@info "ceigen"
  DM, VM = eigen(expose(matrixT))

  # Sort by largest to smallest eigenvalues
  p = sortperm(DM; by=abs, rev = true)
  DM = DM[p]
  VM = VM[:,p]

  if any(!isnothing, (maxdim, cutoff))
    #println("TRUNCATING @ $maxdim, $cutoff, last eig  = $(DM[end]) ")
    DM, truncerr, _ = truncate!!(
      DM; mindim, maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff
    )
    dD = length(DM)
    if dD < size(VM, 2)
      VM = VM[:, 1:dD]
    end
  else
       #println("**NOT**TRUNCATING @ $maxdim, $cutoff,  last eig  = $(DM[end])  ")
    dD = length(DM)
    truncerr = 0.0
  end

  # TODO it seems that truncate!! can return complex truncerr for corner cases
  spec = 0
  try
    spec = Spectrum(abs.(DM), abs(truncerr))
  catch e
    println("not good, $e, $(abs.(DM)), $truncerr")
  end

  i1, i2 = inds(T)

  # Make the new indices to go onto D and V
  l = typeof(i1)(dD)
  r = dag(sim(l))
  Dinds = (l, r)
  Vinds = (dag(i2), r)
  D = complex(tensor(Diag(DM), Dinds))
  V = complex(tensor(Dense(vec(VM)), Vinds))
  return D, V, spec
end
