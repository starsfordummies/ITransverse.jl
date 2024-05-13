
#=
Should extend matrix_decomposition.jl


For LEFT >>> sweep:

method 1 [no inverse required] : diagonalize     

        |            _
      --O-\       ==| |          O   D  O^T             O
          |   =>    | |   =>   ==|>--o--<|==  , take   =|>-  as new Ai 
      --O-/       ==|_|
        |


method 2 [require inverting eigenvalues]: diagonalize
                _
/-O--          | |--                                 |
| |       =>   | |        => --|>==⊘==<|--  , take  -O--<|-1/sqrt(⊘))-- as new Ai
\-O--          |_|--
  
=#




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

  if which_decomp == "gen_one"
    L, R, spec = factorize_twosite_gen_one(
      A,
      Linds...;
      ortho,
      maxdim,
      cutoff,
      tags,
      use_absolute_cutoff,
      use_relative_cutoff
    )

  elseif which_decomp == "gen_two"
    L, R, spec = factorize_twosite_gen_two(
      A,
      Linds...;
      ortho, 
      maxdim,
      cutoff,
      tags,
      nsites="two"
    )

  elseif which_decomp == "gen_svd"
    L, R, spec = factorize_twosite_gen_svd(
      A,
      Linds...;
      ortho, # ="left"??
      maxdim,
      cutoff,
      tags,
      nsites="two"
    )
  else
    error("pick a which_decomp (gen_one/gen_two/gen_svd) ")
  end

  # Set the tags and prime level
  l = commonind(L, R)
  l̃ = setprime(settags(l, tags), plev)
  L = replaceind(L, l, l̃)
  R = replaceind(R, l, l̃)
  l = l̃

  @debug("Checking if factorization is good")
  
  if norm(L * R - A)/norm(A) > 1e-6
    @warn("Truncation is large ?!  $(norm(L * R - A)/norm(A))")
    @show which_decomp
  end

  return L, R, spec, l
end



""" Copied from factorize_eigen - but shouldn't this also work for the onesite ? CHECK"""
function factorize_twosite_gen_one(
  A::ITensor,
  Linds...;
  ortho="left",
  eigen_perturbation=nothing,
  mindim=nothing,
  maxdim=nothing,
  cutoff=nothing,
  tags=nothing,
  use_absolute_cutoff=nothing,
  use_relative_cutoff=nothing,
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


  F = _ortho_eigen(
    A2,
    Lis,
    simLis;
    ishermitian=false,
    mindim,
    maxdim,
    cutoff,
    tags
  )
  @show F

  F2 = symm_oeig(A2, Lis; cutoff, maxdim) # TODO , tags)

  @show F 
  #D, _, spec = F
  L = F.Vt
  R = (L) * A
  if ortho == "right"
    L, R = R, L
  end
  return L, R, F.spec
end



function factorize_twosite_gen_two(
  A::ITensor,
  Linds...;
  ortho,
  eigen_perturbation=nothing,
  mindim=nothing,
  maxdim=nothing,
  cutoff=nothing,
  tags=nothing,
  nsites
  )

  # ! CHECK
  # ? Do we need this ? Does this even work for onesite ?
  if nsites=="one"

    Lis = uniqueinds(A, indices(Linds...))

  elseif nsites=="two"

    if ortho == "right"
      Lis = commoninds(A, indices(Linds...))
    elseif ortho == "left"
      Lis = uniqueinds(A, indices(Linds...))
    else
      error("In factorize using eigen decomposition, ortho keyword
      $ortho not supported. Supported options are left or right.")
    end

  end

  simLis = sim(Lis)
  A2 = A * replaceinds(A, Lis, simLis)



  #@show inds(A2)
  #@show Lis, simLis

  #F = ortho_eigen(A2, Lis, simLis; ishermitian=false, mindim, maxdim, cutoff, tags)
  F = symm_oeig(A2, Lis; cutoff, maxdim) # TODO , tags)


  sqD = F.D.^(0.5)
  isqD = diagITensor(vector(diag(pinv(sqD.tensor))), inds(sqD))

  spec = F.spec

  O = F.V # TODO or Vt ??

  L = A * O * isqD
  R = sqD * O 


  if ortho == "right"
    L, R = R, L
  end

  return L, R, spec
end



function factorize_twosite_gen_svd(
  A::ITensor,
  Linds...;
  ortho,
  mindim=nothing,
  maxdim=nothing,
  cutoff=nothing,
  tags=nothing,
  nsites="two"
  )

  # ! CHECK
  # ? Do we need this ? Does this even work for onesite ?
  if nsites=="one"

    Lis = uniqueinds(A, indices(Linds...))

  elseif nsites=="two"

    if ortho == "right"
      Lis = commoninds(A, indices(Linds...))
    elseif ortho == "left"
      Lis = uniqueinds(A, indices(Linds...))
    else
      error("In factorize using eigen decomposition, ortho keyword
      $ortho not supported. Supported options are left or right.")
    end

  end

  simLis = sim(Lis)
  A2 = A * replaceinds(A, Lis, simLis)

  F = symm_svd(A2, Lis; mindim, maxdim, cutoff, tags)

  sqS = F.S.^(0.5)
  isqS = diagITensor(vector(diag(pinv(sqS.tensor))), inds(sqS))

  spec = F.spec

  O = F.U

  L = A * dag(O) * isqS
  R = sqS * O 


  if ortho == "right"
    L, R = R, L
  end

  return L, R, spec
end


