""" My modification of ITensors' truncate!() allowing for complex values. 
We truncate on the absolute value of the spectrum
"""
function ctruncate!(
    P::AbstractVector;
    mindim=nothing,
    maxdim=nothing,
    cutoff=nothing,
    use_absolute_cutoff=nothing,
    use_relative_cutoff=nothing,
)

@info "ctruncate"
  mindim = replace_nothing(mindim, 1)
  maxdim = replace_nothing(maxdim, length(P))
  #cutoff = replace_nothing(cutoff, typemin(eltype(P)))
  cutoff = replace_nothing(cutoff, 0.)

  use_absolute_cutoff = replace_nothing(use_absolute_cutoff, default_use_absolute_cutoff(P))
  use_relative_cutoff = replace_nothing(use_relative_cutoff, default_use_relative_cutoff(P))

  origm = length(P)
  docut = zero(eltype(P))

  #if P[1] <= 0.0
  #  P[1] = 0.0
  #  resize!(P, 1)
  #  return 0.0, 0.0
  #end
  #
 
  if origm == 1
    docut = abs(P[1]) / 2
    return zero(eltype(P)), docut
  end


  n = origm
  #truncerr = zero(eltype(P))
  truncerr = zero(Float64)
  while n > maxdim
    truncerr += abs(P[n]) #P[n]
    n -= 1
  end

  if use_absolute_cutoff
    #Test if individual prob. weights fall below cutoff
    #rather than using *sum* of discarded weights
    while P[n] <= cutoff && n > mindim
      truncerr += P[n]
      n -= 1
    end
  else
      scale = one(eltype(P))
    if use_relative_cutoff
      scale = sum(abs.(P))  #sum(P)
      (scale == zero(eltype(P))) && (scale = one(eltype(P)))
    end

    #Continue truncating until *sum* of discarded probability
    #weight reaches cutoff reached (or m==mindim)
    # while (truncerr + P[n] <= cutoff * scale) && (n > mindim)
    #   truncerr += P[n]
    #   n -= 1
    # end
    while (truncerr + abs(P[n]) <= cutoff * scale) && (n > mindim)
      truncerr += abs(P[n])
      n -= 1
    end

    truncerr /= scale
  end

  if n < 1
    n = 1
  end

  if n < origm
    docut = (P[n] + P[n + 1]) / 2
    if abs(P[n] - P[n + 1]) < (1e-3)* abs(P[n]) #eltype(P)(1e-3) * P[n]
      docut += eltype(P)(1e-3) * P[n]
    end
  end

  #@show n

  #s < 0 && (P .*= s)
  resize!(P, n)
    return truncerr, docut
end

