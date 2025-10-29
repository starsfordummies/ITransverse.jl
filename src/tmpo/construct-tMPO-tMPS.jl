

"""
    construct_tMPS_tMPO(ψ_i::MPS, Ut::Vector{MPO}, ϕ_f::MPS)
```
|ϕ_f⟩    o———o———o
         |   |   |
Ut[end]  U———U———U
         |   |   |
Ut[…]          ⋮
         |   |   |
Ut[2]    U———U———U
         |   |   |  time
Ut[1]    U———U———U    ↑
         |   |   |    |
⟨ψ_i|    o———o———o    |
                      |
        ⟨L|  T  |R⟩
```

Construct two boundary tMPS (⟨L| and |R⟩) and two tMPOs (TL and TR) for further use in the power method.
- `TR = swapprime(TL, 0, 1)`.
- We assume the input states to have 3 spatial sites!
- `ψ_i` and `ϕ_f` are assumed to be a valid (space-like) MPS
- each element of the vector `Ut` is assumed to be a valid MPO whereas the links match, respectively,
- The (physical) sites are not necessarily(!) correctly primed to link up in the time direction.
- The final MPS will be automatically daggered! If you do not want it to be daggered,
  use the flag `dagger_final=false`.
- If you not wish to return TL and TR and only TL, use `return_swapped_T=false`.
"""
function construct_tMPS_tMPO(psi_i::MPS, in_Uts::Vector{MPO}, psi_f::MPS;
  return_swapped_T::Bool=false,
)

  psi_i = replace_siteinds(psi_i, firstsiteinds(in_Uts[1]))
  psi_f = replace_siteinds(psi_f, firstsiteinds(in_Uts[end]))

  @assert siteinds(psi_i) == firstsiteinds(in_Uts[1])

  siteinds_Ut = [firstsiteinds(Ut) for Ut in in_Uts]
  if length(unique(siteinds_Ut)) > 1
    for (ii, Ut) in enumerate(in_Uts)
      in_Uts[ii] = replace_siteinds(Ut, siteinds_Ut[1])
    end
  end

  @assert siteinds(psi_f) == firstsiteinds(in_Uts[1])
  @assert length(psi_i) == 3

  Nrows = length(in_Uts)

  Uts = sim.(linkinds, in_Uts)


  #Incorporate initial and final state in MPOs. First remove trivial links for QN mental sanity
  maxlinkdim(psi_i) == 1 && ITransverse.delete_link_from_prodMPS!(psi_i)
  maxlinkdim(psi_i) == 1 && ITransverse.delete_link_from_prodMPS!(psi_f)


  Uts[1] = applyn(Uts[1], psi_i)
  Uts[end] = applyns(Uts[end], dag(psi_f))

  for ii = 3:Nrows
    Uts[ii] = prime(siteinds, Uts[ii], ii - 2)
  end

  Lcol_data = [Uts[ii][1] for ii = (1:Nrows)]
  Ccol_data = [Uts[ii][2] for ii = (1:Nrows)]
  Rcol_data = [Uts[ii][end] for ii = (1:Nrows)]

  psiL = MPS(Lcol_data)
  Tc = MPO(Ccol_data)
  psiR = MPS(Rcol_data)

  # Now the mess: relabel the indices 

  ssL = siteinds(psiL)
  ssLn = [noprime(sim(ssL[ii], tags="Site,nt=$(ii)")) for ii = 1:length(ssL)]
  ssRn = dag(ssLn)
  ssR = siteinds(psiR)

  for ii in eachindex(ssL)
    psiL[ii] = replaceind(psiL[ii], ssL[ii] => ssLn[ii])
    Tc[ii] = replaceinds(Tc[ii], ssL[ii] => ssLn[ii]', ssR[ii] => ssRn[ii])
    psiR[ii] = replaceind(psiR[ii], ssR[ii] => ssRn[ii])
  end

  for col = [psiL, Tc, psiR]
    ll = linkinds(col)
    for ii in eachindex(ll)
      lln = noprime(sim(ll[ii], tags="Link,lt=$(ii)"))
      col[ii] = replaceind(col[ii], ll[ii] => lln)
      col[ii+1] = replaceind(col[ii+1], ll[ii] => dag(lln))
    end
  end

  if return_swapped_T
    return psiL, swapprime(Tc, 0, 1), Tc, psiR
  else
    return psiL, Tc, psiR
  end
end

