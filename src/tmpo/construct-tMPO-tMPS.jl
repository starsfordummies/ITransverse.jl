# using ITensorMPS, ITensors
# using NDTensors: @Algorithm_str
using ITensors.SiteTypes: SiteTypes, siteind, siteinds, state
# using ITensors: Algorithm, contract, hassameinds, inner, mapprime
#import ITensorMPS: maxlinkdim
#import ITensorMPS: replace_siteinds, replace_siteinds!

function delete_link_from_prodMPS(ψ)
  @assert maxlinkdim(ψ) == 1
  N = length(ψ)
  links = linkinds(ψ)
  new_ψ = similar(ψ)
  new_ψ[1] = ψ[1] * dag(onehot(links[1] => 1))
  for ii in 2:N-1
    new_ψ[ii] = onehot(links[ii-1] => 1) * ψ[ii] * dag(onehot(links[ii] => 1))
  end
   new_ψ[N] = onehot(links[N-1] => 1) * ψ[N]
  return new_ψ
end


function delete_link_from_prodMPS!(psi::AbstractMPS)
  if maxlinkdim(psi) == 1
    ss = siteinds(psi)
    for ii in eachindex(psi)
      psi[ii] = ITensor(array(psi[ii]), ss[ii])
    end
  end
end
 

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
function construct_tMPS_tMPO(ψ_i::MPS, Ut::Vector{MPO}, ϕ_f::MPS;
  dagger_final::Bool=true,
  return_swapped_T::Bool=true,
  )
  if hasqns(ψ_i)
    if maxlinkdim(ψ_i) > 1 || maxlinkdim(ϕ_f) > 1
      throw(ArgumentError("For now, we cannot process a QN-preserving initial/final state that is NOT a product state!"))
    end
  end
  if maxlinkdim(ψ_i) == 1 && maxlinkdim(ϕ_f) == 1
    ψi = delete_link_from_prodMPS(ψ_i)
    ϕf = delete_link_from_prodMPS(ϕ_f)

    Ut_1 = noprime.(ψi.data .* Ut[1].data)
    Ut_end = dagger_final ? noprime.(Ut[end].data .* prime.(dag.(ϕf.data))) : noprime.(Ut[end].data .* prime.(ϕf.data))

    input = if length(Ut)-1 >= 2 
      hcat(Ut_1, hcat([(Ut[ii]).data for ii in 2:length(Ut)-1]...), Ut_end)
    else
      hcat(Ut_1, Ut_end)
    end

    return construct_tMPS_tMPO(input; dagger_final=false, return_swapped_T )
  else
    input = hcat(
      ψ_i.data,
      hcat([Ut_ele.data for Ut_ele in Ut]...),
      ϕ_f.data
    )
    return construct_tMPS_tMPO(input; dagger_final=dagger_final, return_swapped_T )
  end
end


function construct_tMPS_tMPO(input::Matrix{ITensor};
  dagger_final::Bool=true,
  return_swapped_T::Bool=true,
  )
  nrows, ncolumns = size(input)
  @assert nrows == 3 "Assume input Matrix to have 3 rows!"


  links_cols = hcat([linkinds(input[:, n]) for n in 1:ncolumns]...)
  sites_cols = hcat([siteinds(input[:, n]) for n in 1:ncolumns]...)

  links_tMPS_L = [sim(only(sites_cols[1, 1]), tags="Link,time,nₜ=$(n-1)") for n in 1:ncolumns-1]
  links_tMPO =   [sim(only(sites_cols[2, 1]), tags="Link,time,nₜ=$(n-1)") for n in 1:ncolumns-1]
  links_tMPS_R = [sim(only(sites_cols[3, 1]), tags="Link,time,nₜ=$(n-1)") for n in 1:ncolumns-1]

  # the new sites are a bit special, the first (and perhaps last) site is (are) given by the boundary MPS
  # while the bulk of the new sites are given by the old link of the time evolution operator
  sites = [sim(links_cols[1, n], tags="Site,time,nₜ=$(n-1)") for n in 1:ncolumns]
  # we have now created all the new indices for the new tMPSs and tMPOs and effectively
  # rotated the network by relabelling sites with links and links with sites.


  ### first row, MPS left state ⟨L|
  first_L = replaceinds(
    input[1, 1],
    links_cols[1, 1] => sites[1],
    only(sites_cols[1, 1]) => links_tMPS_L[1],
  )
  bulk_L = [
    replaceinds(
      input[1, n],
      links_cols[1, n] => sites[n],
      sites_cols[1, n][1] => links_tMPS_L[n],
      sites_cols[1, n][2] => dag(links_tMPS_L[n-1]),
    ) for n in 2:ncolumns-1 # difference to no MPS boundary at end!
  ]
  last_L_tensor = dagger_final ? conj(input[1, end]) : input[1, end]
  last_L = replaceinds(
    last_L_tensor,
    links_cols[1, end] => sites[end],
    only(sites_cols[1, end]) => dag(links_tMPS_L[end]),
  )

  ### second row, MPO transfer matrix column TL,
  ### corresponding to the left state ⟨L|
  first_MPO = replaceinds(
    input[2, 1],
    dag(links_cols[1, 1]) => dag(sites[1]),
    links_cols[2, 1] => prime(sites[1]),
    only(sites_cols[2, 1]) => links_tMPO[1]
  )
  bulk_MPO = [replaceinds(
    input[2, n],
    dag(links_cols[1, n]) => dag(sites[n]),
    links_cols[2, n] => prime(sites[n]),
    sites_cols[2, n][1] => links_tMPO[n],
    sites_cols[2, n][2] => dag(links_tMPO[n-1]),
  ) for n in 2:ncolumns-1 # difference to no MPS boundary at end!
  ]
  last_TL_tensor = dagger_final ? conj(input[2, end]) : input[2, end]
  last_MPO = replaceinds(
    last_TL_tensor,
    dag(links_cols[1, end]) => dag(sites[end]),
    links_cols[2, end] => prime(sites[end]),
    only(sites_cols[2, end]) => dag(links_tMPO[end]),
  )

  ### third row, MPS right state |R⟩
  first_R = replaceinds(
    input[3, 1],
    dag(links_cols[2, 1]) => dag(sites[1]),
    only(sites_cols[3, 1]) => links_tMPS_R[1]
  )
  bulk_R = [replaceinds(
    input[3, n],
    dag(links_cols[2, n]) => dag(sites[n]),
    sites_cols[3, n][1] => links_tMPS_R[n],
    sites_cols[3, n][2] => dag(links_tMPS_R[n-1]),
  ) for n in 2:ncolumns-1
  ]
  last_R_tensor = dagger_final ? conj(input[3, end]) : input[3, end]
  last_R = replaceinds(
    last_R_tensor,
    dag(links_cols[2, end]) => dag(sites[end]),
    only(sites_cols[3, end]) => dag(links_tMPS_R[end]),
  )

  T = MPO([first_MPO, bulk_MPO..., last_MPO])
  if return_swapped_T
    return (
      MPS([first_L, bulk_L..., last_L]),
      T,
      swapprime(T,0,1),
      MPS([first_R, bulk_R..., last_R])
    )
  else
    return (
      MPS([first_L, bulk_L..., last_L]),
      T,
      MPS([first_R, bulk_R..., last_R])
    )
  end
end


function ITensorMPS.linkind(M::AbstractVector{ITensor}, j::Integer)
  N = length(M)
  (j ≥ length(M) || j < 1) && return nothing
  return commonind(M[j], M[j+1])
end

function ITensorMPS.linkinds(M::AbstractVector{ITensor}, j::Integer)
  N = length(M)
  (j ≥ length(M) || j < 1) && return IndexSet()
  return commoninds(M[j], M[j+1])
end

ITensorMPS.linkinds(ψ::AbstractVector{ITensor}) = [ITensorMPS.linkind(ψ, b) for b in 1:(length(ψ)-1)]

function ITensorMPS.linkinds(::typeof(all), ψ::AbstractVector{ITensor})
  return IndexSet[ITensorMPS.linkinds(ψ, b) for b in 1:(length(ψ)-1)]
end


function SiteTypes.siteind(::typeof(first), M::AbstractVector{ITensor}, j::Integer; kwargs...)
  N = length(M)
  (N == 1) && return firstind(M[1]; kwargs...)
  if j == 1
    si = uniqueind(M[j], M[j+1]; kwargs...)
  elseif j == N
    si = uniqueind(M[j], M[j-1]; kwargs...)
  else
    si = uniqueind(M[j], M[j-1], M[j+1]; kwargs...)
  end
  return si
end


function SiteTypes.siteinds(M::AbstractVector{ITensor}, j::Integer; kwargs...)
  N = length(M)
  (N == 1) && return inds(M[1]; kwargs...)
  if j == 1
    si = uniqueinds(M[j], M[j+1]; kwargs...)
  elseif j == N
    si = uniqueinds(M[j], M[j-1]; kwargs...)
  else
    si = uniqueinds(M[j], M[j-1], M[j+1]; kwargs...)
  end
  return si
end

function SiteTypes.siteinds(::typeof(all), ψ::AbstractVector{ITensor}, n::Integer; kwargs...)
  return siteinds(ψ, n; kwargs...)
end

function SiteTypes.siteinds(::typeof(first), ψ::AbstractVector{ITensor}; kwargs...)
  return [siteind(first, ψ, j; kwargs...) for j in 1:length(ψ)]
end

function SiteTypes.siteinds(::typeof(only), ψ::AbstractVector{ITensor}; kwargs...)
  return [siteind(only, ψ, j; kwargs...) for j in 1:length(ψ)]
end

function SiteTypes.siteinds(::typeof(all), ψ::AbstractVector{ITensor}; kwargs...)
  return [siteinds(ψ, j; kwargs...) for j in 1:length(ψ)]
end

function SiteTypes.siteinds(ψ::AbstractVector{ITensor}; kwargs...)
  return [siteinds(all, ψ, j; kwargs...) for j in 1:length(ψ)]
end


function ITensorMPS.maxlinkdim(M::AbstractVector{ITensor})
    md = 1
    for b in eachindex(M)[1:(end - 1)]
        l = linkind(M, b)
        linkdim = isnothing(l) ? 1 : dim(l)
        md = max(md, linkdim)
    end
    return md
end




""" Alternative version. Does some basic siteinds matching if the MPS/MPO do not have the same sets.
Ideally in the future we may want to feed a common indexset for them to share """
function construct_tMPS_tMPO_2(psi_i::MPS, in_Uts::Vector{MPO}, psi_f::MPS;
  return_swapped_T::Bool=false,
  )

  psi_i = replace_siteinds(psi_i, firstsiteinds(in_Uts[1]))
  psi_f = replace_siteinds(psi_f, firstsiteinds(in_Uts[end]))

  @assert siteinds(psi_i) == firstsiteinds(in_Uts[1])

  siteinds_Ut = [firstsiteinds(Ut) for Ut in in_Uts]
  if length(unique(siteinds_Ut)) > 1
    for (ii,Ut) in enumerate(in_Uts)
      in_Uts[ii] = replace_siteinds(Ut, siteinds_Ut[1])
    end
  end

  @assert siteinds(psi_f) == firstsiteinds(in_Uts[1])
  @assert length(psi_i) == 3 

  Nrows = length(in_Uts)

  Uts = sim.(linkinds, in_Uts)

  #Incorporate initial and final state in MPOs. First remove trivial links for QN mental sanity
  ITransverse.delete_link_from_prodMPS!(psi_i)
  ITransverse.delete_link_from_prodMPS!(psi_f)

  Uts[1] = applyn(Uts[1], psi_i)
  Uts[end] = applyns(Uts[end], dag(psi_f))

  for ii = 3:Nrows
    Uts[ii] = prime(siteinds, Uts[ii] ,ii-2)
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