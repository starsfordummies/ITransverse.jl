using ITensorMPS, ITensors
# using NDTensors: @Algorithm_str
using ITensors.SiteTypes: SiteTypes, siteind, siteinds, state
# using ITensors: Algorithm, contract, hassameinds, inner, mapprime

"""
    construct_tMPS_tMPO(ψ_i::MPS, Ut::Vector{MPO}, ϕ_f::MPS)
```
|ϕ_f⟩    o———o———o———o
         |   |   |   |
Ut[end]  U———U———U———U
         |   |   |   |
Ut[…]          ⋮
         |   |   |   |
Ut[2]    U———U———U———U
         |   |   |   |  time
Ut[1]    U———U———U———U    ↑
         |   |   |   |    |
⟨ψ_i|    o———o———o———o    |
                          |
        ⟨L|  TL  TR |R⟩
```

Construct two boundary tMPS (⟨L| and |R⟩) and two tMPOs(TL and TR) for further use in the power method.

Note that we assume the input states to have 4 spatial sites!
`ψ_i` and `ϕ_f` are assumed to be a valid (space-like) MPS and each element of the vector 
Ut is assumed to be a valid MPO whereas the links match, respectively,
but the (physical) sites are not necessarily(!) correctly primed to link up correctly in the time direction.
"""
function construct_tMPS_tMPO(ψ_i::MPS, Ut::Vector{MPO}, ϕ_f::MPS)
  if hasqns(ψ_i)
    if maxlinkdim(ψ_i) > 1 && maxlinkdim(ϕ_f) > 1
      throw(ArgumentError("For now, we cannot process a QN-preserving initial/final state that is NOT a product state!"))
    end
  end
  if maxlinkdim(ψ_i) == 1 && maxlinkdim(ϕ_f) == 1
    ψi = delete_link_from_prodMPS(ψ_i)
    ϕf = delete_link_from_prodMPS(ϕ_f)

    Ut_1 = noprime.(ψi.data .* Ut[1].data)
    Ut_end = noprime.(Ut[end].data .* ϕf.data)

    input = hcat(
      Ut_1,
      hcat([(Ut[ii]).data for ii in 2:length(Ut)-1]...),
      Ut_end
    )
    return construct_tMPS_tMPO(input)
  else
    input = hcat(
      ψ_i.data,
      hcat([Ut_ele.data for Ut_ele in Ut]...),
      ϕ_f.data
    )
    return construct_tMPS_tMPO(input)
  end
end


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

function construct_tMPS_tMPO(input::Matrix{ITensor})
  nrows, ncolumns = size(input)
  @assert nrows == 4 "Assume input Matrix to have 4 rows!"


  links_cols = hcat([linkinds(input[:, n]) for n in 1:ncolumns]...)
  sites_cols = hcat([siteinds(input[:, n]) for n in 1:ncolumns]...)

  links_tMPS_L = [sim(only(sites_cols[1, 1]), tags="Link,time,nₜ=$(n-1)") for n in 1:ncolumns-1]
  links_tMPO_L = [sim(only(sites_cols[2, 1]), tags="Link,time,nₜ=$(n-1)") for n in 1:ncolumns-1]
  links_tMPO_R = [sim(only(sites_cols[3, 1]), tags="Link,time,nₜ=$(n-1)") for n in 1:ncolumns-1]
  links_tMPS_R = [sim(only(sites_cols[4, 1]), tags="Link,time,nₜ=$(n-1)") for n in 1:ncolumns-1]

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
  last_L = replaceinds(
    input[1, end],
    links_cols[1, end] => sites[end],
    only(sites_cols[1, end]) => dag(links_tMPS_L[end]),
  )

  ### second row, MPO transfer matrix column TL,
  ### corresponding to the left state ⟨L|
  first_MPO_L = replaceinds(
    input[2, 1],
    dag(links_cols[1, 1]) => dag(sites[1]),
    links_cols[2, 1] => prime(sites[1]),
    only(sites_cols[2, 1]) => links_tMPO_L[1]
  )
  bulk_MPO_L = [replaceinds(
    input[2, n],
    dag(links_cols[1, n]) => dag(sites[n]),
    links_cols[2, n] => prime(sites[n]),
    sites_cols[2, n][1] => links_tMPO_L[n],
    sites_cols[2, n][2] => dag(links_tMPO_L[n-1]),
  ) for n in 2:ncolumns-1 # difference to no MPS boundary at end!
  ]
  last_MPO_L = replaceinds(
    input[2, end],
    dag(links_cols[1, end]) => dag(sites[end]),
    links_cols[2, end] => prime(sites[end]),
    only(sites_cols[2, end]) => dag(links_tMPO_L[end]),
  )

  ### third row, MPO transfer matrix column TR,
  ### corresponding to the right state |R⟩
  first_MPO_R = replaceinds(
    input[3, 1],
    dag(links_cols[2, 1]) => prime(dag(sites[1])),
    links_cols[3, 1] => sites[1],
    only(sites_cols[3, 1]) => links_tMPO_R[1]
  )
  bulk_MPO_R = [replaceinds(
    input[3, n],
    dag(links_cols[2, n]) => prime(dag(sites[n])),
    links_cols[3, n] => sites[n],
    sites_cols[3, n][1] => links_tMPO_R[n],
    sites_cols[3, n][2] => dag(links_tMPO_R[n-1]),
  ) for n in 2:ncolumns-1 #
  ]
  last_MPO_R = replaceinds(
    input[3, end],
    dag(links_cols[2, end]) => prime(dag(sites[end])),
    links_cols[3, end] => sites[end],
    only(sites_cols[3, end]) => dag(links_tMPO_R[end]),
  )

  ### fourth row, MPS right state |R⟩
  first_R = replaceinds(
    input[4, 1],
    dag(links_cols[3, 1]) => dag(sites[1]),
    only(sites_cols[4, 1]) => links_tMPS_R[1]
  )
  bulk_R = [replaceinds(
    input[4, n],
    dag(links_cols[3, n]) => dag(sites[n]),
    sites_cols[4, n][1] => links_tMPS_R[n],
    sites_cols[4, n][2] => dag(links_tMPS_R[n-1]),
  ) for n in 2:ncolumns-1
  ]
  last_R = replaceinds(
    input[4, end],
    dag(links_cols[3, end]) => dag(sites[end]),
    only(sites_cols[4, end]) => dag(links_tMPS_R[end]),
  )

  return (
    MPS([first_L, bulk_L..., last_L]),
    MPO([first_MPO_L, bulk_MPO_L..., last_MPO_L]),
    MPO([first_MPO_R, bulk_MPO_R..., last_MPO_R]),
    MPS([first_R, bulk_R..., last_R])
  )
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


function maxlinkdim(M::AbstractVector{ITensor})
    md = 1
    for b in eachindex(M)[1:(end - 1)]
        l = linkind(M, b)
        linkdim = isnothing(l) ? 1 : dim(l)
        md = max(md, linkdim)
    end
    return md
end
