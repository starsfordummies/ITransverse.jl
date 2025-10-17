using ITensorMPS, ITensors
# using NDTensors: @Algorithm_str
# using ITensors.SiteTypes: SiteTypes, siteind, siteinds, state
# using ITensors: Algorithm, contract, hassameinds, inner, mapprime




"""
    construct_tMPS_tMPO(input::Matrix{ITensor})
```
   ⟨ψ|     U(t)    |ϕ⟩     
⟨L| o—U—U—U—U—U—U—U—o
    | | | | | | | | |
TL  o—U—U—U—U—U—U—U—o
    | | | | | | | | |
TR  o—U—U—U—U—U—U—U—o
    | | | | | | | | |
|R⟩ o—U—U—U—U—U—U—U—o
```

Construct two boundary tMPS and two tMPOs such for further use in the power method.
Note that we assume the Matrix to have 4 rows and Nₜ+1 columns where Nₜ counts the time steps,
and the first column describes the boundary (initial t=0) MPS in space. 

"""
function construct_unfolded_tMPS_tMPO(input::Matrix{ITensor})
  nrows, ncolumns = size(input)
  @assert nrows == 4 "Assume input Matrix to have 4 rows!"

  # MPS boundary defines the old physical site and the new link Index
  # conceptually we rotate from old Site to new Link by +90°
  has_MPS_boundary_at_end = if order(input[1,end]) == 2 
    true
  elseif order(input[1,end]) == 3
    false
  else
    error("Order of boundary ITensor is $(order(input[1,end])), don't know what to do with it")
  end

  number_of_new_links = has_MPS_boundary_at_end ? ncolumns-1 : ncolumns

  links_tMPS_L = [sim(only(inds(input[1,1],"Site")), tags="Link,time,nₜ=$(n)") for n in 0:number_of_new_links]
  links_tMPO_L = [sim(only(inds(input[2,1],"Site")), tags="Link,time,nₜ=$(n)") for n in 0:number_of_new_links]
  links_tMPO_R = [sim(only(inds(input[3,1],"Site")), tags="Link,time,nₜ=$(n)") for n in 0:number_of_new_links]
  links_tMPS_R = [sim(only(inds(input[4,1],"Site")), tags="Link,time,nₜ=$(n)") for n in 0:number_of_new_links]

  # the new sites are a bit special, the first (and perhaps last) site is (are) given by the boundary MPS
  # while the bulk of the new sites are given by the old link of the time evolution operator
  sites = [sim(only(inds(input[1,n],"Link")), tags="Site,time,nₜ=$(n-1)") for n in 1:ncolumns]
  # we have now created all the new indices for the new tMPSs and tMPOs and effectively
  # rotated the network by relabelling sites with links and links with sites.
  first_L = replaceinds(
      input[1,1],
      only(inds(input[1,1],tags="Link,l=1", plev=0)) => sites[1],
      only(inds(input[1,1];tags="Site", plev=0)) => links_tMPS_L[1]
    )
  bulk_L = [replaceinds(
        input[1,n],
        only(inds(input[1,n],tags="Link,l=1", plev=0)) => sites[n],
        only(inds(input[1,n];tags="Site", plev=0)) => dag(links_tMPS_L[n-1]),
        only(inds(input[1,n];tags="Site", plev=1)) => links_tMPS_L[n]
      ) for n in 2:ncolumns-1 # difference to no MPS boundary at end!
    ]
  last_L =  has_MPS_boundary_at_end ? replaceinds(
        input[1,end],
        only(inds(input[1,end],tags="Link,l=1", plev=0)) => sites[end],
        # only(inds(input[1,end],tags="Link,l=2", plev=0)) => prime(sites[end]),
        only(inds(input[1,end];tags="Site", plev=0)) => dag(links_tMPS_L[end-1]),
      ) : replaceinds(
        input[1,end],
        only(inds(input[1,end],tags="Link,l=1", plev=0)) => sites[end],
        # only(inds(input[1,end],tags="Link,l=2", plev=0)) => prime(sites[end]),
        only(inds(input[1,end];tags="Site", plev=0)) => dag(links_tMPS_L[end-1]),
        only(inds(input[1,end];tags="Site", plev=1)) => links_tMPS_L[end]
      )
  
  L_vec =  [first_L, bulk_L..., last_L]

  first_MPO_L = replaceinds(
      input[2,1],
      only(inds(input[2,1],tags="Link,l=1", plev=0)) => dag(sites[1]),
      only(inds(input[2,1],tags="Link,l=2", plev=0)) => prime(sites[1]),
      only(inds(input[2,1];tags="Site", plev=0)) => links_tMPO_L[1]
    )
  bulk_MPO_L = [replaceinds(
        input[2,n],
        only(inds(input[2,n],tags="Link,l=1", plev=0)) => dag(sites[n]),
        only(inds(input[2,n],tags="Link,l=2", plev=0)) => prime(sites[n]),
        only(inds(input[2,n];tags="Site", plev=0)) => dag(links_tMPO_L[n-1]),
        only(inds(input[2,n];tags="Site", plev=1)) => links_tMPO_L[n]
      ) for n in 2:ncolumns-1 # difference to no MPS boundary at end!
    ]
  last_MPO_L = has_MPS_boundary_at_end ? replaceinds(
        input[2,end],
        only(inds(input[2,end],tags="Link,l=1", plev=0)) => dag(sites[end]),
        only(inds(input[2,end],tags="Link,l=2", plev=0)) => prime(sites[end]),
        only(inds(input[2,end];tags="Site", plev=0)) => dag(links_tMPO_L[end-1]),
      ) : replaceinds(
        input[2,end],
        only(inds(input[2,end],tags="Link,l=1", plev=0)) => dag(sites[end]),
        only(inds(input[2,end],tags="Link,l=2", plev=0)) => prime(sites[end]),
        only(inds(input[2,end];tags="Site", plev=0)) => dag(links_tMPO_L[end-1]),
        only(inds(input[2,end];tags="Site", plev=1)) => links_tMPO_L[end]
      )
  MPO_L_vec = [first_MPO_L, bulk_MPO_L..., last_MPO_L]


  first_MPO_R = replaceinds(
      input[3,1],
      only(inds(input[3,1],tags="Link,l=2", plev=0)) => prime(dag(sites[1])),
      only(inds(input[3,1],tags="Link,l=3", plev=0)) => sites[1],
      only(inds(input[3,1];tags="Site", plev=0)) => links_tMPO_R[1]
    )
  bulk_MPO_R = [replaceinds(
        input[3,n],
        only(inds(input[3,n],tags="Link,l=2", plev=0)) => prime(dag(sites[n])),
        only(inds(input[3,n],tags="Link,l=3", plev=0)) => sites[n],
        only(inds(input[3,n];tags="Site", plev=0)) => dag(links_tMPO_R[n-1]),
        only(inds(input[3,n];tags="Site", plev=1)) => links_tMPO_R[n]
      ) for n in 2:ncolumns-1 # difference to no MPS boundary at end!
    ]
  last_MPO_R = has_MPS_boundary_at_end ? replaceinds(
        input[3,end],
        only(inds(input[3,end],tags="Link,l=2", plev=0)) => prime(dag(sites[end])),
        only(inds(input[3,end],tags="Link,l=3", plev=0)) => sites[end],
        only(inds(input[3,end];tags="Site", plev=0)) => dag(links_tMPO_R[end-1]),
      ) : replaceinds(
        input[3,end],
        only(inds(input[3,end],tags="Link,l=2", plev=0)) => prime(dag(sites[end])),
        only(inds(input[3,end],tags="Link,l=3", plev=0)) => sites[end],
        only(inds(input[3,end];tags="Site", plev=0)) => dag(links_tMPO_R[end-1]),
        only(inds(input[3,end];tags="Site", plev=1)) => links_tMPO_R[end]
      )
  MPO_R_vec = [first_MPO_R, bulk_MPO_R..., last_MPO_R]

  first_R = replaceinds(
      input[4,1],
      only(inds(input[4,1],tags="Link,l=3", plev=0)) => dag(sites[1]),
      only(inds(input[4,1];tags="Site", plev=0)) => links_tMPS_R[1]
    )
  bulk_R = [replaceinds(
        input[4,n],
        only(inds(input[4,n],tags="Link,l=3", plev=0)) => dag(sites[n]),
        only(inds(input[4,n];tags="Site", plev=0)) => dag(links_tMPS_R[n-1]),
        only(inds(input[4,n];tags="Site", plev=1)) => links_tMPS_R[n]
      ) for n in 2:ncolumns-1 # difference to no MPS boundary at end!
    ]
  last_R = has_MPS_boundary_at_end ? replaceinds(
        input[4,end],
        only(inds(input[4,end],tags="Link,l=3", plev=0)) => dag(sites[end]),
        only(inds(input[4,end];tags="Site", plev=0)) => dag(links_tMPS_R[end-1]),
      ) : replaceinds(
        input[4,end],
        only(inds(input[4,end],tags="Link,l=3", plev=0)) => dag(sites[end]),
        only(inds(input[4,end];tags="Site", plev=0)) => dag(links_tMPS_R[end-1]),
        only(inds(input[4,end];tags="Site", plev=1)) => links_tMPS_R[end]
      )
  R_vec = [first_R, bulk_R..., last_R]

  return (MPS(L_vec), MPO(MPO_L_vec), MPO(MPO_R_vec), MPS(R_vec))
end



# """
#     temporal MPS

# A finite size matrix product state type in the temporal direction.
# There is no orthogonality center!
# """
# mutable struct tMPS <: AbstractMPS
#     data::Vector{ITensor}
#     llim::Int
#     rlim::Int
# end

# function tMPS(A::Vector{<:ITensor}; ortho_lims::UnitRange = 1:length(A))
#     return tMPS(A, first(ortho_lims) - 1, last(ortho_lims) + 1)
# end

# set_data(A::tMPS, data::Vector{ITensor}) = tMPS(data, A.llim, A.rlim)

# @doc """
#     tMPS(v::Vector{<:ITensor})

# Construct an MPS from a Vector of ITensors.
# """ tMPS(v::Vector{<:ITensor})

# """
#     tMPS()

# Construct an empty MPS with zero sites.
# """
# tMPS() = tMPS(ITensor[], 0, 0)

# """
#     tMPS(N::Int)

# Construct an MPS with N sites with default constructed
# ITensors.
# """
# function tMPS(N::Int; ortho_lims::UnitRange = 1:N)
#     return tMPS(Vector{ITensor}(undef, N); ortho_lims = ortho_lims)
# end


# tMPS(sites::Vector{<:Index}, args...; kwargs...) = tMPS(Float64, sites, args...; kwargs...)


# """
#     tMPO

# A finite size matrix product operator type in the temporal direction.
# Keeps track of the orthogonality center.
# """
# mutable struct tMPO <: AbstractMPS
#     data::Vector{ITensor}
#     llim::Int
#     rlim::Int
# end

# function tMPO(A::Vector{<:ITensor}; ortho_lims::UnitRange = 1:length(A))
#     return tMPO(A, first(ortho_lims) - 1, last(ortho_lims) + 1)
# end

# set_data(A::tMPO, data::Vector{ITensor}) = tMPO(data, A.llim, A.rlim)

# tMPO() = tMPO(ITensor[], 0, 0)

# function convert(::Type{tMPS}, M::tMPO)
#     return MPS(data(M); ortho_lims = ortho_lims(M))
# end

# function convert(::Type{tMPO}, M::tMPS)
#     return tMPO(data(M); ortho_lims = ortho_lims(M))
# end


# tMPO(sites::Vector{<:Index}) = tMPO(Float64, sites)

# """
#     tMPO(N::Int)

# Make an tMPO of length `N` filled with default ITensors.
# """
# tMPO(N::Int) = tMPO(Vector{ITensor}(undef, N))




# """
#     siteind(M::tMPS, j::Int; kwargs...)

# Get the first site Index of the MPS. Return `nothing` if none is found.
# """
# SiteTypes.siteind(M::tMPS, j::Int; kwargs...) = siteind(first, M, j; kwargs...)

# """
#     siteind(::typeof(only), M::tMPS, j::Int; kwargs...)

# Get the only site Index of the MPS. Return `nothing` if none is found.
# """
# function SiteTypes.siteind(::typeof(only), M::tMPS, j::Int; kwargs...)
#     is = siteinds(M, j; kwargs...)
#     if isempty(is)
#         return nothing
#     end
#     return only(is)
# end

# """
#     siteinds(M::tMPS)
#     siteinds(::typeof(first), M::tMPS)

# Get a vector of the first site Index found on each tensor of the MPS.

#     siteinds(::typeof(only), M::tMPS)

# Get a vector of the only site Index found on each tensor of the MPS. Errors if more than one is found.

#     siteinds(::typeof(all), M::tMPS)

# Get a vector of the all site Indices found on each tensor of the MPS. Returns a Vector of IndexSets.
# """
# SiteTypes.siteinds(M::tMPS; kwargs...) = siteinds(first, M; kwargs...)

# function replace_siteinds!(M::tMPS, sites)
#     for j in eachindex(M)
#         sj = only(siteinds(M, j))
#         M[j] = replaceinds(M[j], sj => sites[j])
#     end
#     return M
# end

# replace_siteinds(M::tMPS, sites) = replace_siteinds!(copy(M), sites)


# """
#     siteind(M::tMPO, j::Int; plev = 0, kwargs...)

# Get the first site Index of the MPO found, by
# default with prime level 0.
# """
# SiteTypes.siteind(M::tMPO, j::Int; kwargs...) = siteind(first, M, j; plev = 0, kwargs...)

# # TODO: make this return the site indices that would have
# # been used to create the MPO? I.e.:
# # [dag(siteinds(M, j; plev = 0, kwargs...)) for j in 1:length(M)]
# """
#     siteinds(M::tMPO; kwargs...)

# Get a Vector of IndexSets of all the site indices of M.
# """
# SiteTypes.siteinds(M::tMPO; kwargs...) = siteinds(all, M; kwargs...)

# function SiteTypes.siteinds(Mψ::Tuple{MPO, MPS}, n::Int; kwargs...)
#     return siteinds(uniqueinds, Mψ[1], Mψ[2], n; kwargs...)
# end

# function nsites(Mψ::Tuple{MPO, MPS})
#     M, ψ = Mψ
#     N = length(M)
#     @assert N == length(ψ)
#     return N
# end

# function SiteTypes.siteinds(Mψ::Tuple{MPO, MPS}; kwargs...)
#     return [siteinds(Mψ, n; kwargs...) for n in 1:nsites(Mψ)]
# end

# # XXX: rename originalsiteinds?
# """
#     firstsiteinds(M::tMPO; kwargs...)

# Get a Vector of the first site Index found on each site of M.

# By default, it finds the first site Index with prime level 0.
# """
# firstsiteinds(M::tMPO; kwargs...) = siteinds(first, M; plev = 0, kwargs...)

# function hassameinds(::typeof(siteinds), ψ::tMPS, Hϕ::Tuple{tMPO, tMPS})
#     N = length(ψ)
#     @assert N == length(Hϕ[1]) == length(Hϕ[1])
#     for n in 1:N
#         !hassameinds(siteinds(Hϕ, n), siteinds(ψ, n)) && return false
#     end
#     return true
# end


# function _contract(::Algorithm"naive", A::tMPO, ψ::tMPS; truncate = false, kwargs...)
#     A = sim(linkinds, A)
#     ψ = sim(linkinds, ψ)

#     N = length(A)
#     if N != length(ψ)
#         throw(DimensionMismatch("lengths of MPO ($N) and MPS ($(length(ψ))) do not match"))
#     end

#     ψ_out = typeof(ψ)(N)
#     for j in 1:N
#         ψ_out[j] = A[j] * ψ[j]
#     end

#     for b in 1:(N - 1)
#         Al = commoninds(A[b], A[b + 1])
#         ψl = commoninds(ψ[b], ψ[b + 1])
#         l = [Al..., ψl...]
#         tags = only(unique(ITensors.tags.(l)))
#         if !isempty(l)
#             C = combiner(l, tags=tags)
#             ψ_out[b] *= C
#             ψ_out[b + 1] *= dag(C)
#         end
#     end

#     # This truncation method needs to be fixed because
#     # it will use orthogonality which we do not have!
#     if truncate
#         # truncate!(ψ_out; kwargs...)
#         error("Truncation does not exist yet")
#     end

#     return ψ_out
# end

# function ITensors.contract(alg::Algorithm"naive", A::tMPO, ψ::tMPS; kwargs...)
#     return ITransverse._contract(alg, A, ψ; kwargs...)
# end


# function ITensors.contract(A::tMPO, B::tMPS; alg = "naive", kwargs...)
#     return ITransverse._contract(Algorithm(alg), A, B; kwargs...)
# end


# function ITensors.contract(alg::Algorithm"naive", A::tMPO, ψ::tMPS; kwargs...)
#     return ITransverse._contract(alg, A, ψ; kwargs...)
# end


# function apply(A::tMPO, B::tMPS; kwargs...)
#     AB = ITransverse.contract(A, B; kwargs...)
#     return replaceprime(AB, 1 => 0)
# end

# # function apply(A1::tMPO, A2::tMPO, A3::tMPO, As::tMPO...; kwargs...)
# #     return apply(apply(A1, A2; kwargs...), A3, As...; kwargs...)
# # end

# # (A::tMPO)(B::tMPO; kwargs...) = apply(A, B; kwargs...)