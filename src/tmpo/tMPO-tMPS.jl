using ITensorMPS

"""
    temporal MPS

A finite size matrix product state type in the temporal direction.
There is no orthogonality center!
"""
mutable struct tMPS <: AbstractMPS
    data::Vector{ITensor}
    llim::Int
    rlim::Int
end

function tMPS(A::Vector{<:ITensor}; ortho_lims::UnitRange = 1:length(A))
    return tMPS(A, first(ortho_lims) - 1, last(ortho_lims) + 1)
end

set_data(A::tMPS, data::Vector{ITensor}) = tMPS(data, A.llim, A.rlim)

@doc """
    tMPS(v::Vector{<:ITensor})

Construct an MPS from a Vector of ITensors.
""" tMPS(v::Vector{<:ITensor})

"""
    tMPS()

Construct an empty MPS with zero sites.
"""
tMPS() = tMPS(ITensor[], 0, 0)

"""
    tMPS(N::Int)

Construct an MPS with N sites with default constructed
ITensors.
"""
function tMPS(N::Int; ortho_lims::UnitRange = 1:N)
    return tMPS(Vector{ITensor}(undef, N); ortho_lims = ortho_lims)
end


tMPS(sites::Vector{<:Index}, args...; kwargs...) = tMPS(Float64, sites, args...; kwargs...)


"""
    tMPO

A finite size matrix product operator type in the temporal direction.
Keeps track of the orthogonality center.
"""
mutable struct tMPO <: AbstractMPS
    data::Vector{ITensor}
    llim::Int
    rlim::Int
end

function tMPO(A::Vector{<:ITensor}; ortho_lims::UnitRange = 1:length(A))
    return tMPO(A, first(ortho_lims) - 1, last(ortho_lims) + 1)
end

set_data(A::tMPO, data::Vector{ITensor}) = tMPO(data, A.llim, A.rlim)

tMPO() = tMPO(ITensor[], 0, 0)

function convert(::Type{tMPS}, M::tMPO)
    return MPS(data(M); ortho_lims = ortho_lims(M))
end

function convert(::Type{tMPO}, M::MPS)
    return tMPO(data(M); ortho_lims = ortho_lims(M))
end


tMPO(sites::Vector{<:Index}) = tMPO(Float64, sites)

"""
    tMPO(N::Int)

Make an tMPO of length `N` filled with default ITensors.
"""
tMPO(N::Int) = tMPO(Vector{ITensor}(undef, N))


"""
    construct_tMPS_tMPO(input::Matrix{ITensor})

Construct two boundary tMPS and two tMPOs such for further use in the power method.
Note that we assume the Matrix to have 4 rows and Nₜ+1 columns where Nₜ counts the time steps,
and the first column describes the boundary (initial t=0) MPS in space. 
"""
function construct_tMPS_tMPO(input::Matrix{ITensor})
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

  L_vec = has_MPS_boundary_at_end ? [
    replaceinds(
      input[1,1],
      only(inds(input[1,1],"Site")) => links_tMPS_L[1],
      only(inds(input[1,1],"Link")) => sites[1],
    ),
    [replaceinds(
        input[1,n],
        only(inds(input[1,n],"Link")) => sites[n],
        only(inds(input[1,n];tags="Site", plev=0)) => dag(links_tMPS_L[n-1]),
        only(inds(input[1,n];tags="Site", plev=1)) => links_tMPS_L[n]
      ) for n in 2:ncolumns-1 # difference to no MPS boundary at end!
    ]...,
    # difference to no MPS boundary at end!
    replaceinds(
      input[1,end],
      only(inds(input[1,end],"Site")) => dag(links_tMPS_L[end]),
      only(inds(input[1,1],"Link")) => sites[end],
    )
  ] : [
    replaceinds(input[1,1], only(inds(input[1,1],"Site")) => links_tMPS_L[1], only(inds(input[1,1],"Link")) => sites[1] ),
    [replaceinds(
        input[1,n],
        only(inds(input[1,n],"Link")) => sites[n],
        only(inds(input[1,n];tags="Site", plev=0)) => dag(links_tMPS_L[n-1]),
        only(inds(input[1,n];tags="Site", plev=1)) => links_tMPS_L[n]
      ) for n in 2:ncolumns
    ]...
  ]

  MPO_L_vec = has_MPS_boundary_at_end ? [
    replaceinds(
      input[2,1],
      only(inds(input[2,1],tags="Link,l=1", plev=0)) => dag(sites[1]),
      only(inds(input[2,1],tags="Link,l=2", plev=0)) => prime(sites[1]),
      only(inds(input[2,1];tags="Site", plev=0)) => links_tMPS_L[1]
    ),
    [replaceinds(
        input[2,n],
        only(inds(input[2,n],"Link")) => sites[n],
        only(inds(input[2,n];tags="Site", plev=0)) => dag(links_tMPP_L[n-1]),
        only(inds(input[2,n];tags="Site", plev=1)) => links_tMPS_L[n]
      ) for n in 2:ncolumns-1 # difference to no MPS boundary at end!
    ]...,
    # difference to no MPS boundary at end!
    replaceinds(
      input[1,end],
      only(inds(input[1,end],"Site")) => dag(links_tMPS_L[end]),
      only(inds(input[1,1],"Link")) => sites[end],
    )
  ] : [
    replaceinds(input[1,1], only(inds(input[1,1],"Site")) => links_tMPS_L[1], only(inds(input[1,1],"Link")) => sites[1] ),
    [replaceinds(
        input[1,n],
        only(inds(input[1,n],"Link")) => sites[n],
        only(inds(input[1,n];tags="Site", plev=0)) => dag(links_tMPS_L[n-1]),
        only(inds(input[1,n];tags="Site", plev=1)) => links_tMPS_L[n]
      ) for n in 2:ncolumns
    ]...
  ]

end

