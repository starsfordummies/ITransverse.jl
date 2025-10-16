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

"""
    tMPS([::Type{ElT} = Float64, ]sites; linkdims=1)

Construct an MPS filled with Empty ITensors of type `ElT` from a collection of indices.

Optionally specify the link dimension with the keyword argument `linkdims`, which by default is 1.

In the future we may generalize `linkdims` to allow specifying each individual link dimension as a vector,
and additionally allow specifying quantum numbers.
"""
function tMPS(
        ::Type{T}, sites::Vector{<:Index}; linkdims::Union{Integer, Vector{<:Integer}} = 1
    ) where {T <: Number}
    _linkdims = _fill_linkdims(linkdims, sites)
    N = length(sites)
    v = Vector{ITensor}(undef, N)
    if N == 1
        v[1] = ITensor(T, sites[1])
        return tMPS(v)
    end

    spaces = if hasqns(sites)
        [[QN() => _linkdims[j]] for j in 1:(N - 1)]
    else
        [_linkdims[j] for j in 1:(N - 1)]
    end

    l = [Index(spaces[ii], "Link,l=$ii") for ii in 1:(N - 1)]
    for ii in eachindex(sites)
        s = sites[ii]
        if ii == 1
            v[ii] = ITensor(T, l[ii], s)
        elseif ii == N
            v[ii] = ITensor(T, dag(l[ii - 1]), s)
        else
            v[ii] = ITensor(T, dag(l[ii - 1]), s, l[ii])
        end
    end
    return tMPS(v)
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

function tMPO(::Type{ElT}, sites::Vector{<:Index}) where {ElT <: Number}
    N = length(sites)
    v = Vector{ITensor}(undef, N)
    if N == 0
        return tMPO()
    elseif N == 1
        v[1] = ITensor(ElT, dag(sites[1]), sites[1]')
        return tMPO(v)
    end
    space_ii = all(hasqns, sites) ? [QN() => 1] : 1
    l = [Index(space_ii, "Link,l=$ii") for ii in 1:(N - 1)]
    for ii in eachindex(sites)
        s = sites[ii]
        if ii == 1
            v[ii] = ITensor(ElT, dag(s), s', l[ii])
        elseif ii == N
            v[ii] = ITensor(ElT, dag(l[ii - 1]), dag(s), s')
        else
            v[ii] = ITensor(ElT, dag(l[ii - 1]), dag(s), s', l[ii])
        end
    end
    return tMPO(v)
end

tMPO(sites::Vector{<:Index}) = tMPO(Float64, sites)

"""
    tMPO(N::Int)

Make an tMPO of length `N` filled with default ITensors.
"""
tMPO(N::Int) = tMPO(Vector{ITensor}(undef, N))