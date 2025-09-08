""" Columns struct for a uniform system of length N: stores
 - a left MPS, 
 - a set of columns of length N (the first and last are unused/unintialized)
 - a right MPS

 Ideally we initialize them once and the beginning and not touch them again.
"""
struct Columns
    ledge::MPS
    cols::Vector{MPO}
    redge::MPS
end

function Columns(L::Int, ledge::MPS, col::MPO, redge::MPS) 
    cols = fill(col, L)
    Columns(ledge, cols, redge)
end

Adapt.adapt_structure(to, cc::Columns) = Columns(
    adapt(to, cc.ledge),
    adapt(to, cc.cols), 
    adapt(to, cc.redge)
)


Base.size(cc::Columns) = length(cc.cols)
Base.eachindex(cc::Columns) = eachindex(cc.cols)  # Forward to .cols 
Base.lastindex(cc::Columns) = length(cc.cols)
Base.length(cc::Columns) = length(cc.cols)

function Base.getindex(cc::Columns, i::Int)
    if i == 1 
        return cc.ledge
    elseif i == length(cc)
        return cc.redge
    else
        return cc.cols[i]
    end
end
function Base.getindex(cc::Columns, r::AbstractRange{Int})
    return [cc[i] for i in r] 
end

function Base.iterate(cc::Columns, state=1)
    n = length(cc.cols)  # total number of elements: ledge + cols + redge
    if state > n
        return nothing
    elseif state == 1
        return (cc.ledge, 2)     # yield ledge first, next state = 2
    elseif state == n
        return (cc.redge, state + 1)   # yield redge last
    else
        return (cc.cols[state], state + 1)  # yield cols at state-1
    end
end


function Base.show(io::IO, ::MIME"text/plain", cc::Columns)
    npl1(x) = count(!isempty, siteinds(x, plev=1))
    npl0(x) = count(!isempty, siteinds(x, plev=0))

    LR = vcat("L", [  npl0(col) > npl1(col) ? "L" : npl1(col) == npl0(col) ? "C" : "R" for col in cc[2:end-1] ], "R")
    print(io, "Columns $(length(cc))|$(length.(cc)) $(LR)")
end

""" Given a Columns struct, 
performs the *exact* contraction from the edges towawrds the center, without using environments """
function contract_cols(cc::Columns; maxdim=1024)

    NN = length(cc)

    ll = cc[1]
    rr = cc[end]

    for jj = 2:div(NN,2)
        ll = applyns(cc[jj], ll; truncate=true, maxdim)
    end

    for jj = NN-1:-1:div(NN,2)+1
        rr = applyn(cc[jj], rr; truncate=true, maxdim)
    end

    overlap_noconj(ll,rr)
end