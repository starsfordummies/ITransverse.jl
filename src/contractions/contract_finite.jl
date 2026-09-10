""" Simple transverse contraction: builds Left and Right vectors
 by applying the first Nhalf MPO to left_edge and the last Nhalf to right_edge """
function build_LR(left_mps::TMPSorMPS, mpos_bulk::Vector, right_mps::TMPSorMPS; cutoff=1e-12, maxdim=512, Nhalf::Int=halfsite(mpos_bulk))
    left_mps, right_mps = unsided(left_mps), unsided(right_mps)  # accept a tagged boundary vector, work on the MPS

    @info "Total length L=$(1+length(mpos_bulk)+1) Nt = $(length(left_mps)) ||  L=1:$(Nhalf), R=$(Nhalf+1:length(mpos_bulk)) "
    L = left_mps
    @showprogress for mpo in mpos_bulk[1:Nhalf]
        L = applyns(mpo, L; cutoff, maxdim)
    end

    R = right_mps
    @showprogress for mpo in reverse(mpos_bulk[Nhalf+1:end])
        R = apply(mpo, R; cutoff, maxdim)
    end

    return L, R
end


""" Simple transverse contraction: applies the first Nhalf MPO to left_edge and the last Nhalf to right_edge, using build_LR(),
then computes their overlap """
function contract_tn_transverse(left_edge::TMPSorMPS, MPO_list::Vector{MPO}, right_edge::TMPSorMPS; kwargs...)
    left_edge, right_edge = unsided(left_edge), unsided(right_edge)  # accept a tagged boundary vector, work on the MPS

    ll, rr = build_LR(left_edge, MPO_list, right_edge; kwargs...)
    overlap_noconj(ll,rr), max(maxlinkdim(ll), maxlinkdim(rr))
end


""" Traditional contraction scheme: applies all N rows of rows_mpo to bottom_mps and computes <top_mps|evolved bottom mps>, 
so the order of rows is from bottom to top """
function contract_tn_tetris(bottom_mps::TMPSorMPS, rows_mpo::Vector{MPO}, top_mps::TMPSorMPS; cutoff::Float64=1e-10, maxdim::Int=512)
    bottom_mps, top_mps = unsided(bottom_mps), unsided(top_mps)  # accept a tagged boundary vector, work on the MPS

    bottom_evolved = bottom_mps
    for mm in rows_mpo 
        bottom_evolved = apply(mm, bottom_evolved; cutoff, maxdim)
    end

    return inner(top_mps, bottom_evolved),  maxlinkdim(bottom_evolved)
end
