""" Initializes the light cone folded and rotated temporal MPS |R> given `tMPOParams`
builds a (length n) tMPS with (time_fold)  legs.
Returns (psi[the light cone right MPS], b[the folded tMPO building blocks])"""
# function init_cone(tp::tMPOParams, n::Int=10; full::Bool=true)
#     b = FoldtMPOBlocks(tp)
#     init_cone(b, n)
# end

function init_cone(b::FoldtMPOBlocks, n::Int=6; LR::Symbol=:right, full::Bool=true)

    @assert b.tp.nbeta == 0  # not implemented yet otherwise
    time_dim = dim(b.WWc,1)
    
    ts = addtags(siteinds(time_dim, n; conserve_qns=false), "time_fold")

    init_cone(b, ts; LR, full)

    # psi = folded_tMPS(b, [ts[1]]; LR)

    # for jj = 2:n
    #     m = folded_tMPO_ext(b,ts[1:jj]; LR)
    #     psi = applyn(m, psi)
    #     orthogonalize!(psi, length(psi))
    # end

    # return psi, b
end


function init_cone(b::FoldtMPOBlocks, ts::Vector{Index{Int64}}; LR::Symbol, full::Bool)

    @assert b.tp.nbeta == 0  # not implemented yet otherwise
    
    if full 
        psi = folded_tMPS(b, ts; LR)
        for jj = 2:length(ts)
            m = folded_tMPO(b,ts)
            psi = applyn(m, psi)
            orthogonalize!(psi, length(psi))
            orthogonalize!(psi,1)
        end
    else

        psi = folded_tMPS(b, [ts[1]]; LR)

        for jj = 2:length(ts)

            m = folded_tMPO_ext(b,ts[1:jj]; LR)
            psi = applyn(m, psi)
            orthogonalize!(psi, length(psi))
        end
    end

    return psi
end

