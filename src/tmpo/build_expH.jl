
function build_expH(tp::tMPOParams; dt::Number=tp.dt)
    tp.expH_func(tp.mp, dt)
end

""" Exp(-τH) for imaginary time evolution """
function build_expHim(tp::tMPOParams)
    tp.expH_func(tp.mp, -im*tp.dt )
end

""" Quick building a FwtMPOBlocks or FoldtMPOBlocks struct for playing around """
function quick_b(; folded::Bool=true)

    tp = ising_tp()
    b = folded ? FoldtMPOBlocks(tp) : FwtMPOBlocks(tp)

    return b
end