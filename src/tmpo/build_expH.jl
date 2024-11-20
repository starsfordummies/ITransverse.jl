
function build_expH(tp::tMPOParams; dt::Number=tp.dt)
    tp.expH_func(tp.mp, dt)
end

""" Exp(-τH) for imaginary time evolution """
function build_expHim(tp::tMPOParams)
    tp.expH_func(tp.mp, dt=-im*tp.dt )
end

""" Quick building a FoldtMPOBlocks struct for playing around """
function quick_b()

    tp = ising_tp()
    b = FoldtMPOBlocks(tp)
    return b
end