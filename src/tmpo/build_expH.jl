
function build_expH(tp::tmpo_params; dt::Number=tp.dt)
    tp.expH_func(tp.mp, dt)
end

""" Exp(-τH) for imaginary time evolution """
function build_expHim(tp::tmpo_params)
    tp.expH_func(tp.mp, dt=-im*tp.dt )
end

