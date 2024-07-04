
function build_expH(p::tmpo_params)
    p.expH_func(p.mp)
end

""" Exp(-τH) for imaginary time evolution """
function build_expHim(p::tmpo_params)
    mp = model_params(p.mp, dt = -im*p.mp.dt)
    p.expH_func(mp)

end

