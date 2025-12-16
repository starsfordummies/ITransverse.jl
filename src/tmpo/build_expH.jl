struct ModelUt{T <: tMPOParams}
    tp::T
    Ut::MPO
end

function ModelUt(tp::tMPOParams)
    s = [sim(tp.mp.phys_site) for ii in 1:3]
    return ModelUt(s, tp)
end
function ModelUt(sites::Vector{<:Index}, tp::tMPOParams)
    Ut = tp.expH_func(sites, modelparams(tp.mp))
    return ModelUt(tp,Ut)
end


#= 
function build_expH(tp::tMPOParams; dt::Number=tp.dt)
    tp.expH_func(tp.mp, dt)
    #ModelUt(tp).Ut
end

""" Exp(-τH) for imaginary time evolution """
function build_expHim(tp::tMPOParams)
    tp.expH_func(tp.mp, -im*tp.dt )
end


=# 