import .ChainModels: build_Ut

function build_Ut(tp::tMPOParams; dt::Number=tp.dt, build_imag::Bool=false)
    Ut = build_imag ? build_Ut(tp.expH_func, tp.mp; dt= -im*dt) : build_Ut(tp.expH_func, tp.mp; dt)
    return Ut 
end
