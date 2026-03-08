function build_Ut(sites::Vector{<:Index}, fUt::Function, par1::Number, par2::Number, par3::Number=0; dt::Number, build_imag::Bool=false)
    if build_imag
        dt = -im*dt
    end
    expanded_params = (par1,par2,par3)
    fUt(sites, expanded_params...; dt)
end

function build_Ut(sites::Vector{<:Index}, fUt::Function,  mp::ModelParams; dt::Number, build_imag::Bool=false)
    if build_imag
        dt = -im*dt
    end
    expanded_params = modelparams(mp)
    fUt(sites, expanded_params...; dt)
end


function build_Ut(fUt::Function, mp::ModelParams; dt::Number, build_imag::Bool=false)
    ss = [sim(mp.phys_site) for _ in 1:3]
    build_Ut(ss, fUt, mp; dt, build_imag)
end

function build_Ut(tp::tMPOParams; dt::Number=tp.dt, build_imag::Bool=false)
    build_Ut(tp.expH_func, tp.mp; dt, build_imag) 
end

function build_Ut(b::FwtMPOBlocks; dt::Number=b.tp.dt, build_imag::Bool=false)
    Ut = build_Ut(b.tp.expH_func, b.tp.mp; dt, build_imag)
    return Ut 
end
