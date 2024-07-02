struct PMParams
    trunc_params::trunc_params
    itermax::Int64
    ds2_converged::Float64
    increase_chi::Bool
end

function PMParams(; truncp=trunc_params(), itermax::Int=200,
    ds2_converged::Float64=1e-5, increase_chi::Bool=false)
    return PMParams(truncp, itermax, ds2_converged, increase_chi)
end