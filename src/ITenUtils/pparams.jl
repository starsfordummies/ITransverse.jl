


struct trunc_params
    cutoff::Float64
    maxbondim::Int64
    ortho_method::String

end

trunc_params() = trunc_params(1e-10, 100, "SVD")

 

