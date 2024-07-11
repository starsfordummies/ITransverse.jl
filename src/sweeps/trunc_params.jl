struct TruncParams
    cutoff::Float64
    maxbondim::Int64
    ortho_method::String

end

# just some default for when we're lazy
TruncParams() = TruncParams(1e-10, 100, "SVD")

 

