struct TruncParams
    cutoff::Float64
    maxbondim::Int64
    direction::String
end

# just some default for when we're lazy
TruncParams() = TruncParams(1e-12, 256)

# Default to right sweeps
TruncParams(cutoff::Float64, maxbondim::Int64) = TruncParams(cutoff, maxbondim, "right")
 

