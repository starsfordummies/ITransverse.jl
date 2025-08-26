struct TruncParams
    cutoff::Float64
    maxbondim::Int64
    direction::String
end

# just some default for when we're lazy
TruncParams() = TruncParams(1e-12, 256)

# Default to right sweeps
TruncParams(cutoff::Float64, maxbondim::Int64) = TruncParams(cutoff, maxbondim, "right")
 
TruncParams(tp::TruncParams; cutoff::Real=tp.cutoff, maxbondim::Int=tp.maxbondim, direction=tp.direction) = TruncParams(cutoff,maxbondim,direction)