struct TruncParams
    cutoff::Float64
    maxdim::Int64
    direction::String
end

# just some default for when we're lazy
TruncParams() = TruncParams(1e-12, 256)

# Default to right sweeps
TruncParams(cutoff::Float64, maxdim::Int64) = TruncParams(cutoff, maxdim, "right")
 
TruncParams(tp::TruncParams; cutoff::Real=tp.cutoff, maxdim::Int=tp.maxdim, direction=tp.direction) = TruncParams(cutoff,maxdim,direction)