"""
My modification of ITensors' truncate!() allowing for complex values.
Truncates on the absolute value of the spectrum. GPU-safe (no scalar indexing).
"""

function ctruncate!!(P::AbstractArray; kwargs...)
    truncerr, docut = ctruncate!(P; kwargs...)
    return P, truncerr, docut
end

function ctruncate!(
    P::AbstractVector;
    mindim=1,
    maxdim=length(P),
    cutoff=0.0,
    use_absolute_cutoff=default_use_absolute_cutoff(P),
    use_relative_cutoff=default_use_relative_cutoff(P),
)

    mindim = replace_nothing(mindim, 1)
    maxdim = replace_nothing(maxdim, length(P))
    #cutoff = replace_nothing(cutoff, typemin(eltype(P)))
    cutoff = replace_nothing(cutoff, 0.)

    use_absolute_cutoff = replace_nothing(use_absolute_cutoff, default_use_absolute_cutoff(P))
    use_relative_cutoff = replace_nothing(use_relative_cutoff, default_use_relative_cutoff(P))

    origm = length(P)
    absP  = abs.(Array(P))   

    # Edge case: single element
    if origm == 1
        return zero(eltype(P)), absP[1] / 2
    end

    # --- Phase 1: hard cap at maxdim ---
    n        = min(origm, maxdim)
    truncerr = sum(@view absP[n+1:end]; init=0.0)   # tail beyond maxdim

    # --- Phase 2: cutoff-based trimming ---
    scale = use_relative_cutoff ? max(sum(absP), eps(Float64)) : 1.0

    if use_absolute_cutoff
        while n > mindim && absP[n] <= cutoff
            truncerr += absP[n]
            n -= 1
        end
    else
        while n > mindim && (truncerr + absP[n]) / scale <= cutoff
            truncerr += absP[n]
            n -= 1
        end
        truncerr /= scale
    end

    # --- Phase 3: compute cut value ---
    docut = if n < origm
        mid = (absP[n] + absP[n+1]) / 2
        gap = abs(absP[n] - absP[n+1])
        gap < 1e-3 * absP[n] ? mid + 1e-3 * absP[n] : mid
    else
        zero(Float64)
    end

    resize!(P, n)
    return truncerr, docut
end