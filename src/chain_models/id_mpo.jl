function H_id(sites, unusedparams=nothing)

    N = length(sites)
    os = OpSum()

    for j in 1:N
        os += 1/N, "Id", j
    end

    return MPO(os, sites)
end