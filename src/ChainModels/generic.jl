
function build_H_id(sites)

    N = length(sites)
    os = OpSum()

    for j in 1:N
        os += 1/N, "Id", j
    end

    return MPO(os, sites)
end