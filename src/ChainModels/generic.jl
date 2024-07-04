# Useful functions for model building

braket(A::ITensor, B::ITensor) = replaceprime(prime(A) * B + prime(B) * A, 2, 1)
braket(A::ITensor, B::ITensor, C::ITensor) = replaceprime(
    (A'' * B' * C) + (A'' * C' * B) + (B'' * A' * C) + (B'' * C' * A) + (C'' * A' * B) + (C'' * B' * A),
    3, 1)
braket2(A::ITensor, B::ITensor) = replaceprime((A'' * B' * B) + (B'' * A' * B) + (B'' * B' * A), 3, 1)


function build_expH(p::tmpo_params)
    p.expH_func(p.mp)
end

""" Exp(-τH) for imaginary time evolution """
function build_expHim(p::tmpo_params)
    mp = model_params(p.mp, dt = -im*p.mp.dt)
    p.expH_func(mp)

end



function build_H_id(sites)

    N = length(sites)
    os = OpSum()

    for j in 1:N
        os += 1/N, "Id", j
    end

    return MPO(os, sites)
end