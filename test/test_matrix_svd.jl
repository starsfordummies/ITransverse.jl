using ITensors
using ITransverse
using ITransverse: matrix_svd
using Test

a = random_itensor(Index(10), Index(10))

f = svd(a, ind(a,1), maxdim=7, cutoff=1e-3)

mf = matrix_svd(a, maxdim=7,cutoff=1e-3)

@test f.U.tensor ≈ mf.U.tensor
@test diag(f.S)≈ diag(mf.S)
@test f.V.tensor ≈ mf.V.tensor

#= 
n = 300
i1 = Index(n)
i2 = Index(n)

errs = []
for _ = 1:100

    ws = [ITensor(ITransverse.ITenUtils.make_powerlaw(n), i1,i2),
    ITensor(ITransverse.ITenUtils.make_exponential(n), i1,i2),
    ITensor(ITransverse.ITenUtils.make_step(n, 20), i1,i2),
    ITensor(ITransverse.ITenUtils.make_plateau(n, 20), i1,i2),
    ITensor(ITransverse.ITenUtils.make_flat(n), i1,i2)]

    cutoff = 1e-4
    for w in ws
        f = svd(w, i1; cutoff=cutoff^2, use_relative_cutoff=false)
        wrec = f.U * f.S * f.V
        @show dim(f.S,1)
        @show norm(w - wrec)
        push!(errs, abs( tr(matrix(w)) - tr(matrix(wrec)))/abs(tr(matrix(w))))
    end
end


using Plots

plot(errs)

errs2 = []
for _ = 1:100

    ws = [ITensor(ITransverse.ITenUtils.make_powerlaw(n), i1,i2),
    ITensor(ITransverse.ITenUtils.make_exponential(n), i1,i2),
    ITensor(ITransverse.ITenUtils.make_step(n, 20), i1,i2),
    ITensor(ITransverse.ITenUtils.make_plateau(n, 20), i1,i2),
    ITensor(ITransverse.ITenUtils.make_flat(n), i1,i2),
    ITensor(ITransverse.ITenUtils.randmat_decayspec(n), i1,i2),
    random_itensor(i1,i2)
    ]

    ws ./ 10000

    cutoff = 1e-6
    for w in ws
        f = svd(w, i1; cutoff=cutoff^2*abs2(tr(matrix(w))), use_relative_cutoff=false)
        wrec = f.U * f.S * f.V
        @show dim(f.S,1)
        @show norm(w - wrec)
        push!(errs2, abs( tr(matrix(w)) - tr(matrix(wrec)))/abs(tr(matrix(w))))
    end
end

plot(errs2)
#hline!([sqrt(cutoff)])
hline!([cutoff])
=# 