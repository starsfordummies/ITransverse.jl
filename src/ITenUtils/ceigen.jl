"""
    ceigen(a::ITensor, linds, rinds = uniqueinds(a, linds); kwargs...)

Truncated eigendecomposition for a (generally non-Hermitian, complex) ITensor:
eigenvalues sorted by decreasing |λ| and truncation applied to the absolute
values of the spectrum (via [`mytrunc_eig`](@ref) / `ctruncate!`).

This is a package-local replacement for the former override of
`LinearAlgebra.eigen(::DenseTensor{<:Complex,2})`: same behavior, but under a
name we own, so dispatch for every other ITensors user is left untouched.

Index conventions match `ITensors.eigen`: returns `TruncEigen(D, V, Vt, spec, l, r)`
where `V` carries `(rinds..., r)`, `D` carries `(l, r)` with `l = r'`, and
`Vt` is `V` re-mapped onto `(linds..., l)`, so `a * V ≈ Vt * D` for right
eigenvectors. Assumes dense (non-QN) indices, with `linds`/`rinds` paired in
order with equal dimensions.

Keyword arguments: `maxdim`, `mindim`, `cutoff`, `use_absolute_cutoff`,
`use_relative_cutoff`, `lefttags` (tags of the new eigen index).
"""
function ceigen(a::ITensor, linds, rinds = uniqueinds(a, linds);
        maxdim = nothing,
        mindim = 1,
        cutoff = nothing,
        use_absolute_cutoff = nothing,
        use_relative_cutoff = true,
        lefttags = "Link,ceig")

    Lis = linds isa Index ? (linds,) : Tuple(linds)
    Ris = rinds isa Index ? (rinds,) : Tuple(rinds)

    @assert length(Lis) == length(Ris) "ceigen: need equal numbers of left/right indices"
    no_qns_supported("ceigen (non-hermitian eigendecomposition)", a)
    @assert all(hasind(a, i) for i in (Lis..., Ris...)) "ceigen: indices not found in tensor"

    cL = combiner(Lis...)
    cR = combiner(Ris...)
    iL = combinedind(cL)
    iR = combinedind(cR)

    am = matrix(permute((a * cL) * cR, iL, iR))

    F, spec = mytrunc_eig(am; maxdim, mindim, cutoff, use_absolute_cutoff, use_relative_cutoff)
    DM, VM = F.values, F.vectors

    r = Index(length(DM), lefttags)
    l = r'

    D = diag_itensor(DM, l, r)
    # columns of VM are right eigenvectors; their row index lives in the cR space
    V = ITensor(VM, iR, r) * dag(cR)
    Vt = replaceinds(V, (Ris..., r), (Lis..., l))

    return ITensors.TruncEigen(D, V, Vt, spec, l, r)
end

""" Matrix-like convenience: eigendecompose a 2-index ITensor in ind1 vs ind2 """
function ceigen(a::ITensor; kwargs...)
    @assert ndims(a) == 2
    ceigen(a, ind(a, 1); kwargs...)
end


""" Overriding eigvals()"""
LinearAlgebra.eigvals(T::ITensor) = eigvals(matrix(T))
LinearAlgebra.eigvals(T::ITensor, inds_T) = eigvals(Matrix(T, inds_T...))
