using ITensors, JLD2
using LinearAlgebra
using KrylovKit: eigsolve
using Plots


if Base.Sys.islinux()
    println("On linux - setting MKL and 8 threads")
    using MKL
    using LinearAlgebra

    BLAS.set_num_threads(8)  # 8 threads seems to be a sweet spot
    #unicodeplots()
else
    using LinearAlgebra
end


include("../../power_method/utils.jl")
include("../../models/ising.jl")


function plot_matrix(a::Matrix; xlabel="i", ylabel="j", title="matrix")
    heatmap(1:size(a,1), 1:size(a,2), abs.(a),
        c=cgrad([:blue, :white,:red, :yellow]),
        xlabel=xlabel, ylabel=ylabel,
        title=title)
end

function plot_matrix(a::ITensor; title="tensor")
    if order(a) != 2
        @warn("error tensor has inds: $(inds(a))")
    else
        plot_matrix(matrix(a), title=title)
    end
end

function mat_to_iten(a::Matrix)

    iL = Index(size(a,1), tags="L")
    iR = Index(size(a,2), tags="R")

    aT = ITensor(a, iL, iR)

    return aT, iL, iR
end


function check_id_matrix(d::Matrix, cutoff::Float64=1e-6)
    if size(d,1) == size(d,2)
        delta_diag = norm(d - I(size(d,1)))/norm(d)
        if delta_diag > cutoff
            println("Not identity: off by(norm) $delta_diag")
            return false
        end
        return true
    else
        println("Not even square? $(size(d))")
        return false
    end
end


function myoverlap(ψ::MPS, ϕ::MPS, conjugate::Bool=true)
    @assert length(ψ) == length(ϕ)

    overlap = ITensor(1.)
    for ii in eachindex(ψ)
        if conjugate
            overlap *= dag(ψ[ii])
        else
            overlap *= ψ[ii]
        end
        overlap *= prime(ϕ[ii],commoninds(ϕ[ii], linkinds(ϕ))) # prime just in case
    end

    return scalar(overlap)
end
