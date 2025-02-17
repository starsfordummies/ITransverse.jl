""" Given `s` input sites and an MPO defined on three sites (Wl-Wc-Wr),
extends it on the `s` input sites as (Wl-Wc-Wc-...-Wc-Wr) """
function extend_mpo(s::Vector{<:Index}, w::MPO)

    N = length(s)
    @assert N > 3
    @assert length(w) == 3 

    vR, vL = linkinds(w)
    
    @assert dim(vL) == dim(vR)

    link_dimension = dim(vL)
    link_indices = [Index(link_dimension, "Link,l=$(n)") for n = 1:N-1]

    wlist = [w[2] for _ in 1:N] 
    wlist[1] = w[1] * delta(siteind(w,1), s[1]) 
    wlist[1] *= delta(siteind(w,1)', s[1]') 
    wlist[1] *= delta(vR, link_indices[1])
    wlist[end] = w[end] * delta(siteind(w,3), s[end])
    wlist[end] *= delta(siteind(w,3)', s[end]')
    wlist[end] *= delta(vL, link_indices[end])

    for ii in 2:N-1
        wlist[ii] *= delta(siteind(w,2), s[ii])
        wlist[ii] *= delta(siteind(w,2)', s[ii]')
        wlist[ii] *= delta(vL, link_indices[ii-1])
        wlist[ii] *= delta(vR, link_indices[ii])
    end

    wmpo = MPO(wlist)

    return wmpo
end



up_state = Vector{ComplexF64}([1, 0])
down_state = Vector{ComplexF64}([0, 1])
plus_state = Vector{ComplexF64}([1 / sqrt(2), 1 / sqrt(2)])
minus_state = Vector{ComplexF64}([1 / sqrt(2), -1 / sqrt(2)])