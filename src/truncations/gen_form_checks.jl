
""" Check that two MPS are in (generalized-symmetric) *left* canonical form """
function check_gencan_left(psi::MPS, phi::MPS; verbose::Bool=false)

    is_left_gencan = true
    @assert length(psi) == length(phi)

    phi = match_siteinds(psi, sim(linkinds, phi))

    if verbose
        @info "Checking LEFT gen/sym form"
    end
    
    if abs(overlap_noconj(psi,phi) - 1.) > 1e-7 || abs(1. - scalar(psi[end]*phi[end]*delta(linkinds(psi)[end],linkinds(phi)[end] ))) > 1e-7
        println("overlap = $(overlap_noconj(psi,phi))), alt = $(scalar(psi[end]*phi[end]*delta(linkinds(psi)[end],linkinds(phi)[end] )))")
    end

    # Start from the left 
    left_env = ITensor(1.)
    for (ii, (Ai,Bi)) in enumerate(zip(psi[1:end-1], phi[1:end-1]))
        left_env =  left_env * Ai
        left_env = left_env * prime(Bi, commoninds(Bi,linkinds(phi)))    #* delta(wLa, wLb) )
        @assert order(left_env) == 2
        if norm(array(left_env)- diagm(diag(array(left_env)))) > 0.1
            @warn("[L]non-diag@[$ii]")
            is_left_gencan = false
        end
        delta_norm = norm(array(left_env) - I(size(left_env)[1])) 
        if delta_norm > 0.0001
            @warn("[L]non-can@[$ii], $delta_norm")
            @show array(left_env)
            is_left_gencan = false
        end
    end
    if verbose
        @info("Done checking RIGHT gen/sym form")
    end
    
    return is_left_gencan
end


""" Check that two MPS are in (generalized-symmetric) *right* canonical form
TODO not implemented yet  """
function check_gencan_right(psi::MPS, phi::MPS; verbose::Bool=false)

    is_right_gencan = true 

    if verbose
        @info "Checking RIGHT gen/sym form"
    end
    
    if abs(overlap_noconj(psi,phi) - 1.) > 1e-7 || abs(1. - scalar(psi[end]*phi[end]*delta(linkinds(psi)[end],linkinds(phi)[end] ))) > 1e-7
        println("overlap = $(overlap_noconj(psi,phi))), alt = $(scalar(psi[end]*phi[end]*delta(linkinds(psi)[end],linkinds(phi)[end] )))")
    end

    # Start from right side
    right_env = ITensor(1.)
    for (ii, (Ai,Bi)) in enumerate(zip(psi[end:-1:1], phi[end:-1:1]))
        right_env =  right_env * Ai
        right_env = right_env * prime(Bi, commoninds(Bi,linkinds(phi)))    #* delta(wLa, wLb) )
        @assert order(right_env) == 2
        if norm(array(right_env)- diagm(diag(array(right_env)))) > 0.1
            @warn("[R]non-diag@[$ii]")
            is_right_gencan = false
        end
        delta_norm = norm(array(right_env) - I(size(right_env)[1])) 
        if delta_norm > 0.01
            @warn("[R]non-can@[$ii], $delta_norm")
            is_right_gencan = false
        end
    end
    if verbose
        @info("Done checking RIGHT gen/sym form")
    end
    
    return is_right_gencan
end
