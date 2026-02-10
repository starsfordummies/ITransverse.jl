""" Check that two MPS are in (generalized-symmetric) *left* canonical form """
function check_gencan_left(psi::MPS, phi::MPS=psi; verbose::Bool=false)

    mpslen = length(psi)

    is_left_s = falses(mpslen-1)

    @assert length(psi) == length(phi)

    phi = match_siteinds(psi, sim(linkinds, phi))

    if verbose
        @info "Checking LEFT gen/sym form"
    end

    # Start from the left 
    left_env = ITensors.OneITensor()
    for ii = 1:mpslen-1
        left_env *= psi[ii]
        left_env *= phi[ii]
        @assert order(left_env) == 2
        
        #is_left_gencan = isapproxdiag(matrix(left_env))
        is_left_s[ii] = check_id_matrix(matrix(left_env))

    end

    if verbose
        @info("Done checking LEFT gen/sym form")
        @info is_left_s
    end

    is_left_gencan = all(is_left_s)

    if is_left_gencan
        @info("overlap = $(overlap_noconj(psi,phi))), 
        alt = $(scalar(psi[end]* (phi[end]*left_env) )))")
    end
    
    return is_left_gencan
end


""" Check that two MPS are in (generalized-symmetric) *right* canonical form"""
function check_gencan_right(psi::MPS, phi::MPS=psi; verbose::Bool=false)

    mpslen = length(psi)

    is_right_s = falses(mpslen-1)

    # this should make that psi and phi have different linkinds but same siteinds
    phi = match_siteinds(psi, sim(linkinds, phi))

    if verbose
        @info "Checking RIGHT gen/sym form"
    end
    

    # Start from right side
    right_env = ITensors.OneITensor()
    for ii = mpslen:-1:2
        right_env *= psi[ii]
        right_env *= phi[ii]

        @assert order(right_env) == 2

        #is_right_gencan = isapproxdiag(matrix(right_env))
        is_right_s[ii-1] = is_right_gencan = check_id_matrix(matrix(right_env))

    end

    if verbose
        @info("Done checking RIGHT gen/sym form")
        @info is_right_s
    end

    if all(is_right_s)
        @info ("overlap = $(overlap_noconj(psi,phi))), 
        alt = $(scalar((right_env * psi[1])*phi[1]))")
    end
    
    return all(is_right_s)
end
