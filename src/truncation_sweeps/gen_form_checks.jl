""" Check that two MPS are in (generalized-symmetric) *left* canonical form """
function check_gencan_left(psi::MPS, phi::MPS; verbose::Bool=false)

    #TODO that's another way of doing it ...
    dtype = mapreduce(NDTensors.unwrap_array_type, promote_type, psi)
    #adapt(dtype,...)

    mpslen = length(psi)

    is_left_gencan = true
    is_left_s = Bool[]

    @assert length(psi) == length(phi)

    phi = match_siteinds(psi, sim(linkinds, phi))

    if verbose
        @info "Checking LEFT gen/sym form"
    end
    

    # Start from the left 
    left_env = (ITensor(1.))
    for ii = 1:mpslen-1
        left_env *= psi[ii]
        left_env *= phi[ii]
        @assert order(left_env) == 2
        
        is_left_gencan = check_diag_matrix(matrix(left_env))
        is_left_gencan = check_id_matrix(matrix(left_env))

        push!(is_left_s, is_left_gencan)
    end

    if verbose
        @info("Done checking LEFT gen/sym form")
    end

    is_left_gencan = all(is_left_s)

    if is_left_gencan
        @info("overlap = $(overlap_noconj(psi,phi))), 
        alt = $(scalar(psi[end]* (phi[end]*left_env) )))")
    end
    
    return is_left_gencan
end


""" Check that two MPS are in (generalized-symmetric) *right* canonical form"""
function check_gencan_right(psi::MPS, phi::MPS; verbose::Bool=false)

    mpslen = length(psi)

    is_right_s = Bool[] 

    # this should make that psi and phi have different linkinds but same siteinds
    phi = match_siteinds(psi, sim(linkinds, phi))

    if verbose
        @info "Checking RIGHT gen/sym form"
    end
    

    # Start from right side
    right_env = ITensor(1.)
    for ii = mpslen:-1:2
        right_env *= psi[ii]
        right_env *= phi[ii]

        @assert order(right_env) == 2

        is_right_gencan = check_diag_matrix(matrix(right_env))
        is_right_gencan = check_id_matrix(matrix(right_env))

        push!(is_right_s, is_right_gencan)
    end

    if verbose
        @info("Done checking RIGHT gen/sym form")
    end

    if all(is_right_s)
        @info ("overlap = $(overlap_noconj(psi,phi))), 
        alt = $(scalar((right_env * psi[1])*phi[1]))")
    end
    
    return all(is_right_s)
end
