
function build_lenvs(psi::MPS, phi::MPS)
    lenvs =[] 

    left_env = ITensor(1.)
    for (ii, (Ai,Bi)) in enumerate(zip(psi[1:end-1], phi[1:end-1]))
        left_env =  left_env * Ai
        left_env = left_env * prime(Bi, commoninds(Bi,linkinds(phi)))   
        push!(lenvs, left_env)
    end

    return lenvs
end

function build_renvs(psi::MPS, phi::MPS)
    renvs = [] 

    right_env = ITensor(1.)
    for (ii, (Ai,Bi)) in enumerate(zip(psi[end:-1:1], phi[end:-1:1]))
        right_env =  right_env * Ai
        right_env = right_env * prime(Bi, commoninds(Bi,linkinds(phi)))  
        push!(renvs, right_env)
    end
    return renvs
end

function print_norms_envs(psi::MPS, phi::MPS)
    lenvs = build_lenvs(psi, phi)
    renvs = build_renvs(psi, phi)

    for (ii, (Ai,Bi)) in enumerate(zip(lenvs, renvs))
        @info ii, norm(Ai), norm(Bi)
    end
end
