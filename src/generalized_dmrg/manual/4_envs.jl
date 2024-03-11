
""" Generalized Left environment L-O-L """
function build_left_gen_env(psi::MPS, h::MPO)
    @assert length(psi) == length(h)
    #@assert siteinds(psi) == siteinds(h)

    @info "calculating Lenv"
    temp_left = ITensor(1.)

    left_env = fill(temp_left, length(psi))

    for i in eachindex(psi)

        a = psi[i]
        w = h[i]

        temp_left = temp_left * a
        temp_left = temp_left * w 
        temp_left = temp_left * a'

        left_env[i] = temp_left

    end

    return left_env
end

function update_left_gen_env!(psi::MPS, h::MPO, lenv, ii::Int)
    @info "updating Lenv[$(ii)]"
    LL = length(psi)
    temp_lenv = get_lenv(lenv,ii-1) * psi[ii]
    temp_lenv *= h[ii] 
    temp_lenv *= psi[ii]'

    lenv[ii] = temp_lenv

    return lenv
end


function update_right_gen_env!(psi::MPS, h::MPO, renv, ii::Int)
    @info "updating Renv[$(ii)]"
    LL = length(psi)
    temp_renv = get_renv(renv, ii+1) 

    @debug @show inds(temp_renv)
    @debug @show inds(psi[ii])
    @debug @show inds(h[ii])

    temp_renv = temp_renv * psi[ii]
    temp_renv *= h[ii] 
    temp_renv *= psi[ii]'

    renv[ii] = temp_renv

    @debug @show inds(temp_renv)

    return renv
end


function build_right_gen_env(psi::MPS, h::MPO, stop::Int=1)
    @assert length(psi) == length(h)

    @info "calculating Renv"

    ITensors.check_hascommoninds(siteinds, h, psi)
    ITensors.check_hascommoninds(siteinds, h, psi')
    
    #@assert siteinds(psi) == siteinds(h)

    temp_right = ITensor([1.])

    right_env = fill(temp_right, length(psi))

    for i in length(psi):-1:stop

        a = psi[i]
        w = h[i]

        temp_right = temp_right * a
        temp_right = temp_right * w 
        temp_right = temp_right * a'

        right_env[i] = temp_right

    end

    return right_env
end



function get_lenv(lenv, j::Int)
    @assert j <= length(lenv) #there's something wrong in the f call otherwise

    if j < 1
        return ITensor(1.)
    else
        return lenv[j]
    end
end

function get_renv(renv, j::Int)
    @assert j > 0 #there's something wrong in the f call otherwise

    if j > length(renv)
        return ITensor(1.)
    else
        return renv[j]
    end
end


