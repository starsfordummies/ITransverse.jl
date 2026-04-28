""" Floquet Ising exp(-iJXX - iλX)exp(-igZ) """
function expH_ising_floquet(sites::Vector{<:Index}, mp::IsingParams; dt=1.0)
    (; Jtwo, gperp, hpar) = mp

    Uxx = expXX_murg(sites, -Jtwo; dt)

    # Recall Ra(theta) = exp(-i sigma_a(theta/2))
    Ux = MPO([op(s, "Rx", θ=2*hpar*dt) for s in sites])
    Uz = MPO([op(s, "Rz", θ=2*gperp*dt) for s in sites])

    U_t = iszero(hpar) ? Uxx : applyn(Ux, Uxx) 
    U_t = iszero(gperp) ? U_t : applyn(Uz, U_t) 

    return U_t

end
