""" Floquet Ising exp(-iJXX - iλX)exp(-igZ) """
function expH_ising_floquet(sites::Vector{<:Index}, JXX::Number, gz::Number, λx::Number; dt=1.0)

    Uxx = expXX_murg(sites, -JXX; dt)

    # Recall Ra(theta) = exp(-i sigma_a(theta/2))
    Ux = MPO([op(s, "Rx", θ=2*λx*dt) for s in sites])
    Uz = MPO([op(s, "Rz", θ=2*gz*dt) for s in sites])

    U_t = iszero(λx) ? Uxx : applyn(Ux, Uxx) 
    U_t = iszero(gz) ? U_t : applyn(Uz, U_t) 

    return U_t

end
