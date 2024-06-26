""" Builds *rotated* and *folded* MPO for Ising, defined on `time_sites`.
Closed with `fold_op` on the left and `init_state` to the right. 
"""
function build_ising_folded_tMPO(build_expH_function::Function, JXX::Real, hz::Real, 
    dt::Number, 
    init_state::AbstractVector,
    fold_op::AbstractVector,
    time_sites::Vector{<:Index})

    space_sites = siteinds("S=1/2", 3; conserve_qns = false)

    # Real time evolution
    eH = build_expH_function(space_sites, JXX, hz, dt)

    #@info "using $(build_expH_function)"
    build_folded_tMPO(eH, init_state, fold_op, time_sites)

end

function build_ising_folded_tMPS(build_expH_function::Function, par::pparams,
    time_sites::Vector{<:Index})

    space_sites = siteinds("S=1/2", 3; conserve_qns = false)

    # Real time evolution
    #eH = build_expH_function(space_sites, JXX, hz, dt)
    eH = build_expH_function(space_sites, par.JXX, par.hz, par.dt)

    #@info "using $(build_expH_function)"
    build_folded_left_tMPS(eH, par.init_state, time_sites)

end

function build_ising_folded_tMPO(build_expH_function::Function, p::pparams,
    fold_op::Vector{ComplexF64},
    time_sites::Vector{<:Index})

    build_ising_folded_tMPO(build_expH_function, p.JXX, p.hz, p.dt, p.init_state, fold_op, time_sites)

end

