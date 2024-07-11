struct ConeParams
    truncp::TruncParams
    opt_method::String
    sweep_method::String
    op::Vector{ComplexF64}
    which_evs::Vector{String}
    which_ents::Vector{String}
    save_cp::Int
end

ConeParams( truncp::TruncParams,
opt_method::String = "RTM",
sweep_method::String = "SVD",
op::Vector{ComplexF64} = [1,0,0,1],
which_evs::Vector{String} = ["X"],
which_ents::Vector{String}= [],
save_cp::Int=20) = ConeParams(truncp, opt_method, sweep_method, op, which_evs, which_ents, save_cp)
