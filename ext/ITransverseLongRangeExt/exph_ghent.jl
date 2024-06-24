# using LongRangeITensors

#= 
"""
Builds exp(Hising) with 2nd order approximation from the Ghent group.
Bond dimension is 7 
"""
function build_expH_ising_2o_Jan(sites, 
    J::Real, f::Real,
    dt::Number)


    #(sites,
    # AStrings::Vector{String}, JAs::Vector{<:Number}, 
    # BStrings::Vector{String}, JBs::Vector{<:Number}, 
    # CStrings::Vector{String}, JCs::Vector{<:Number}, 
    # DString::String, JD::Number, 
    # t::Number)

    U_t = timeEvo_MPO_2ndOrder(sites, 
    ["Id"], [0.], 
    ["X"], [-J],
    ["X"], [1.],
    "Z", -f,
    dt)

    return U_t
end
=#




"""
Builds exp(Hpotts) with 2nd order approximation from the Ghent group.
Bond dimension is 7 
"""
function build_expH_potts_2o_Jan(sites, 
    J::Real, f::Real,
    dt::Number)

    #(sites,
    # AStrings::Vector{String}, JAs::Vector{<:Number}, 
    # BStrings::Vector{String}, JBs::Vector{<:Number}, 
    # CStrings::Vector{String}, JCs::Vector{<:Number}, 
    # DString::String, JD::Number, 
    # t::Number)

    
    U_t = timeEvo_MPO_2ndOrder(sites, 
    ["Id", "Id"], [0., 0], 
    ["Σ","Σdag"], [-J, -J],
    ["Σdag", "Σ"], [1., 1.],
    "τplusτdag", -f,
    dt)

    return U_t
end



