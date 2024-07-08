""" Functions which should be overloaded by extensions (ugly)""" 

function plot_matrix end
function plotr end 
function ploti end 
function plotri end



function togpu(x)
    return x
end 

function tocpu(x)
    dtype = promote_type(NDTensors.unwrap_array_type(x))
    if dtype <: CuArray
        return NDTensors.cpu(x)
    end
    return x
end