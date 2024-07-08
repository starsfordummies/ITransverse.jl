""" Functions which should be overloaded by extensions (ugly)""" 

function plot_matrix end
function plotr end 
function ploti end 
function plotri end


function gpu_run_cone()
    @info "CUDA not loaded" 
end

togpu(x) = x 