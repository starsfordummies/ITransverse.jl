""" Functions which should be overloaded by extensions (ugly)""" 

function plot_matrix end
function plotr end 
function ploti end 
function plotri end


function gpu_run_cone()
    @info "CUDA not loaded" 
end
function gpu_truncate_sweep end
function gpu_truncate_sweep! end
function gpu_expval_LR end
function gpu_expval_LL_sym end
function cpu_expval_LR end
function gpu_run_cone_svd end
function gpu_compute_expvals end