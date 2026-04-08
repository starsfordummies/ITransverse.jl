""" Functions which should be overloaded by extensions (ugly)""" 

# function plot_matrix end
# function plotr end 
# function ploti end 
# function plotri end


function togpu end

""" Move x to CPU. Uses NDTensors.cpu  """
tocpu(x) = NDTensors.cpu(x)
