
""" Resumes a light cone simulation from a checkpoint file """
function resume_cone(checkpoint::String, nT_final::Int, run_on::String="CPU")

    c = jldopen(checkpoint, "r")

    psi = c["rrcp"]
    cp = c["infos"][:coneparams]
    tp = c["infos"][:tp]


    @info "Resuming from $(length(psi)) until $(nT_final)"



    if run_on == "GPU"  
        @info "Trying to run on GPU"
        # tp = NDTensors.cu(tp)
        # psi = NDTensors.cu(psi)
        tp = togpu(tp)
        psi = togpu(psi)
    end

    b = FoldtMPOBlocks(tp)

    # TODO extend with prev results 
    return run_cone(psi, b, cp, nT_final)
    
end
