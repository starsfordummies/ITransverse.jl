function shrink_cone!(cc::Columns, left_envs::Environments, right_envs::Environments)

    if maximum([maxlinkdim(left_envs[1]),maxlinkdim(right_envs[1]),maxlinkdim(right_envs[end]),maxlinkdim(right_envs[end])]) == 1 

        @info "Trimming edges.. "
      
        popfirst!(left_envs)
        popfirst!(right_envs)
        pop!(left_envs)
        pop!(right_envs)
    end

end

