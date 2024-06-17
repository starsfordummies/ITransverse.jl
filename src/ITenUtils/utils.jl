







# function check_diag_matrix(d::Matrix, cutoff::Float64=1e-6)
#     delta_diag = norm(d - Diagonal(d))/norm(d)
#     if delta_diag > cutoff
#         println("Warning, matrix non diagonal: $delta_diag")
#         return false
#     end
#     return true
# end


# function check_id_matrix(d::Matrix, cutoff::Float64=1e-6)
#     if size(d,1) == size(d,2)
#         delta_diag = norm(d - I(size(d,1)))/norm(d)
#         if delta_diag > cutoff
#             println("Not identity: off by(norm) $delta_diag")
#             return false
#         end
#         return true
#     else
#         println("Not even square? $(size(d))")
#         return false
#     end
# end

# function isid(a::ITensor, cutoff::Float64=1e-8)
#     @assert ndims(a) == 2
#     @assert size(a,1) == size(a,2)

#     am = array(a)

#     check_id_matrix(am)
# end

""" Takes output of a funcion and saves all elements in a jld2 """
function cpsave(myfunc::Function, cp_name = "checkpoint.jld2")
    #TODO change default filename as date/time ? 
    f = myfunc()
    jldsave(cp_name; f)
    @info "Saving checkpoint to $(cp_name)"
end
