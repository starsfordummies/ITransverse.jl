
function basic_left_can!(ψ::MPS; normalize::Bool=true)

    # make a few regular truncs to chop bond dim from edges in case
    orthogonalize!(ψ,length(ψ))
    orthogonalize!(ψ,1)


    L = length(ψ)

    for i in 1:L-1
        A = ψ[i]
        left_env = matrix(A * prime(A, linkind(ψ,i)) )
        u, s = basic_sym_svd(left_env)

        tempidx = Index(size(s)[1],"temp")
        linkidx = linkind(ψ,i)

        uL = ITensor(conj(u), linkidx, tempidx)
        isqs = diag_itensor(s.^(-0.5), tempidx, tempidx')
        sqs = diag_itensor(s.^(0.5), tempidx', tempidx)
        uR = ITensor(u, linkidx, tempidx)

        #debug 
        #display(plot_matrix((uL * isqs) * prime(sqs * uR, "temp"), title="id?"))

        ψ[i] = noprime(ψ[i] * uL * isqs)
        ψ[i+1] = noprime(sqs * uR * ψ[i+1])

        #@show i 
        #check_id_matrix(matrix(ψ[i] * prime(ψ[i], linkind(ψ,i))), 1e-12)
        #display(plot_matrix(matrix(ψ[i] * prime(ψ[i], linkind(ψ,i)))))
    end

    if normalize
        ψ[L] = ψ[L] / sqrt( scalar(ψ[L]*ψ[L])) 
    end
end


function check_gen_left_can_form(ψ::MPS)
    for i in eachindex(ψ)[1:end-1]
        #@show i 
        check_id_matrix(matrix(ψ[i] * prime(ψ[i], linkind(ψ,i))), 1e-12)
        display(plot_matrix(matrix(ψ[i] * prime(ψ[i], linkind(ψ,i)))))
    end
    @info "Overlap: $(overlap_noconj(ψ,ψ)) =? $(scalar(ψ[end] * ψ[end])) " 
end

function check_gen_right_can_form(ψ::MPS)
    for i in reverse(eachindex(ψ))[1:end-1]
        #@show i 
        check_id_matrix(matrix(ψ[i] * prime(ψ[i], linkind(ψ,i-1))), 1e-12)
        display(plot_matrix(matrix(ψ[i] * prime(ψ[i], linkind(ψ,i-1)))))
    end
    @info "Overlap: $(overlap_noconj(ψ,ψ)) =? $(scalar(ψ[1] * ψ[1])) " 

end

