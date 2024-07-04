function diagonalize_rdm(psi::MPS)

    workpsi = normalize(psi)

    evs_rho = Vector{Float64}[]

    for icut=1:length(workpsi)-1
        push!(evs_rho, diagonalize_rdm!(workpsi, icut))
    end

    return evs_rho
end

function diagonalize_rdm!(psi::MPS, cut::Int)

    orthogonalize!(psi, cut)

      _,S,_ = svd(psi[cut], (linkind(psi, cut-1), siteind(psi,cut)))
      eigenvals_rho = S.^2 
   
      return array(diag(eigenvals_rho))
end




""" Computes the Von Neumann entanglement entropy of an MPS `psi` at a given `cut` 
It is a (!) function cause we orthogonalize along the way, gauging `psi` """
function vn_entanglement_entropy!(psi::MPS, cut::Int)

    orthogonalize!(psi, cut)

    if cut == 1
        _,S,_ = svd(psi[cut], (siteind(psi,cut)))
    else
        _,S,_ = svd(psi[cut], (linkind(psi, cut-1), siteind(psi,cut)))
    end

    SvN = 0.0
    for n=1:dim(S, 1)
        p = S[n,n]^2
        SvN -= p * log(p)
    end

    return SvN
end



""" Computes the Von Neumann entanglement entropy of an MPS `psi` at all links, 
returns a vector of floats containing the VN entropies 
"""
function vn_entanglement_entropy(psi::MPS)

    workpsi = normalize(psi)

    ents_vn = Vector{Float64}()

    for icut=1:length(workpsi)-1
        Si = vn_entanglement_entropy!(workpsi, icut)
        push!(ents_vn, Si)
    end

    return ents_vn
end



function renyi_entanglement_entropy!(in_psi::MPS, cut::Int, αr::Int)

    S_ren = 0.0

    psi = normalize(in_psi)

    if αr == 1  # VN entropy
        S_ren = vn_entanglement_entropy_cut!(psi, cut)
    else  # Renyi n
            
        orthogonalize!(psi, cut)
        #println(norm(psi))

        if cut == 1
            _,S,_ = svd(psi[cut], (siteind(psi,cut)))
        else
            _,S,_ = svd(psi[cut], (linkind(psi, cut-1), siteind(psi,cut)))
        end

        sum_sN = 0.0
        for n=1:dim(S, 1)
            p = S[n,n]^2
            sum_sN += p^αr
        end
        S_ren = -log(sum_sN)

    end

    return S_ren
end



""" Computes the `α`-th Renyi entanglement entropy of an MPS `psi` at all links, 
S_α = -log(sum λ^α), where λ are the eigenvalues of the RDM (=SV^2 ).
returns a vector of floats containing the entropies 
"""
function renyi_entanglement_entropy!(psi::MPS, α::Int=2)

    workpsi = normalize(psi)

    ents_renyi = Vector{Float64}()

    for icut=1:length(workpsi)-1
        Si = renyi_entanglement_entropy_cut(workpsi, icut, α)
        push!(ents_renyi, Si)
    end

    return ents_renyi
end

