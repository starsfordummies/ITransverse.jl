""" Given an input MPS, computes the spectra of its (normalized) RDM via SVD decompositions - these
can be used to calculate the usual entanglement entropies""" 
function diagonalize_rdm(psi::MPS)

    workpsi = orthogonalize(psi,1) 
    workpsi = normalize(workpsi)

    evs_rho = Vector{Float64}[]

    for icut=1:length(workpsi)-1
        push!(evs_rho, diagonalize_rdm!(workpsi, icut))
    end

    return evs_rho
end

""" At a given cut, performs SVD and returns the squares of the SVs, ie. the eigenvalues of the RDM"""
function diagonalize_rdm!(psi::MPS, cut::Int)

    orthogonalize!(psi, cut)

    _,S,_ = svd(psi[cut], (linkinds(psi, cut-1)..., siteinds(psi,cut)...))
    eigenvals_rho = S.^2 
   
    return array(diag(eigenvals_rho))
end


vn_from_sv(sv::ITensor; normalize::Bool) = vn_from_sv(tensor(NDTensors.cpu(sv)); normalize)

function vn_from_sv(sv::Tensor; normalize::Bool)

    if normalize
        sv = sv/norm(sv)
    end

   SvN = zero(eltype(sv))
    for n=1:dim(sv, 1)
        p = sv[n,n]^2
        SvN -= p * log(p)
    end
    return SvN
end

""" MPS-modifying VN entropy (orthogonalizes), by default assumes that MPS is already normalized """ 
function vn_entanglement_entropy!(psi::MPS, bond::Int; normalize::Bool=false)
    orthogonalize!(psi, bond)
    _,S,_ = svd(psi[bond], uniqueinds(psi[bond],psi[bond+1]))
    return vn_from_sv(S; normalize) 
end


""" Computes the Von Neumann entanglement entropy of an MPS `psi` at all links (normalizing if necessary), 
returns a vector of floats containing the VN entropies 
"""
function vn_entanglement_entropy(psi::MPS)

    workpsi = normalize(psi)

    ents_vn = Vector{Float64}()

    for icut=1:length(workpsi)-1
        Si = vn_entanglement_entropy!(workpsi, icut, normalize=false)
        # no need to normalize if we normalize psi already before
        push!(ents_vn, Si)
    end

    return ents_vn
end


function renyi_entropy(in_psi::MPS, cut::Int, αr::Number)

    S_ren = 0.0

    psi = normalize(in_psi)

    if αr == 1  # VN entropy
        S_ren = vn_entanglement_entropy!(psi, cut)
    else  # Renyi n
            
        orthogonalize!(psi, cut)
        #println(norm(psi))

        if cut == 1
            _,S,_ = svd(psi[cut], (siteind(psi,cut)))
        else
            _,S,_ = svd(psi[cut], (linkind(psi, cut-1), siteind(psi,cut)))
        end

        S2α = S.^(2*αr)

        sum_sN = sum(S2α)
    
        S_ren = log(sum_sN)/(1-αr)

    end

    return S_ren
end


""" Computes the `α`-th Renyi entanglement entropy of an MPS `psi` at all links, 
S_α = -log(sum λ^α), where λ are the eigenvalues of the RDM (=SV^2 ).
returns a vector of floats containing the entropies 
"""
function renyi_entropy(psi::MPS, α::Number=2)

    workpsi = normalize(psi)

    ents_renyi = Vector{Float64}()

    for icut=1:length(workpsi)-1
        Si = renyi_entropy(workpsi, icut, α)
        push!(ents_renyi, Si)
    end

    return ents_renyi
end


function renyi_entropies(in_psi::MPS; which_ents = [0.5, 1, 2])

    renyi_entropies(diagonalize_rdm(in_psi); which_ents)
    
end
