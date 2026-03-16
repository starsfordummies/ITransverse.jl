"""
Bring the MPS to generalized *left* canonical form without truncating (as far as possible)
This is symmetric, so we use symmetric eigenvalue decomposition, A => O D O^T with O complex orthogonal 
"""
function gen_canonical_left(in_mps::MPS)  # TODO: polar decomp?
    temp = deepcopy(in_mps)
    psi_leftgencan, _ = truncate_lsweep_sym(temp; cutoff=1e-14, maxdim=2*maxlinkdim(in_mps), method="EIG")
    return psi_leftgencan
end

"""
Just bring the MPS to generalized *right* canonical form without truncating (as far as possible)
TODO should use chi_min here to make sure ! 
"""
function gen_canonical_right(in_mps::MPS)
    psi_rightgencan, _ = truncate_rsweep_sym(in_mps; cutoff=1e-14, maxdim=2*maxlinkdim(in_mps), method="EIG")
    return psi_rightgencan
end
