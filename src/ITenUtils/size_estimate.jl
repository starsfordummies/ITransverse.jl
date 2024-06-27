# Now this is a silly way to estimate the memory requirement for an MPS

function mem_req(N::Int, chi::Int, phys_dim::Int=2, dtyp::Type=ComplexF64)
    elem_size = sizeof(dtyp)  # in bytes
    ten_size = elem_size*chi*phys_dim*chi
    mps_size = N*ten_size
    mps_size_kb = mps_size/1024
    mps_size_mb = mps_size_kb/1024
    mps_size_gb = mps_size_mb/1024

    return mps_size_gb
end

function mem_req(psi::MPS)
    Base.summarysize(psi)/1024/1024/1024
end

mem_req(1_000, 100, 2)