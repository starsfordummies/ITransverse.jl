struct TruncatedMPS{T<:Number}
    psi::MPS 
    SV::Matrix{T}
end

TruncatedMPS(psi::MPS; SV=zeros(Float64, 1,1)) = TruncatedMPS(psi, SV)
