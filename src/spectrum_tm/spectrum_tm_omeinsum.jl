if Base.Sys.islinux()
    println("On linux - setting MKL and 8 threads")
    using MKL
    using LinearAlgebra

    BLAS.set_num_threads(8)  # 8 threads seems to be a sweet spot

else
    using LinearAlgebra
end

using JLD2
using KrylovKit: eigsolve
using Plots
using OMEinsum

include("../power_method/utils.jl")
include("../generalized_dmrg/manual/2_trunc.jl") # mostly for ortho_eigen.. 

const sigma_x = [0 1. ; 1 0]
const sigma_z = [1. 0 ; 0 -1]
const id = [1. 0 ; 0 1]



""" Builds Wi of Ising exp(iHt) MPO  H = XX + Z . 
Returns Wi, Wi_beta (with dtau = -idt) """
function build_Wi_manual(hz, dt)


    cosg = cos(hz*dt*0.5)
    sing = sin(hz*dt*0.5)


    combz = (1 - 2*sing^2)*id + im*2*sing*cosg*sigma_z
    X = sigma_x

    Wi = zeros(ComplexF64, 2,2,2,2)

    Wi[1,1,:,:] = cos(dt)*combz
    Wi[1,2,:,:] = sqrt(im*sin(dt))*sqrt(cos(dt))*X
    Wi[2,1,:,:] = sqrt(im*sin(dt))*sqrt(cos(dt))*X
    Wi[2,2,:,:] = im*sin(dt)*combz


    # Imaginary time part
    dt = -im*dt

    cosg = cos(hz*dt*0.5)
    sing = sin(hz*dt*0.5)

    combz = (1 - 2*sing^2)*id + im*2*sing*cosg*sigma_z

    Wi_beta = zeros(ComplexF64, 2,2,2,2)

    Wi_beta[1,1,:,:] = cos(dt)*combz
    Wi_beta[1,2,:,:] = sqrt(im*sin(dt))*sqrt(cos(dt))*X
    Wi_beta[2,1,:,:] = sqrt(im*sin(dt))*sqrt(cos(dt))*X
    Wi_beta[2,2,:,:] = im*sin(dt)*combz


    return Wi, Wi_beta
end




function build_transverse_tm(nn, nbeta=0, trace=false)
    
    wmi, wmi_im =  build_Wi_manual(1., 0.1)

    if nbeta == 0
        wx = wmi
    else
        wx = wmi_im
    end

    for jj = 2:nn
        wx = ein"lrpx,λsxπ->lλrspπ"(wx, wmi) 
        wx = reshape(wx,(2^jj,2^jj,2,2))
    end

    if trace
        wx = ein"lrpp->lr"(wx)
    end

    return wx
end



""" Builds temporal TM with an insertion of operator `op`
in the physical legs between site `tinsert-1` and `tinsert`
"""
function build_transverse_tm_op_phys(nn::Int, tinsert::Int, op::Matrix, trace::Bool=true)

    @assert tinsert <= nn
    @assert tinsert > 1 

    wmi, wmi_beta = build_Wi_manual(1, 0.1)

    wx = wmi

    for jj = 2:nn
        # insert additional operator on top of physical leg of TM W 
        if jj == tinsert
            wx = ein"lrpx,xπ->lrpπ"(wx, op)
        end
        wx = ein"lrpx,λsxπ->lλrspπ"(wx, wmi) 

        wx = reshape(wx,(2^jj,2^jj,2,2))
    end

    if trace
        wx = ein"lrpp->lr"(wx)
    end

    return wx
end


""" builds a vertical matrix with just identities except for an operator `op`
    on the virtual (space) leg of the `ninsert` (default=1) site 
"""
function build_op_onesite(nn::Int, op::Matrix, ninsert::Int=1 )
    
    @assert ninsert <= nn 
    @assert ninsert >= 1

    if ninsert == 1 
        mat = op
    else
        mat = id
    end

    for jj = 2:nn
        if jj == ninsert
            mat = ein"lr,ab->larb"(mat,op) 
        else
            mat = ein"lr,ab->larb"(mat,id) 
        end
        mat = reshape(mat,(2^jj,2^jj))
    end

    return mat
end

    

""" Builds transverse TM with insertion of `op` on `n_insert`-th virtual (space) leg.
Same as product TM * build_op_onesite(nn, op) if we put the op on 1st tensor
"""
function build_transverse_tm_with_op_virt(nn::Int, n_insert::Int, op::Matrix)
    wxTr = build_transverse_tm(nn, 0, true)
    @show size(wxTr)
    wxTr = reshape(wxTr, ( 2^(n_insert-1), 2, 2^(nn-n_insert), 2^(n_insert-1), 2, 2^(nn-n_insert)))
    wxTr = ein"abcjxl,xk->abcjkl"(wxTr,op) 

    return reshape(wxTr,(2^nn, 2^nn))

end



""" Diagonalizes periodic TM 
and builds correlations between operators put on virtual (space) legs """
function main_correl_virt(nmax::Int, nevals::Int, opp::Matrix, nbeta::Int=0)

    @assert nbeta < nmax
    valss = []
    gss = [] 
    corrs_allT = []
    corrs_allT_2 = []

    for jj = 1:nmax

        wxTr= build_transverse_tm(jj, nbeta, true)

        @info "[$(jj)] Mem usage TM (GB) = $(Base.summarysize(wxTr)/1024/1024/1024)"

        
        if size(wxTr)[1] > 2100
            @info "matrix is large, using eigsolve"
            vals, vecs = eigsolve(wxTr, nevals, :LM, ComplexF64)
            gs = vecs[1]
            gs2 = vecs[2]
        else
            vals,vecs = eigen(wxTr, sortby= x->-abs(x)) 
            gs = vecs[:,1]
            gs2 = vecs[:,2]
        end
        
        #chop to first N eigvals
        if length(vals) > nevals
            vals = vals[1:nevals]
        end
        
        push!(valss, vals)

        @assert transpose(gs) * wxTr * gs / (transpose(gs) * gs) ≈ vals[1]

        push!(gss, gs)

        # build corelations 
        if jj > 1 # build correls only for longer T 

            one_op = build_op_onesite(jj, opp, jj)


            corrs = []
            temp_num = one_op * gs
            temp_den = gs

            for distance = 0:40
                resu = transpose(gs) * one_op * temp_num / (transpose(gs) * temp_den)
                push!(corrs, resu)

                temp_num = wxTr * temp_num
                temp_den = wxTr * temp_den
            end
            push!(corrs_allT, corrs)

            # with 2nd largest (just in case.. )
            corrs = []
            temp_num = one_op * gs2
            temp_den = gs2

            for distance = 0:40
                resu = transpose(gs2) * one_op * temp_num / (transpose(gs2) * temp_den)
                push!(corrs, resu)

                temp_num = wxTr * temp_num
                temp_den = wxTr * temp_den
            end
            push!(corrs_allT_2, corrs)

        end

        #jldsave("spectrum_tm_wcorrs_nbeta_$(nbeta).jld2" ; valss, gss, corrs_allT, corrs_allT_2)
    end

    return valss, gss, corrs_allT, corrs_allT_2

end




""" Diagonalizes periodic TM 
and builds correlations between operators put on physical (time) legs """
function main_correl_phys(nmax::Int, nevals::Int, opp::Matrix, nbeta::Int=0)

    @assert nbeta < nmax
    valss = []
    gss = [] 

    corrs_allT_onept = []
    corrs_allT = []
    corrs_allT_2 = []

    for jj = 1:nmax

        wxTr= build_transverse_tm(jj, nbeta, true)

        @info "[$(jj)] Mem usage TM (GB) = $(Base.summarysize(wxTr)/1024/1024/1024)"
        
        if size(wxTr)[1] > 2100
            @info "matrix is large, using eigsolve"
            vals, vecs = eigsolve(wxTr, nevals, :LM, ComplexF64)

            gs = vecs[1]
            gs2 = vecs[2]

        else
            vals,vecs = eigen(wxTr, sortby= x->-abs(x)) 

            gs = vecs[:,1]
            gs2 = vecs[:,2]

        end
        
        
        #chop to first N eigvals
        if length(vals) > nevals
            vals = vals[1:nevals]
        end
        
        push!(valss, vals)

        push!(gss, gs)

        if jj > 1 # build correls only for longer T 


            corrs_onept = []
            for pos_x = 2:jj
                one_op = build_transverse_tm_op_phys(jj, pos_x, opp)
                resu = transpose(gs) * one_op * gs / (transpose(gs) * gs)
                push!(corrs_onept, resu)
            end
            push!(corrs_allT_onept, corrs_onept)


            wxOp = build_transverse_tm_op_phys(jj, 2, opp)

            corrs = []
            temp_num = wxOp * gs
            temp_den = wxTr * gs
            for distance = 1:40
                resu = transpose(gs) * wxOp * temp_num / (transpose(gs)* wxTr * temp_den)
                push!(corrs, resu)
                temp_num = wxTr * temp_num
                temp_den = wxTr * temp_den
            end
            push!(corrs_allT, corrs)

        
            corrs = []
            temp_num = wxOp * gs2
            temp_den = wxTr * gs2
            for distance = 1:40
                resu = transpose(gs2) * wxOp * temp_num / (transpose(gs2)* wxTr * temp_den)
                push!(corrs, resu)
                temp_num = wxTr * temp_num
                temp_den = wxTr * temp_den
            end
            push!(corrs_allT_2, corrs)

        end

        #jldsave("spectrum_tm_onepoint_timecorrel_nbeta_$(nbeta).jld2" ; valss, gss, corrs_allT_1, corrs_allT_2, corrs_allT_onept)
    end

    return valss, gss, corrs_allT, corrs_allT_2, corrs_allT_onept

end

function twopoint_cft(l, T, x=(1/8))
    num = (2π/T)^(2x)
    den = (2*cosh(2π*l/T) - 2)^x

    return num/den
end


vals_v,gss_v,corrs_v,corrs2_v = main_correl_virt(11, 16, sigma_z, 0)

#vals_p,gss_p,corrs_p,corrs2_p, corrs_1s_p  = main_correl_phys(11, 16, sigma_z, 0)
