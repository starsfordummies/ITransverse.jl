using ITensors, JLD2
using KrylovKit: eigsolve
using Plots
using Tullio

ITensors.set_warn_order(24)

function _periodic_tmpo_ising(gz, dt, mpolen)

    aa = siteinds("S=1/2",50)
    pars = pparams(1., gz, dt, 1, [1,0])
    h_aa = build_expH_ising_murg(aa, pars)

    tmpo = h_aa[4]
    for ii = 5:4+mpolen
        tmpo *= h_aa[ii]
    end

    tmpo *= delta(inds(tmpo,"Link"))


    comb = combiner(inds(tmpo, tags="Site", plev=0))
    tmpo *= comb
    tmpo *= comb'

    return tmpo

end


function _main_iten_tm()
    gz = 1. ; dt = 0.1
 
    times = []
    energies = []
    for mpolen = 4:20
            tmpo = matrix(periodic_tmpo_ising(gz, dt, mpolen))
            if mpolen > 10
                vals, vecs, infoKrylov = eigsolve(
                    tmpo,
                    rand(size(tmpo,1)),
                    16,
                    :LM;
                    ishermitian=false,
                    tol=1e-14,
                    krylovdim=30,
                    maxiter=100,
                    verbosity=0
                )
            else
                vals, vecs = eigen(tmpo, sortby=x->-abs(x))
            end

            push!(times, mpolen*dt)
            push!(energies, vecs)
    end
 
    jldsave("out_spectrum.jld2" ; times, energies)
 end






 function build_transverse_tm(nn=5, nbeta=0)
    
   wmi, wmi_im = build_Wi_itensors(1., 0.1)

    #ll, rr, pp, ppss = inds(wi)
    #wmi = Array(wi, ll, rr, pp, ppss)
    if nbeta == 0
        wx = wmi
    else
        wx = wmi_im
    end

    for jj = 2:nn
       @tullio wx2[l,l2,r,r2,p,ps2] := wx[l,r,p,xx] * wmi[l2, r2, xx,ps2] 
       wx = reshape(wx2,(2^jj,2^jj,2,2))
    end

    return wx
end


function build_Wi_manual(hz, dt)

    sigma_x = [0 1. ; 1 0]
    sigma_z = [1. 0 ; 0 -1]
    id = [1. 0 ; 0 1]


    cosg = cos(hz*dt*0.5)
    sing = sin(hz*dt*0.5)


    combz = (1 - 2*sing^2)*id + im*2*sing*cosg*sigma_z
    X = sigma_x

    #Wi = [vL, vR, p, p']

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
    X = sigma_x

    #Wi = [vL, vR, p, p']

    Wi_beta = zeros(ComplexF64, 2,2,2,2)

    Wi_beta[1,1,:,:] = cos(dt)*combz
    Wi_beta[1,2,:,:] = sqrt(im*sin(dt))*sqrt(cos(dt))*X
    Wi_beta[2,1,:,:] = sqrt(im*sin(dt))*sqrt(cos(dt))*X
    Wi_beta[2,2,:,:] = im*sin(dt)*combz


    return Wi, Wi_beta
end


function build_Wi_itensors(hz, dt)

    aa = siteinds("S=1/2", 10)
    hmpo = build_expH_ising_murg(aa, 1., hz, dt)
    wi = hmpo[5]
    
    ll = linkind(hmpo,4)
    rr = linkind(hmpo,5)
    pps = siteinds(hmpo,5)
    @show inds(wi)
    #ll, rr, pp, ppss = inds(wi)
    wmi = Array(wi, ll, rr, pps)
    #ll, rr, pp, ppss = inds(wi)
    #wmi = Array(wi, ll, rr, pp, ppss)

    hmpo = build_expH_ising_murg(aa, 1., hz, -im*dt)
    wi = hmpo[5]
    
    ll = linkind(hmpo,4)
    rr = linkind(hmpo,5)

    @show inds(wi)
    #ll, rr, pp, ppss = inds(wi)
    wmi_im = Array(wi, ll, rr, pps)
    #ll, rr, pp, ppss = inds(wi)
    #wmi = Array(wi, ll, rr, pp, ppss)

    return wmi, wmi_im
end




function build_transverse_tm_manual(nn=5, nbeta=0)
    
    Wi, Wi_beta = build_Wi_manual(1, 0.1)

    if nbeta == 0
        wx = Wi
    else
        wx = Wi_beta
    end

    for jj = 2:nn
       @tullio wx2[l,l2,r,r2,p,ps2] := wx[l,r,p,xx] * Wi[l2, r2, xx,ps2] 
       wx = reshape(wx2,(2^jj,2^jj,2,2))
    end

    return wx
end



function build_transverse_tm_sx_phys_manual(nn=5, tinsert=3, do_trace=true)
    @assert tinsert < nn

    sigma_x = [0 1. ; 1 0]


    wmi, wmi_beta = build_Wi_itensors(1., 0.1)
    wmi, wmi_beta = build_Wi_manual(1, 0.1)

    wx = wmi

    for jj = 2:nn
        @tullio wx2[l,l2,r,r2,p,ps2] := wx[l,r,p,xx] * wmi[l2, r2, xx,ps2] 
        if jj == tinsert
            @tullio wx3[l,l2,r,r2,p,ps2] := wx2[l,l2,r,r2,xx,ps2] * sigma_x[p,xx]
            wx2 = wx3
        end

        wx = reshape(wx2,(2^jj,2^jj,2,2))
    end


    if do_trace
    # trace top/bottom 
    @tullio wxTr[ll,rr] := wx[ll, rr, xx, xx]
    wx = wxTr
    end

    return wx
end



function build_sx_onesite(nn=5)
    
    mat = [0 1. ; 1 0]
    id = [1 0 ; 0 1]

    for jj = 2:nn
       @tullio temp[l,l2,r,r2] := mat[l,r] * id[l2, r2] 
       mat = reshape(temp,(2^jj,2^jj))
    end

    return mat
end


function build_op_onesite_alt(nn=5, n_insert=2, operator::Matrix)

    @assert n_insert < nn 
    @assert n_insert > 1
    
    #sigmax = [0 1. ; 1 0]
    id = [1 0 ; 0 1]

    mat = id
    for jj = 2:nn
        if jj == n_insert
            @tullio temp[l,l2,r,r2] := mat[l,r] * operator[l2, r2] 
        else
            @tullio temp[l,l2,r,r2] := mat[l,r] * id[l2, r2] 
        end
    
        mat = reshape(temp,(2^jj,2^jj))
    end

    return mat
end




function build_traced_tm(nn::Int, nbeta::Int)

    wx = build_transverse_tm(nn, nbeta)
    @show nbeta
    @show nn 
    @show size(wx)

    # trace top/bottom 
    @tullio wxTr[ll,rr] := wx[ll, rr, xx, xx]

    return wxTr
end

    


function build_transverse_tm_with_sx_virt(nn=5, n_insert=2)
    sigma_x = [0 1. ; 1 0]
    wxTr = build_traced_tm(nn)
    @show size(wxTr)
    wxTr = reshape(wxTr, ( 2^(n_insert-1), 2, 2^(nn-n_insert), 2^(n_insert-1), 2, 2^(nn-n_insert)))
    @tullio wx_sx[l1, l2, l3, r1, r2, r3] := wxTr[l1, l2, l3, r1, rx, r3] * sigma_x[rx, r2]

    return reshape(wx_sx,(2^nn, 2^nn))

end

function main_eigen_transverse(nmax::Int=5, nevals::Int=16, nbeta::Int=0)

    @assert nbeta < nmax
    valss = []
    gss = [] 
    corrs_allT = []
    for jj = 1:nmax

        wxTr= build_traced_tm(jj, nbeta)

        @show Base.summarysize(wxTr)/1024/1024/1024

        
        if size(wxTr)[1] > 2100
            @info "matrix is large, using eigsolve"
            vals, vecs = eigsolve(wxTr, nevals, :LM, ComplexF64)
        else
            vals,vecs = eigen(wxTr, sortby= x->-abs(x)) 
        end
        
        #chop to first N eigvals
        if length(vals) > nevals
            vals = vals[1:nevals]

        end
        
        push!(valss, vals)

        gs = vecs[:,1]

        push!(gss, gs)

        if jj > 2 # build correls only for longer T 
            one_sx = build_sx_onesite_alt(jj, 2)

            temp = one_sx * gs
            corrs = []
            for dist = 1:40
                temp = wxTr * temp
                resu = transpose(gs) * one_sx * temp
                push!(corrs, resu)
            end

            # Alternative 

            # one_sx = build_sx_onesite_alt(jj, jj-1)

            # temp = one_sx * gs
            # corrs_alt = []
            # for dist = 1:100
            #     temp = wxTr * temp
            #     resu = transpose(gs) * one_sx * temp
            #     push!(corrs_alt, resu)
            # end


            push!(corrs_allT, corrs)
        end

        jldsave("spectrum_tm_wcorrs_nbeta_$(nbeta).jld2" ; valss, gss, corrs_allT)
    end

    return valss, gss, corrs_allT

end


function main_onepoint_time(nmax::Int=5, nevals::Int=16, nbeta::Int=0)

    @assert nbeta < nmax
    valss = []
    gss = [] 
    corrs_allT_1 = []
    corrs_allT_2 = []

    for jj = 1:nmax

        wxTr= build_traced_tm(jj, nbeta)

        @show Base.summarysize(wxTr)/1024/1024/1024
        
        if size(wxTr)[1] > 2100
            @info "matrix is large, using eigsolve"
            vals, vecs = eigsolve(wxTr, nevals, :LM, ComplexF64)
            @show size(vals)
            @show size(vecs)
            @show size(vecs[1])

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

        if jj > 2 # build correls only for longer T 

            corrs = []
            for pos_x = 1:jj-1
                one_sx = build_transverse_tm_sx_phys(jj, pos_x)
                resu = transpose(gs) * one_sx * gs
                push!(corrs, resu)
            end

            push!(corrs_allT_1, corrs)

            corrs = []
            for pos_x = 1:jj-1
                one_sx = build_transverse_tm_sx_phys(jj, pos_x)
                resu = transpose(gs2) * one_sx * gs2
                push!(corrs, resu)
            end

            push!(corrs_allT_2, corrs)

        end

        jldsave("spectrum_tm_onepoint_timecorrel_nbeta_$(nbeta).jld2" ; valss, gss, corrs_allT_1, corrs_allT_2)
    end

    return valss, gss, corrs_allT_1, corrs_allT_2

end


aa,bb,cc = main_eigen_transverse(10, 16, 0)

dd,ee,ff = main_eigen_transverse(10, 16, 1)

#gg, hh, ii, jj = main_onepoint_time(12,16,0)