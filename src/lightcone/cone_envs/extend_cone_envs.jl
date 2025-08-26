""" Extend the cone: replace the middle column with an extended one by vwidth, adds
    - two columns (neighboring the center) in Columns
    - two left envs
    - two right envs 
"""
function extend_cone!(b::FoldtMPOBlocks, cc::Columns, left_envs, right_envs; fold_op, vwidth::Int=1)

    NN = length(cc)

    half = div(NN,2)+1

    ts = firstsiteinds(cc[half])
    for jj = 1:vwidth
        push!(ts, Index(dim(ts[end]), tags="Site,n=$(length(ts)+1),time"))
    end

    newL = folded_tMPO_ext(b, ts; LR=:left, n_ext=vwidth)
    newC = folded_tMPO(b, ts; fold_op = fold_op)
    newR = folded_tMPO_ext(b, ts; LR=:right, n_ext=vwidth)

    splice!(cc.cols, half, [newL, newC, newR])
   
    #Extend environments 

    ll = left_envs[half-1]
    llO = applyns(newL, ll, truncate=true, maxdim=maxlinkdim(ll)+4)
    llO = orthogonalize(llO, length(llO))
  
    llOO = applyns(newC, llO, truncate=true, maxdim=maxlinkdim(ll)+4)
    llOO = orthogonalize(llOO, length(llOO))

    llOOO = applyns(newR, llOO, truncate=true, maxdim=maxlinkdim(ll)+4)
    llOOO = orthogonalize(llOOO, length(llOOO))

    splice!(left_envs.norms, half, [norm(llO), norm(llOO), norm(llOOO)])
    splice!(left_envs.envs, half, [normalize(llO), normalize(llOO), normalize(llOOO)])


    rr = right_envs[half]
    Orr = applyn(newR, rr, truncate=true, maxdim=maxlinkdim(rr)+4)
    Orr = orthogonalize(Orr, length(Orr))
  
    OOrr = applyn(newC, Orr, truncate=true, maxdim=maxlinkdim(rr)+4)
    OOrr = orthogonalize(OOrr, length(OOrr))

    OOOrr = applyn(newL, OOrr, truncate=true, maxdim=maxlinkdim(rr)+4)
    OOOrr = orthogonalize(OOOrr, length(OOOrr))

    splice!(right_envs.norms, half-1, [norm(OOOrr), norm(OOrr), norm(Orr)])
    splice!(right_envs.envs, half-1, [normalize(OOOrr), normalize(OOrr), normalize(Orr)])

end
