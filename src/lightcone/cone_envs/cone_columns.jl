
""" Builds the columns in a light cone form. Input is Nt timesteps and velocity for the light cone `vwidth`.
We build 
  - a central column where the operator is, of length Nt
  - two central ±1 columns of length Nt
  - as we move away from the center, each following column is wvidth shorter than the prev one
"""
function build_cols_cone(b::FoldtMPOBlocks, Nt::Int; fold_op, vwidth::Int=1)

    @info "Building cone columns"

    ts = addtags(siteinds(4, Nt; conserve_qns=false), "time")

    # Fix the column lengths 
    edge_length = max(vwidth, Nt - div(Nt, vwidth)*vwidth)         # ascending range (inclusive)
    right_lengths = Nt:-vwidth:edge_length    # descending range 
    left_lengths = reverse(right_lengths)

    clens = vcat(collect(left_lengths), Nt, collect(right_lengths))

    @info clens

    ll = folded_tMPS(b, ts[1:clens[1]]; LR=:left)

    Ncols = length(clens)
    mid = length(left_lengths)

    cols = fill(MPO(), Ncols)
    
    for jj = 2:mid
        cols[jj] = folded_tMPO_ext(b, ts[1:clens[jj]];  LR=:left, n_ext=vwidth)
    end
    cols[mid+1] = folded_tMPO(b, ts[1:clens[mid+1]]; fold_op)

    for jj = mid+2:Ncols-1
        cols[jj] = folded_tMPO_ext(b, ts[1:clens[jj]]; LR=:right, n_ext=vwidth)
    end

    rr = folded_tMPS(b, ts[1:clens[1]]; LR=:right)

    return Columns(ll, cols, rr)

end
