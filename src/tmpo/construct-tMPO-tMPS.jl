
"""
    construct_tMPS_tMPO(ψ_i::MPS, Ut::Vector{MPO}, ϕ_f::MPS)
```
|ϕ_f⟩    o———o———o
         |   |   |
Ut[end]  U———U———U
         |   |   |
Ut[…]          ⋮
         |   |   |
Ut[2]    U———U———U
         |   |   |  time
Ut[1]    U———U———U    ↑
         |   |   |    |
⟨ψ_i|    o———o———o    |
                      |
        ⟨L|  T  |R⟩
```
-Given a set of MPS-[MPO(1),MPO(2)...MPO(Nt)]-MPS defining a TN with **3** spatial sites,
constructs two boundary tMPS (⟨L| and |R⟩) the tMPO TR with Nt temporal sites
for further use in the power method.
- Optionally if `return_swapped_T` also returns TL = swapprime(TR, 0, 1)`.
- `psi_i` and `psi_f` are assumed to be a valid (space-like) MPS
- each element of the vector `Ut` is assumed to be a valid MPO whereas the links match, respectively,
- The (physical) sites need not be correctly primed to link up in the time direction.
- The final MPS **will be automatically daggered**! If you do not want it to be daggered,
  use the flag `dagger_final=false`.
"""
function construct_tMPS_tMPO(psi_i::MPS, in_Uts::Vector{MPO}, psi_f::MPS;
  return_swapped_T::Bool=false, dagger_psif::Bool=true, new_siteinds=nothing
)

  @assert length(psi_i) == 3


  # Fix inds 
  psi_i = replace_siteinds(psi_i, firstsiteinds(in_Uts[1], plev=0))
  psi_f = replace_siteinds(psi_f, firstsiteinds(in_Uts[end], plev=0))

  siteinds_Ut = [firstsiteinds(Ut) for Ut in in_Uts]
  if length(unique(siteinds_Ut)) > 1
    for (ii, Ut) in enumerate(in_Uts)
      in_Uts[ii] = replace_siteinds(Ut, siteinds_Ut[1])
    end
  end


  Nrows = length(in_Uts)

  Uts = sim.(linkinds, in_Uts)


  #Incorporate initial and final state in MPOs. 
  #First remove trivial links from produc states for QN mental sanity
  ITransverse.ITenUtils.delete_link_from_prodMPS!(psi_i)
  ITransverse.ITenUtils.delete_link_from_prodMPS!(psi_f)

  if dagger_psif
    psi_f = dag(psi_f)
  end

  Uts[1] = applyn(Uts[1], psi_i)
  Uts[end] = applyns(Uts[end], psi_f)

  #TODO check: do we still need this ? 
  for ii = 3:Nrows
    Uts[ii] = prime(siteinds, Uts[ii], ii - 2)
  end

  Lcol_data = [Uts[ii][1] for ii = (1:Nrows)]
  Ccol_data = [Uts[ii][2] for ii = (1:Nrows)]
  Rcol_data = [Uts[ii][end] for ii = (1:Nrows)]

  psiL = MPS(Lcol_data)
  Tc = MPO(Ccol_data)
  psiR = MPS(Rcol_data)

  # Replace indices, starting from the right MPS 

  ssR = siteinds(psiR)
  new_siteinds = something(new_siteinds, [noprime(sim(ssR[ii], tags="Site,nt=$(ii)")) for ii in eachindex(ssR)])
  new_siteindsP = dag(new_siteinds')

  replace_siteinds!(psiR, new_siteinds)
 
  stc = allsiteinds(Tc)
  ssL = uniqueinds(stc, ssR)
  @assert length(uniqueinds(stc,ssR)) == length(ssR) "Something likely wrong in index labelling"
  for ii in eachindex(ssR)
      Tc[ii] = replaceinds(Tc[ii], ssR[ii] => new_siteinds[ii], ssL[ii] => new_siteindsP[ii])
  end

  replace_siteinds!(psiL, new_siteinds)


  for col = [psiL, Tc, psiR]
    ll = linkinds(col)
    for ii in eachindex(ll)
      lln = noprime(sim(ll[ii], tags="Link,lt=$(ii)"))
      col[ii] = replaceind(col[ii], ll[ii] => lln)
      col[ii+1] = replaceind(col[ii+1], ll[ii] => dag(lln))
    end
  end

  if return_swapped_T
    return psiL, swapprime(Tc, 0, 1), Tc, psiR
  else
    return psiL, Tc, psiR
  end
end



""" Similar to `construct_tMPS_tMPO`, but takes input MPS/MPO longer than 3 and builts the appropriate sets of columns.
returns left_mps, [list of columns], right mps. In our convention rows are labelled from bottom to top.
```
<ϕ_f|    o———o———o———o———o
         |   |   |   |   |   
Ut[end]  U———U———U———U———U
         |   |   |   |   |   
Ut[…]    ⋮       ⋮       ⋮
         |   |   |   |   |   
Ut[2]    U———U———U———U———U
         |   |   |   |   |     time
Ut[1]    U———U———U———U———U    ↑
         |   |   |   |   |    |
|ψ_i>    o———o———o———o———o    |
                      
        ⟨L|  T1 ..  TNx |R⟩
```
"""
function construct_tMPS_tMPO_finite(psi_i::MPS, mpo_rows::Vector{MPO}, psi_f::MPS; dagger_psif::Bool=true)

  all_rows = (psi_i, mpo_rows..., psi_f)
  ld_matrix = stack(linkdims.(all_rows))  # (Nbonds x Nrows) matrix
  @assert allequal(eachrow(ld_matrix)) "Link dims not uniform across rows: $(ld_matrix)"

  Nrows = length(mpo_rows)
  Ncols = length(psi_i)

  @info "TN made by Nt=$(Nrows) X L=$(Ncols)" 

  # ensure siteinds match before rotation 
  psi_i = replace_siteinds(psi_i, firstsiteinds(mpo_rows[1], plev=0))
  psi_f = replace_siteinds(psi_f, firstsiteinds(mpo_rows[end], plev=0)) # we applys() afterwards so no worries


  # Match siteinds() of all rows 
  siteinds_Ut = [firstsiteinds(Ut, plev=0) for Ut in mpo_rows]
  work_rows = if length(unique(siteinds_Ut)) > 1
    [sim(linkinds, replace_siteinds(Ut, siteinds_Ut[1])) for Ut in mpo_rows]
    # for (ii, Ut) in enumerate(mpo_rows)
    #   in_Uts[ii] = replace_siteinds(Ut, siteinds_Ut[1])
    # end
  else
    sim.(linkinds, mpo_rows)
  end

  # # To avoid repeated inds in case we're feeding copies of the same MPO as rows
  # Uts = sim.(linkinds, in_Uts)

  #Incorporate initial and final state in MPOs. 
  #First remove trivial links from produc states for QN mental sanity
  # ITransverse.ITenUtils.delete_link_from_prodMPS!(psi_i)
  # ITransverse.ITenUtils.delete_link_from_prodMPS!(psi_f)

  if dagger_psif
    psi_f = dag(psi_f)
  end

  # Incorporate initial and final states in the first/last row, which become MPS 
  work_rows[1] = applyn(work_rows[1], ITransverse.ITenUtils.delete_link_from_prodMPS(psi_i))
  work_rows[end] = applyns(work_rows[end], ITransverse.ITenUtils.delete_link_from_prodMPS!(psi_f))

  for ii = 3:Nrows
    work_rows[ii] = prime(siteinds, work_rows[ii], ii - 2)
  end

  # Fill the columns starting from the left 
  Lcol_data = [work_rows[ii][1] for ii = (1:Nrows)]

  # Ccol_data = []
  # for jj = 2:Ncols-1
  #   push!(Ccol_data, [mpo_rows[ii][jj] for ii = (1:Nrows)])
  # end
  Ccol_data = [[work_rows[ii][jj] for ii in 1:Nrows] for jj in 2:Ncols-1]

  Rcol_data = [work_rows[ii][end] for ii = (1:Nrows)]

  psiL = MPS(Lcol_data)
  Tcs = [MPO(cc) for cc in Ccol_data]
  psiR = MPS(Rcol_data)

  @info "Building 2 MPS + $(length(Tcs)) MPO columns of length $(length(psiR))"

  # Adapting indices to transverse structure, starting from the right 
  ssR = siteinds(psiR)
  ssNew = [noprime(sim(ssR[ii], tags="Site,nt=$(ii)")) for ii in eachindex(ssR)]
  ssNewP = dag(ssNew')

  replace_siteinds!(psiR, ssNew)

  for Tc in reverse(Tcs)
      stc = siteinds(Tc)
      #ssL = uniqueinds(stc, ssR)
      ssL = [uniqueind(s,r) for (s,r)in zip(stc,ssR)]
      @assert length(ssL) == length(ssR) "Something likely wrong in index labelling"
      for ii in eachindex(ssR)
          Tc[ii] = replaceinds(Tc[ii], ssR[ii] => ssNew[ii], ssL[ii] => ssNewP[ii])
      end
      ssR = ssL

  end

      replace_siteinds!(psiL, ssNew)


    for col in (psiL, Tcs..., psiR)
      ll = linkinds(col)
      for ii in eachindex(ll)
        lln = noprime(sim(ll[ii], tags="Link,lt=$(ii)"))
        col[ii] = replaceind(col[ii], ll[ii] => lln)
        col[ii+1] = replaceind(col[ii+1], ll[ii] => dag(lln))
      end
    end

      return psiL, Tcs, psiR
  end
