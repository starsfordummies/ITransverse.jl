""" What if the initial state is not a product state ? 
 We can work by defining an open tMPO, and add one tensor at the bottom which,
 instead of a physical temporal leg, has a virtual spatial leg,

`
 |
-o-
 |
-o-
 |
-o-
 |
=□=
 `

 """
 
using ITensors, ITransverse

s = siteinds("S=3/2", 10)
tp = ising_tp()

top, open_up, open_down = build_folded_open_tMPO(tp,s)

siteinds(top)
linkinds(top)
space_link = Index(6,"svirt")
rho0 = ITensor(space_link ,open_down, space_link')

insert!(top.data, length(top)+1, rho0)

trivial_link = Index(1,"trivial")
op = ITensor([1,0,0,1], trivial_link, open_up, trivial_link')

insert!(top.data, 1, op)

siteinds(top)
linkinds(top)

s_ext = deepcopy(s)
insert!(s_ext, 1, trivial_link)
push!(s_ext, space_link)

psi0 = random_mps(s_ext)

opsi = applyn(top, psi0)  # error, mismatched lengths! 

insert!(psi0.data, 1, ITensor(1.))

apply(top, psi0) 
siteinds(top)
siteinds(psi0)
top2, open_up2, open_down2 = build_folded_open_tMPO(tp,s)
insert!(top2.data, length(top2)+1, rho0)
insert!(top2.data, 1, op)

siteinds(top2)
linkinds(top2)

applyn(top2, top)


test1 = MPS()
for jj = 1:10 
    p0 = ITensor([1/sqrt(2), 1/sqrt(2)], Index(2,"$(jj)"))
    push!(test1.data, p0)
end


test2 = MPS()
for jj = 1:10 
    inext = Index(1,"trivial $(jj+1)")
    if jj == 1
        p0 = ITensor([1/sqrt(2), 1/sqrt(2)], Index(2,"$(jj)"), inext)
    elseif jj == 10
        p0 = ITensor([1/sqrt(2), 1/sqrt(2)], iprev, Index(2,"$(jj)"))
    else
        p0 = ITensor([1/sqrt(2), 1/sqrt(2)], iprev, Index(2,"$(jj)"), inext)
    end

    push!(test2.data, p0)
    iprev = inext
end

replace_siteinds!(test2,siteinds(test1))
inner(test1, test2)