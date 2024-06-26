using ITensors

sites =  siteinds("S=1/2", 10; conserve_qns = false)
#addtags(time_sites, "time"

z5 = op("Sz",sites,5)
z6 = op("Sz",sites,6)

z56 = z5*z6

exp_z56 = exp(z56)

iii = inds(exp_z56)

uu,ss,vdd = svd(exp_z56, (iii[1], iii[2]) )

println(ss)

uu,ss,vdd = svd(exp_z56, (iii[1], iii[3]) )

println(ss)


# Now with sigmas 

sites_potts =  siteinds("S=1", 10; conserve_qns = false)


s5 = op("Σ",sites_potts,5)
s5d = op("Σdag",sites_potts,5)

s6 = op("Σ",sites_potts,6)
s6d = op("Σdag",sites_potts,6)

ss56d = s5 * s6d 
ss5d6 = s5d * s6 

e1 = exp(ss56d)
e2 =  exp(ss56d + ss5d6)

jjj=inds(e1)

uu,ss,vdd = svd(e1, [jjj[1],jjj[3]])

@show(diag(ss))


uu,ss,vdd = svd(e2, [jjj[1],jjj[3]])

@show(diag(ss))
#e2

dd, vv = eigen(e2, [jjj[1],jjj[3]], [jjj[2],jjj[4]], ishermitian=true, cutoff=1e-12) 

@show(dd)
# sortby= x->-abs(x),
#e1