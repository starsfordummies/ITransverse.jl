using ITensors 

s = siteinds("S=1/2", 2)

opX1= op("X", s[1])
opX2= op("X", s[2])
opZ1= op("Z", s[1])
opZ2= op("Z", s[2])

eX1 = exp(opX1)

X12 = opX1 * opX2 
inds(X12)
X12tog = X12*combiner(ind(X12,1), ind(X12,3))*combiner(ind(X12,2), ind(X12,4))

eZ1h = exp(opZ1/2)
eZ2h = exp(opZ2/2)

i1 = (s[1])
ZXXZ = (X12 * eZ1h' * eZ1h)
ZXXZa = mapprime(ZXXZ, 2=>1)
inds(ZXXZ)
XZXXZ = eX1 * ZXXZ

#fix 
ZXXZX =  ZXXZa * eX1'