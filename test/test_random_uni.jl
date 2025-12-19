using ITensors, ITransverse
using ITransverse.ITenUtils: random_uni
using Test
c1 = combiner(Index(10), Index(8))

rrq = random_uni(combinedind(c1))
rrq2 = rrq * dag(c1)
@test isid(rrq2 * prime(dag(rrq2),"qr"))


# s1 = Index(2, "Site")

# l1 = Index(10, "link1")
# l2 = Index(12, "link1")

# l3 = Index(14, "link2")
# l3 = Index(16, "link2")

# c1 = combiner(s1,l3)
# c2 = combiner(s1,l4)

# m1 = random_itensor