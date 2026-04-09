using ITensors, ITensorMPS
using ITransverse
using Test

@testset "Building MPO blocks in different ways" begin
  tp = ising_tp()

  tp = tMPOParams(tp; dbeta=0.1im)
  eH = build_Ut(tp)

  b = FwtMPOBlocks(tp)
  b = FwtMPOBlocks(tp, init_state=[0,1])
  b = FwtMPOBlocks(tp, init_state=ITensor(ComplexF64[1,0], Index(2)))

  @test storage(b.Wc) != storage(b.Wc_im)

  b = FwtMPOBlocks(eH, init_state= [0,1])
  b = FwtMPOBlocks(eH, init_state=ITensor(ComplexF64[1,0], Index(2)))

  @test storage(b.Wc) ≈ storage(b.Wc_im)

  b = FoldtMPOBlocks(tp)
  b = FoldtMPOBlocks(tp, init_state=[0,1])
  b = FoldtMPOBlocks(tp, init_state=[im,1])
  b = FoldtMPOBlocks(tp, init_state=ITensor(ComplexF64[1,0], Index(2)))

  @test storage(b.WWc) != storage(b.WWc_im)

  b = FoldtMPOBlocks(eH, init_state= [0,1])
  b = FoldtMPOBlocks(eH, init_state=ITensor(ComplexF64[1,0], Index(2)))

  @test storage(b.WWc) ≈ storage(b.WWc_im)

end