using ITensors, ITensorMPS
using ITransverse
using Test

using ITransverse: up_state, build_expH, phys_ind

@testset "Building MPO blocks in different ways" begin
  tp = ising_tp()
  eH = build_expH(tp)

  b = FwtMPOBlocks(tp)
  b = FwtMPOBlocks(tp, init_state=[0,1])
  b = FwtMPOBlocks(tp, init_state=[0,1], build_imag=false)
  b = FwtMPOBlocks(tp, init_state=ITensor(ComplexF64[1,0], Index(2)))
  b = FwtMPOBlocks(eH, init_state= [0,1])
  b = FwtMPOBlocks(eH, init_state=ITensor(ComplexF64[1,0], Index(2)))


  b = FoldtMPOBlocks(tp)
  b = FoldtMPOBlocks(tp, init_state=[0,1])
  b = FoldtMPOBlocks(tp, init_state=[im,1], build_imag=false)
  b = FoldtMPOBlocks(tp, init_state=ITensor(ComplexF64[1,0], Index(2)))
  b = FoldtMPOBlocks(eH, init_state= [0,1])
  b = FoldtMPOBlocks(eH, init_state=ITensor(ComplexF64[1,0], Index(2)))

  @test true

end