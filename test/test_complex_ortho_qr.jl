using Test
using ITensors, ITensorMPS, ITransverse
using LinearAlgebra

const QR_TOL = 1e-10

@testset "complex_orthogonal_qr" begin

    @testset "square 3×3" begin
        A = [1.0+2im  3-1im  2+0im;
             0+1im    2+3im  1-2im;
             3-2im    1+0im  4+1im]
        Q, R = complex_orthogonal_qr(A)
        @test norm(Q * R - A)              < QR_TOL
        @test norm(transpose(Q) * Q - I)   < QR_TOL
    end

    @testset "tall 25×30" begin
        B = randn(ComplexF64, 25, 30)
        Q, R = complex_orthogonal_qr(B)
        @test norm(Q * R - B)              < QR_TOL
        @test norm(transpose(Q) * Q - I)   < QR_TOL
    end

    @testset "complex symmetric 3×3" begin
        C = [1+1im  2.0-1im  0+3im;
             2-1im  3+0im    1+1im;
             0+3im  1+1im    2-2im]
        Q, R = complex_orthogonal_qr(C)
        @test norm(Q * R - C)              < QR_TOL
        @test norm(transpose(Q) * Q - I)   < QR_TOL
    end

end

