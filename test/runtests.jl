using Test
using Random

# Many test files draw random states and compare against tolerances. Seed once here so the
# whole suite is reproducible; individual files seed themselves as well, so they are also
# reproducible when run on their own.
Random.seed!(20250803)

@info "Running all tests"

@testset "$(@__DIR__)" begin
    filenames = filter(readdir(@__DIR__)) do f
      startswith("test_")(f) && endswith(".jl")(f)
    end
    @testset "Test $(@__DIR__)/$filename" for filename in filenames
      println("Running $(@__DIR__)/$filename")
      @time include(filename)
    end
  end