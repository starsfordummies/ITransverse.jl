if Base.Sys.islinux()
    println("On linux - setting MKL and 8 threads")
    using MKL
    using LinearAlgebra

    BLAS.set_num_threads(8)  # 8 threads seems to be a sweet spot
    #unicodeplots()
end
