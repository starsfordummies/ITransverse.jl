@info "Running all examples"

filenames = filter(readdir(@__DIR__)) do f
    startswith("main_")(f) && endswith(".jl")(f)
end

for filename in filenames
    println("Running $(@__DIR__)/$filename")
    @time include(filename)
end
