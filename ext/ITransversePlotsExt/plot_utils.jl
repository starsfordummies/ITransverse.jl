# TODO put as extension


function ITransverse.plot_matrix(a::Matrix)
    heatmap(1:size(a,1),
           1:size(a,2), abs.(a),
           c=cgrad([:blue, :white,:red, :yellow]),
           xlabel="i", ylabel="j",
           title="matrix")
end

function ITransverse.plot_matrix(a::ITensor)
    @assert order(a) == 2
    plot_matrix(matrix(a))
end

function ITransverse.plotr(a::Array)
    plot(real(a), label="Re($(label))")
end

function ITransverse.ploti(a::Array)
    plot(imag(a), label="Im($(label))")
end

function ITransverse.plotri(a::Array; label::String="")
    p1 = plot(real(a), label="Re($(label))")
    p2 = plot(imag(a), label="Im($(label))")
    plot(p1,p2)
end