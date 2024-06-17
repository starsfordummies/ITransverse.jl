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

