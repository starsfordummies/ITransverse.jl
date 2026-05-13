using Documenter
using ITransverse

makedocs(;
    modules  = [ITransverse],
    sitename = "ITransverse.jl",
    authors  = "Stefano Carignano",
    format   = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://starsfordummies.github.io/ITransverse.jl",
    ),
    pages = [
        "Home"       => "index.md",
        "Algorithms" => "algorithms.md",
        "API"        => [
            "Chain Models"         => "api/chain_models.md",
            "tMPO Construction"    => "api/tmpo.md",
            "Truncation & Sweeps"  => "api/truncation.md",
            "Power Method"         => "api/power_method.md",
            "Light Cone"           => "api/lightcone.md",
            "Entropies"            => "api/entropies.md",
            "Contractions"         => "api/contractions.md",
            "TEBD"                 => "api/tebd.md",
            "ITensor Utilities"    => "api/itutils.md",
        ],
    ],
    warnonly = true,   # don't fail the build on missing docstrings
)

deploydocs(;
    repo   = "github.com/starsfordummies/ITransverse.jl",
    branch = "gh-pages",
    push_preview = true,
)
