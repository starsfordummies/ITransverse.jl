
function get_Wl(b)
    Wtensor  = ITensor() 
    if isa(b, FwtMPOBlocks)
        Wtensor = b.Wl
    elseif isa(b, FoldtMPOBlocks)
        Wtensor = b.WWl
    else
        error("input is a $(typeof(b)), don't know to extract a Wl from it")
    end
    return Wtensor
end

function get_Wc(b)
    Wtensor  = ITensor()
    if isa(b, FwtMPOBlocks)
        Wtensor = b.Wc 
    elseif isa(b, FoldtMPOBlocks)
        Wtensor = b.WWc
    else
        error("input is a $(typeof(b)), don't know to extract a Wc from it")
    end
    return Wtensor
end

function get_Wr(b)
    Wtensor  = ITensor()
    if isa(b, FwtMPOBlocks)
        Wtensor = b.Wr 
    elseif isa(b, FoldtMPOBlocks)
        Wtensor = b.WWr
    else
        error("input is a $(typeof(b)), don't know to extract a Wr from it")
    end
    return Wtensor
end