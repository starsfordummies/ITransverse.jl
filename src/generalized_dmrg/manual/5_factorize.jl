
""" Factorize so the RIGHT tensor is orthogonal/symmetric 
   Bijac = Miab Ojbc"""
function factorize_right_ortho(B::ITensor, rinds; extratags="")
    B2 = B * prime(B, rinds)
    B2mat = matrix(B2 * combiner(rinds) * combiner(rinds'))
    oo, dd = basic_sym_eig(B2mat)
    #@show size(oo), size(dd)
    newlink = Index(size(dd,1), tags="Link")
    newlink = addtags(newlink, extratags) 

    @show size(oo)
    right_tensor = ITensor(oo, rinds, newlink)
    left_tensor = B * right_tensor

    if norm(left_tensor * right_tensor - B)/norm(B) > 1e-6
        @warn "Large truncation in factorization? $(norm(left_tensor * right_tensor - B)/norm(B))"
        sleep(4)
    end

    return left_tensor, right_tensor 
end

function factorize_right_ortho_alt(B::ITensor, rinds; extratags="")
    linds = uniqueinds(B, rinds)
    B2 = B * prime(B, linds)
    B2mat = matrix(B2 * combiner(linds) * combiner(linds'))
    oo, dd = basic_sym_eig(B2mat, cutoff=1e-12)

    sqd = dd.^0.5
    isqd = sqd.^(-1)
    @assert sqd .* isqd ≈ ones(size(sqd))

    #@show size(oo), size(dd)
    newlink = Index(size(dd,1), tags="Link")
    newlink = addtags(newlink, extratags) 
    right_tensor =  ITensor(oo * Diagonal(isqd), linds, newlink) * B
    left_tensor = ITensor(oo * Diagonal(sqd), linds, newlink)

    if norm(left_tensor * right_tensor - B)/norm(B) > 1e-6
        @warn "Large truncation in factorization? $(norm(left_tensor * right_tensor - B)/norm(B))"
        sleep(4)
    end

    return left_tensor, right_tensor 
end


""" Factorize so the LEFT tensor is orthogonal/symmetric 
    Bijac = Oiab Mjbc"""
function factorize_left_ortho(B::ITensor, linds; extratags="")
    B2 = B * prime(B, linds)
    B2mat = matrix(B2 * combiner(linds) * combiner(linds'))
    oo, dd = basic_sym_eig(B2mat)
    #newlink = Index(size(dd,1), tags="Link")
    newlink = Index(size(dd,1), tags="Link")
    newlink = addtags(newlink, extratags) 

    @debug @show size(B2mat), size(oo), size(dd)
    #@show linds, newlink

    #@show size(oo)

    left_tensor = ITensor(oo, linds, newlink)
    right_tensor = B * left_tensor


    if norm(left_tensor * right_tensor - B)/norm(B) > 1e-6
        @warn "Large truncation in factorization? $(norm(left_tensor * right_tensor - B)/norm(B))"
        sleep(4)
    end

    return left_tensor, right_tensor 
end


""" Factorize so the LEFT tensor is orthogonal/symmetric 
    Bijac = Oiab Mjbc"""
function factorize_left_ortho_alt(B::ITensor, linds; extratags="")
    rinds = uniqueinds(B, linds)
    B2 = B * prime(B, rinds)
    B2mat = matrix(B2 * combiner(rinds) * combiner(rinds'))
    oo, dd = basic_sym_eig(B2mat)

    sqd = dd.^0.5
    isqd = sqd.^(-1)
    @assert sqd .* isqd ≈ ones(size(sqd))

    #newlink = Index(size(dd,1), tags="Link")
    newlink = Index(size(dd,1), tags="Link")
    newlink = addtags(newlink, extratags) 

    @debug @show size(B2mat), size(oo), size(dd)
    #@show linds, newlink

    left_tensor = B * ITensor(oo * Diagonal(isqd), rinds, newlink)
    right_tensor = ITensor(oo * Diagonal(sqd), rinds, newlink)


    if norm(left_tensor * right_tensor - B)/norm(B) > 1e-6
        @warn "Large truncation in factorization? $(norm(left_tensor * right_tensor - B)/norm(B))"
        sleep(4)
    end

    return left_tensor, right_tensor 
end




function factorize_ro(B::ITensor, rinds; extratags="", method=1)
    if method == 1 
        factorize_right_ortho(B::ITensor, rinds; extratags)
    elseif method == 2 
        factorize_right_ortho_alt(B::ITensor, rinds; extratags)
    else
        @error "Choose a method 1/2"
    end
end


function factorize_lo(B::ITensor, rinds; extratags="", method=1)
    if method == 1 
        factorize_left_ortho(B::ITensor, rinds; extratags)
    elseif method == 2 
        factorize_left_ortho_alt(B::ITensor, rinds; extratags)
    else
        @error "Choose a method 1/2"
    end
end
