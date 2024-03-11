using QuadGK

# H = XX + Z 

# Start considering a quench from g0=0 to g

function my_integrand_from_zero(k, g)

    ck = cos(k)
    sqq = sqrt(1. + g*g - 2*g*ck)

    one = 2*g*sin(k)/sqq
    two = log(abs( (1-g*ck)/sqq ))

    return one*two/pi
end

function integrate_fk_ofg_from_zero(g)
     return quadgk(k -> my_integrand_from_zero(k,g), 0., pi)
end


function expval_sx_oft_theory_from_zero(t, g, J=1., phi=0.)
    #phi is the offset for the cos(), comes into play for g > 1
    if g < 1
        C =  0.5*(1. + sqrt(1-g*g))
        fac = 1.
    else # g > 1
        C = 1.
        fac = sqrt(1. + cos(4*t*J*sqrt(g*g-1) + phi))
    end

    exp_t = exp(J*t*integrate_fk_ofg_from_zero(g)[1])

    return sqrt(C)*fac*exp_t
end

function build_evs(g)
    evs = []
    for tt in 0.0:0.1:8
        push!(evs, expval_sx_oft_theory_from_zero(tt, g))
    end

    return evs 
end 

ev_x_analytics = build_evs(0.8)