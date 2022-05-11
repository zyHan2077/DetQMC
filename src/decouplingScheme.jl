function decouplingCoefficients_CjWu(U::Float64, Δτ::Float64)
    a = exp(- U * Δτ / 2.0)
    a2 = exp(- U * Δτ)
    d = sqrt( 8 + ( a2*(3 + a2)^2 ) )
    g = a * (3 + a2) / d
    e1 = (a * (1 + a2)^2) / 4.0
    e2 = (a2 - 1) * d / 4
    return ( (1 - g, 1 + g), (acos(e1 + e2), acos(e1 - e2)) )
end

function decouplingCoefficients_Assaad(U::Float64, Δτ::Float64)
    a = sqrt(Δτ * U)
    return (1.0 + sqrt(6.0) / 3.0, 1.0 - sqrt(6.0) / 3.0), a .* (sqrt(3.0 - sqrt(6.0)), sqrt(3.0 + sqrt(6.0)))
end