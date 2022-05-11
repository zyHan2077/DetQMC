# module DQMC

using LinearAlgebra, Printf
using StableDQMC

# export DQMCScheme, DQMCWalker, initializeWalker, AuxFieldUpdaterParams, walkerMarkovMove!

include("decouplingScheme.jl")

const auxDtype = Int8

const component4AuxField = NTuple{4, auxDtype}((-2, -1, 1, 2))

# spin-orbital coupling is NOT considered
struct DQMCScheme{T<:Number}
    Nsite::Int
    Nflavor::Int
    M::Int
    DeltaTau::Float64
    Kinectic::Matrix{T} # c^\dagger_i K_{ij} c_j
    KinecticDeltaTau::Matrix{T}
    invKinecticDeltaTau::Matrix{T}
    function DQMCScheme{T}(Nsite, Nflavor, M, DeltaTau, Kinectic) where T <: Number
        KinecticDeltaTau = exp( - DeltaTau * Kinectic)
        new(Nsite, Nflavor, M, DeltaTau, Kinectic, KinecticDeltaTau, exp( + DeltaTau * Kinectic))
    end
end

mutable struct DQMCWalker{T<:Number}
    HSFieldConfig::Array{auxDtype, 2} # $s_i(l)$, column major therefore $N_sites × M$
    BMats::Array{Array{T, 2}, 1}
    glTemp::Matrix{T}
    UDTsRight::Array{UDT, 1} # (BM B2 ⋯ B1), ⋯ BM
    UDTsLeft::Array{UDT, 1} # B1, B2B1, ... (BM ⋯ B2 B1)
    counter::Int
    refreshInterval::Int
    individualWeight::T
end

#
# \sum_s gamma(s) e^{Im * eta(s) * n}
#
struct AuxFieldUpdaterParams{T<:Number}
    U::Float64
    gamma::Tuple{T, T}
    eta::Tuple{T, T}
    function AuxFieldUpdaterParams{T}(U::Float64, scheme::DQMCScheme{T}) where T <: Number
        gamma, eta = decouplingCoefficients_CjWu(U, scheme.DeltaTau)
        new(U, gamma, eta)
    end
end

function auxCalc(s, params::AuxFieldUpdaterParams{T}) where T
    return sign(s) * params.eta[ abs(s) ]
end

function etaDiffCalc(s1, s2, params::AuxFieldUpdaterParams{T}) where T
    return (sign(s1) * params.eta[ abs(s1) ] - sign(s2) * params.eta[ abs(s2) ])
end

function initializeWalker(scheme::DQMCScheme{T}, params::AuxFieldUpdaterParams{T}, refreshInterval) where T
    fieldInit = rand(component4AuxField, scheme.Nsite, scheme.M)
    BMats = Vector{Matrix{T}}(undef, scheme.M)
    UDTsRight = Vector{UDT}(undef, scheme.M)
    UDTsLeft = Vector{UDT}(undef, scheme.M)

    for i in 1:scheme.M
        UDTsLeft[i] = udt!(Matrix{T}(I, scheme.Nsite, scheme.Nsite))
    end

    for i in 1:scheme.M
        BMats[i] = BMatGenerator(scheme, fieldInit[:, i], params)
    end

    UDTsRight[scheme.M] = udt(BMats[scheme.M])
    for i in scheme.M-1:-1:1
        UDTsRight[i] = fact_mult(UDTsRight[i+1], udt(BMats[i]) )
    end

    glTemp = inv_one_plus(UDTsRight[1])

    return DQMCWalker{T}(fieldInit, BMats, glTemp, UDTsRight, UDTsLeft, 0, refreshInterval, 1.0)
end

function BMatGenerator(scheme::DQMCScheme{T}, s::Vector{<:Integer}, params::AuxFieldUpdaterParams{T}) where T <: Complex
    B = deepcopy(scheme.KinecticDeltaTau)
    f(x) = exp(1.0im * auxCalc(x, params))
    rmul!(B, Diagonal(f.(s))) # B = exp(-Δτ K) [e^{Im η(s_i)}]
    return B
end

function invBMatGenerator(scheme::DQMCScheme{T}, s::Vector{<:Integer}, params::AuxFieldUpdaterParams{T}) where T <: Complex
    B = deepcopy(scheme.invKinecticDeltaTau)
    f(x) = exp(-1.0im * auxCalc(x, params))
    lmul!(Diagonal(f.(s)), B) # B =  [e^{-Im η(s_i)}] exp(+Δτ K)
    return B
end

# function AcceptRatioRaw(BMatsOld, BMatOverried::Matrix{T}, l) where T
#     BProd = det(I + prod(reverse(BMatsOld)))
#     BProdNew = Matrix{T}(I, size(BMatOverried))
    
#     for i in 1:l-1
#         BProdNew = BMatsOld[i] * BProdNew
#     end
#     BProdNew = BMatOverried * BProdNew
#     for i in l+1:length(BMatsOld)
#         BProdNew = BMatsOld[i] * BProdNew
#     end

#     BProdNew = det(I + BProdNew)
#     return BProdNew, BProd
# end

function glFromScratch(walker::DQMCWalker{T}, l) where T
    if l == 1
        return inv_one_plus(walker.UDTsRight[l])
    else
        return inv_one_plus( fact_mult(walker.UDTsLeft[l-1], walker.UDTsRight[l]) )
    end
end

function glFromScratchRaw(walker::DQMCWalker{T}, l, scheme::DQMCScheme{T}, params) where T
    BMats = Vector{Matrix{T}}(undef, scheme.M)
    for i in 1:scheme.M
        BMats[i] = BMatGenerator(scheme, walker.HSFieldConfig[:, i], params)
        # println(i, " ", sum(abs.(BMats[i] - walker.BMats[i])))
    end

    x = udt!(Matrix{T}(I, size(BMats[1])))
    for i in l:scheme.M
        x = fact_mult(udt(BMats[i]), x)
    end
    for i in 1:l-1
        x = fact_mult(udt(BMats[i]), x)
    end
    return inv_one_plus(x)
end

function AcceptRatioR(walker::DQMCWalker{T}, sOld, sNew, i, params::AuxFieldUpdaterParams{T}) where T
    etaDiff = etaDiffCalc(sNew, sOld, params)
    R = 1 + (1 - walker.glTemp[i, i]) * ( exp(1.0im * etaDiff) - 1)
    return R, etaDiff # * exp(-0.5im * etaDiff)
end

function proposeAuxMove(s)
    r = rand()
    if r <= 0.5
        return -s
    elseif r <= 0.75
        return 3 - abs(s)
    else
        return abs(s) - 3
    end
end

function iteratedUpdateGlToNextTime!(walker::DQMCWalker{T}, l, params::AuxFieldUpdaterParams{T}, scheme::DQMCScheme{T}) where T
    if l == scheme.M
        updateUDTsRight!(walker, scheme)
    end


    # if walker.counter % (walker.refreshInterval) == 0
    if walker.counter % walker.refreshInterval == 0
        
        # if walker.counter % 100*walker.refreshInterval == 0
        @debug begin
            glScratch = glFromScratch(walker, (l % scheme.M) + 1)
            # println("debugging")
            gTemp = walker.BMats[l] * walker.glTemp * invBMatGenerator(scheme, walker.HSFieldConfig[:, l], params)
            glScratchRaw = glFromScratchRaw(walker, (l % scheme.M) + 1, scheme, params)
            error1 = sum(abs.(gTemp - glScratchRaw))
            error2 = sum(abs.(glScratch - glScratchRaw))
            "error1 = " * string(error1) * " error2 = " * string(error2) * "\n l=" * string(l)
        end
        # end

        glScratch = glFromScratch(walker, (l % scheme.M) + 1)
        walker.glTemp = glScratch
    else
        walker.glTemp = walker.BMats[l] * walker.glTemp * invBMatGenerator(scheme, walker.HSFieldConfig[:, l], params)
        # walker.glTemp = walker.BMats[l] * walker.glTemp * inv(walker.BMats[l])
    end

    walker.counter += 1
end

function iteratedUpdateGlNewConfig!(walker::DQMCWalker{T}, r, i, etaDiff, l, scheme::DQMCScheme{T}, sNew, sOld) where T
    # N = size(walker.glTemp)[0]
    # gNew = Matrix{T}(undef, N, N)
    # g = walker.glTemp
    γ = exp(1.0im * etaDiff)

    walker.HSFieldConfig[i, l] = sNew
    walker.BMats[l][:, i] *= γ

    # g \to new g
    row = walker.glTemp[:, i]
    row[i] -= 1
    walker.glTemp = walker.glTemp + ( (γ-1) / r * (row * transpose(walker.glTemp[i, :])) )
end

function iteratedUpdateUDTsLeftAfter!(walker::DQMCWalker{T}, l, scheme::DQMCScheme{T}) where T
    BlUDT = udt(walker.BMats[l])
    if l == 1
        walker.UDTsLeft[1] = BlUDT
    else
        walker.UDTsLeft[l] = fact_mult(BlUDT, walker.UDTsLeft[l-1])
    end
end

function updateUDTsRight!(walker::DQMCWalker{T}, scheme::DQMCScheme{T}) where T
    walker.UDTsRight[scheme.M] = udt(walker.BMats[scheme.M])
    for i in scheme.M-1:-1:1
        walker.UDTsRight[i] = fact_mult(walker.UDTsRight[i+1], udt(walker.BMats[i]) )
    end
end

function sumOfEtaCalc(walker::DQMCWalker{T}, params::AuxFieldUpdaterParams{T}) where T
    f(s) = sign(s) * params.eta[ abs(s) ]
    return sum( f.(walker.HSFieldConfig) )
end

function walkerMarkovMove!(walker::DQMCWalker{T}, params::AuxFieldUpdaterParams{T}, scheme::DQMCScheme{T}) where T
    for l in 1:scheme.M
        # sVec = deepcopy(walker.HSFieldConfig[:, l])

        for i in 1:scheme.Nsite
            sOld = walker.HSFieldConfig[i, l]
            sNew = proposeAuxMove(sOld)
            # sVec[i] = sNew
            # BMatNew = BMatGenerator(scheme, sVec, params)
            # sVec[i] = sOld
            # @printf("i, l = %d %d, sOld, sNew = %d %d: ", i, l, sOld, sNew)
            # detNew, detOld = AcceptRatioRaw(walker.BMats, BMatNew, l)
            # rRaw = detNew / detOld * exp(-0.5im * etaDiffCalc(sNew, sOld, params))
            r, etaDiff = AcceptRatioR(walker, sOld, sNew, i, params)
#             p = (r * exp(-0.5im * etaDiff))^(scheme.Nflavor) * (params.gamma[abs(sNew)] / params.gamma[abs(sOld)])
            p = (r)^(scheme.Nflavor) * (params.gamma[abs(sNew)] / params.gamma[abs(sOld)])
            # println("prob: ", p)
            if abs(p) >= 1.0 || rand() < abs(p)
                iteratedUpdateGlNewConfig!(walker, r, i, etaDiff, l, scheme, sNew, sOld)
            end
        end
        
        iteratedUpdateUDTsLeftAfter!(walker, l, scheme)
        iteratedUpdateGlToNextTime!(walker, l, params, scheme)
    end
    walker.individualWeight = individualSpinDet(walker, params)

end

function individualSpinDet(walker::DQMCWalker{T}, params::AuxFieldUpdaterParams{T}) where T
    etasum = sumOfEtaCalc(walker, params)
    U, D, V = walker.UDTsRight[1]
    m = U'/ V
    m[diagind(m)] .+= D
    detIPlusBs = det(U) * det(m) * det(V)
    return exp(-0.5im*etasum) * detIPlusBs
end
# end