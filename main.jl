include("./DQMC.jl")
include("./KinecticReader.jl")
# using .DQMC
import Statistics

const dtype = ComplexF64

nx = 6
ny = 1
Δτ = 0.125
β = 1.0
nflavor = 2
U = 4.0

# M, lattice length in imaginary time direction
M = Int((β+0.01) ÷ Δτ)
# println(M)

thermalizationTime = 200
iterationTime = 1000
# samplesPerIter = 20

Nsites, Kinectic = BuildKinecticFromYAML("1dRingHamiltonian.yml", nx, ny, dtype)
schemeSU2N = DQMCScheme{dtype}(Nsites, nflavor, M, Δτ, Kinectic)
paramsAux = AuxFieldUpdaterParams{dtype}(U, schemeSU2N)
x = initializeWalker(schemeSU2N, paramsAux);

for i in 1:thermalizationTime
    walkerMarkovMove!(x, paramsAux, schemeSU2N)
    # println()
end

function SsqureObs(x::DQMCWalker, scheme)
    sum = 0
    for i in 1:scheme.Nsite
        sum += 2*(1 - x.glTemp[i, i]) * x.glTemp[i, i]
    end
    return 0.75 * sum / scheme.Nsite
end


s2 = Vector{dtype}(undef, iterationTime)

for i in 1:iterationTime
    walkerMarkovMove!(x, paramsAux, schemeSU2N)
    s2[i] = SsqureObs(x, schemeSU2N)
end

Statistics.mean(real.(s2)), Statistics.std(real.(s2))
# for i in 1:iterationTime

# println(auxParams.eta, auxParams.gamma)

# println(SU2NHubbardScheme.KinecticDeltaTau)

# println(x.HSFieldConfig)

# println("=============")

# for i in 1:length(x.BMats)
#     println(x.BMats[i])
# end

