import YAML

function coordinateToIdx(ix, iy, iorbit, Lx, Ly)
    ix = (ix - 1) % Lx + 1
    iy = (iy - 1) % Ly
    return ix + iy * Lx + Lx * Ly * iorbit
end

function addHoppingToKinetic!(K, t, orb1, orb2, dx, dy, Lx, Ly)
    for j in 1:Ly, i in 1:Lx
        s, e = coordinateToIdx(i, j, orb1, Lx, Ly), coordinateToIdx(i + dx, j + dy, orb2, Lx, Ly)
        K[s, e] += t
        K[e, s] += conj(t)
    end
end

function BuildKinecticFromYAML(filename::String, Lx, Ly, T)
    data = YAML.load_file(filename)
    nOrbits = data["nOrbits"]
    Nsites = Lx * Ly * nOrbits
    K = zeros(T, Nsites, Nsites)
    for term in data["terms"]
        scal = term["scaling factor"]
        if scal != 0.0
            for hopping in term["hoppings"]
                orb1, orb2 = eval(Meta.parse(hopping["orbits"]))
                dx, dy = eval(Meta.parse(hopping["dsite"]))
                t = eval(Meta.parse(hopping["value"]))
                # println(orb1, orb2, dx, dy, t)
                addHoppingToKinetic!(K, t * scal, orb1, orb2, dx, dy, Lx, Ly)
            end
        end
    end
    return Nsites, K
end