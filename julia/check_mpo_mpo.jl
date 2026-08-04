# Usage: julia --project=julia julia/check_mpo_mpo.jl <dir> <r>
# Reads instance-r<r>.h5 (MPS groups "f","g", site dim 4 = fused (x,y) bit pair)
# and instance-r<r>.json, evaluates both MPS at random fused grid indices and
# compares to the analytic Gaussian-mixture formula.
using HDF5, ITensors, ITensorMPS, JSON3
using Random
Random.seed!(0)

dir, rstr = ARGS[1], ARGS[2]
R = parse(Int, rstr)
meta = JSON3.read(read(joinpath(dir, "instance-r$rstr.json"), String))
@assert meta.r == R
L = meta.box_l
# The exported TTs are TCI approximations at meta.tolerance, and the truncation
# is norm-relative, so the pointwise error scales with the tolerance rather than
# being bounded by a fixed absolute number (same reasoning as check_elementwise).
threshold = 1e3 * meta.tolerance

function mixture(m, x, y)
    s = 0.0
    for i in eachindex(m.weights)
        cx, cy = m.centers[i]
        s += m.weights[i] * exp(-m.alphas[i] * ((x - cx)^2 + (y - cy)^2))
    end
    return s
end

coord(i) = -L + i * 2L / 2^R

function eval_mps(psi::MPS, locals::Vector{Int})
    s = siteinds(psi)
    v = ITensor(1.0)
    for n in eachindex(psi)
        v *= psi[n] * onehot(s[n] => locals[n] + 1)
    end
    return scalar(v)
end

fails = 0
maxdev = 0.0
h5open(joinpath(dir, "instance-r$rstr.h5"), "r") do file
    for (name, m) in (("f", meta.f), ("g", meta.g))
        psi = read(file, name, MPS)
        @assert length(psi) == R
        @assert all(dim(s) == 4 for s in siteinds(psi))
        for trial in 1:50
            ix, iy = rand(0:(2^R - 1)), rand(0:(2^R - 1))
            xb = [Int((ix >> (R - n)) & 1) for n in 1:R]  # MSB first
            yb = [Int((iy >> (R - n)) & 1) for n in 1:R]
            # Fused local index s = s1 + 2*s2, x (variable 1) least significant;
            # matches the Rust side (see gaussian::to_quantics_mpo).
            fused = [xb[n] + 2 * yb[n] for n in 1:R]
            got = eval_mps(psi, fused)
            want = mixture(m, coord(ix), coord(iy))
            dev = abs(got - want) / max(1.0, abs(want))
            global maxdev = max(maxdev, dev)
            if dev > threshold
                global fails += 1
                println("MISMATCH $name ($(coord(ix)), $(coord(iy))): $got vs $want")
            end
        end
    end
end
println("max relative deviation: $maxdev (threshold $threshold)")
fails == 0 || error("$fails mismatches")
println("check_mpo_mpo: OK")
