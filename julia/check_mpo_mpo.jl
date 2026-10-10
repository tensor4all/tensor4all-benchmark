# Usage: julia --project=julia julia/check_mpo_mpo.jl <dir> <n>
#
# Independent ITensors.jl check of case 3's inputs. Reads instance-n<n>.h5 (MPS
# groups "f" and "g", one site of dimension 4 per quantics bit pair) and
# instance-n<n>.json, evaluates both MPS at sampled fused grid indices, and
# compares with the closed form the interpolative construction approximates:
#
#     f(x, y) = sum_i w_i exp(-(a_i dx^2 + 2 b_i dx dy + c_i dy^2))
#
# The tensors are written by the Rust arm through tensor4all-hdf5's ITensorMPS
# `MPS` schema, so a mismatch here is a mismatch between what Rust stores and
# what an independent implementation reads and contracts.
using HDF5, ITensors, ITensorMPS, JSON3
using Random

Random.seed!(0)

dir, nstr = ARGS[1], ARGS[2]
n = parse(Int, nstr)
meta = JSON3.read(read(joinpath(dir, "instance-n$nstr.json"), String))
@assert meta.schema_version == 1
# Cases 2 and 3 once shared the instance file names, so the name is asserted
# rather than trusted.
@assert meta.case == "gaussian_mpo_contraction"
@assert meta.n_gauss == n
r = meta.r
L = meta.box_l

# The check catches a broken export or a wrong index convention, not an accuracy
# bound: the instance is truncated relative to its own norm, so the pointwise
# deviation scales with that tolerance. Overridable for a tighter local run.
threshold = parse(Float64, get(ENV, "CHECK_THRESHOLD", "1e-4"))

function mixture(m, x, y)
    s = 0.0
    for i in eachindex(m.weights)
        q = m.quad[i]
        a, b, c = q[1], q[2], q[3]
        center = m.centers[i]
        cx, cy = center[1], center[2]
        dx, dy = x - cx, y - cy
        s += m.weights[i] * exp(-(a * dx^2 + 2 * b * dx * dy + c * dy^2))
    end
    return s
end

coord(i) = -L + i * 2.0 * L / 2^r

function eval_mps(psi::MPS, locals::Vector{Int})
    s = siteinds(psi)
    v = ITensor(1.0)
    for m in eachindex(psi)
        v *= psi[m] * onehot(s[m] => locals[m] + 1)
    end
    return scalar(v)
end

fails = 0
maxdev = 0.0
h5open(joinpath(dir, "instance-n$nstr.h5"), "r") do file
    for (name, m) in (("f", meta.left), ("g", meta.right))
        psi = read(file, name, MPS)
        @assert length(psi) == r
        @assert all(dim(s) == 4 for s in siteinds(psi))
        for trial in 1:50
            ix, iy = rand(0:(2^r - 1)), rand(0:(2^r - 1))
            xb = [Int((ix >> (r - b)) & 1) for b in 1:r]  # MSB first
            yb = [Int((iy >> (r - b)) & 1) for b in 1:r]
            # Fused local index s = x + 2y with x least significant, matching
            # gaussian::fused_qtt_to_mpo.
            fused = [xb[b] + 2 * yb[b] for b in 1:r]
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
