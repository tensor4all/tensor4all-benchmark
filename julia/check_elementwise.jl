# Usage: julia --project=julia julia/check_elementwise.jl <dir> <k>
# Reads instance-k<k>.h5 (MPS groups "f","g") and instance-k<k>.json,
# evaluates both MPS at sample grid points, compares to the analytic series.
using HDF5, ITensors, ITensorMPS, JSON3

dir, k = ARGS[1], ARGS[2]
meta = JSON3.read(read(joinpath(dir, "instance-k$k.json"), String))
R = meta.r

coeffs(v) = [complex(c[1], c[2]) for c in v]
series(c, x) = sum(c[j+1] * exp(2im * pi * j * x) for j in 0:length(c)-1)

function eval_mps(psi::MPS, bits::Vector{Int})
    s = siteinds(psi)
    v = ITensor(1.0)
    for n in eachindex(psi)
        v *= psi[n] * onehot(s[n] => bits[n] + 1)
    end
    return scalar(v)
end

fails = 0
h5open(joinpath(dir, "instance-k$k.h5"), "r") do file
    for (name, cs) in (("f", coeffs(meta.f_coeffs)), ("g", coeffs(meta.g_coeffs)))
        psi = read(file, name, MPS)
        @assert length(psi) == R
        for trial in 1:50
            i = rand(0:(2^R - 1))
            bits = [Int((i >> (R - n)) & 1) for n in 1:R]  # MSB first
            x = i / 2^R
            got = eval_mps(psi, bits)
            want = series(cs, x)
            if abs(got - want) > 1e-6
                global fails += 1
                println("MISMATCH $name x=$x got=$got want=$want")
            end
        end
    end
end
fails == 0 || error("$fails mismatches")
println("check_elementwise: OK")
