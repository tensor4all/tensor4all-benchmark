# Factor-level cap diagnostic for the direct-product idea

For the doubled input `F=f tensor_product f`, a Cartesian product of factor patches with factor cap 11 has product-patch rank at most `11^2 = 121 <= 128`. This profile measures the original 2D factor contraction at `R=16`, `N=1`, input rtol `1e-6`, factor patch cap 11. It does not present inferred doubled-space timings.

| arm | median contraction (s) | error | factor input patches L/R | compatible pairs | output groups |
|---|---:|---:|---:|---:|---:|
| global fit | 0.0792 | 7.70e-7 | - | - | - |
| patched fit | 110.508 | 3.07e-6 | 128 / 64 | 1,024 | 64 |

The patched factor contraction is 1,396x slower, before adding the excluded 11.39 s patch preparation. Both measured errors pass the benchmark's `1e-4` external sanity gate; the patched result is not claimed to meet a `1e-6` end-to-end output-error bound. Its exact Cartesian lift to the doubled problem would contain:

- left product patches: `128^2 = 16,384`;
- right product patches: `64^2 = 4,096`;
- compatible product pairs: `1,024^2 = 1,048,576`;
- output projector groups: `64^2 = 4,096`.

This comparison shows that cap-11 patching is expensive on the original 2D factor, while a structure-aware solution should compute the original contraction and form the direct product of its result. It does **not** compare generic doubled-space global and patched contractions: the measured global arm already exploits factorization, and only the patch counts were lifted structurally. Therefore no generic cap-128 direct-product speedup or slowdown is claimed. An N=2 factor run exceeded the 570 s command bound, and its patch-only diagnostic also exceeded 570 s; no N=2 timing is reported.
