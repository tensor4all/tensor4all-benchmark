# Bounded attempts

`status.tsv` columns are Gaussian count, process status, and integer wall seconds for the combined global-plus-patched command. N=1 completed; N=2 reached `timeout 570s` and emitted no record. A separate N=2 patch-only structural diagnostic with reference counting disabled also reached status 124 after 570 s and emitted no record.
