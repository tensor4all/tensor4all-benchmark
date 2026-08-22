# Bounded attempts

`status.tsv` columns are Gaussian count, process status, and integer wall seconds for the combined global-plus-patched command. N=1 completed; N=2 reached `timeout 570s` and emitted no record. `patch-only-status.tsv` uses the same columns for a separate N=2 structural diagnostic with reference counting disabled; it also reached status 124 after 570 s and emitted no record.
