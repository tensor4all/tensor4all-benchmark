import json, os, re, statistics, subprocess

binary = "/tmp/t4a-pr694/target/release/examples/benchmark_src"
base_env = os.environ | {
    "RAYON_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "T4A_BENCH_ALGORITHM": "src-adaptive", "T4A_BENCH_WARMUP": "1",
}
pat = re.compile(r"case=mpo-mps/src-adaptive .* per_run=([0-9.]+)s .*relative_error=([0-9.eE+-]+)")
rows=[]
for bond in [32,64,128]:
    for pair in range(1, 6):
        row={"bond":bond,"pair":pair}
        order = ["scope","baseline"] if pair % 2 else ["baseline","scope"]
        for variant in order:
            env=base_env.copy()
            if variant == "scope": env["T4A_BENCH_SESSION_SCOPE"]="1"
            out=subprocess.check_output(["taskset","-c","63",binary,"10",str(bond),"5","mpo-mps","3","false","32"],env=env,text=True)
            match=pat.search(out)
            assert match, out
            row[variant+"_ms"]=float(match.group(1))*1000
            row[variant+"_error"]=float(match.group(2))
        row["change_pct"]=(row["scope_ms"]/row["baseline_ms"]-1)*100
        rows.append(row); print(json.dumps(row),flush=True)
for bond in [32,64,128]:
    xs=[r for r in rows if r["bond"]==bond]
    b=statistics.median(r["baseline_ms"] for r in xs); s=statistics.median(r["scope_ms"] for r in xs)
    print(json.dumps({"bond":bond,"pairs":len(xs),"baseline_median_ms":b,"scope_median_ms":s,"change_pct":(s/b-1)*100,"paired_change_median_pct":statistics.median(r["change_pct"] for r in xs),"baseline_range_ms":[min(r["baseline_ms"] for r in xs),max(r["baseline_ms"] for r in xs)],"scope_range_ms":[min(r["scope_ms"] for r in xs),max(r["scope_ms"] for r in xs)],"max_error":max(max(r["baseline_error"],r["scope_error"]) for r in xs)}),flush=True)
open('/tmp/pr694_sweep_session_ab.json','w').write(json.dumps(rows,indent=2))
