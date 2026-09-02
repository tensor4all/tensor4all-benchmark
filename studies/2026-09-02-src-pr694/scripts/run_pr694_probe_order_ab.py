import json, os, re, statistics, subprocess
bins={"nary":"/tmp/benchmark_src.pr694-current-0b0513d","explicit":"/tmp/benchmark_src.pr694-explicit-probe-first"}
env=os.environ|{"RAYON_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","T4A_BENCH_SKIP_EXACT":"1","T4A_BENCH_ALGORITHM":"src-adaptive"}
rows=[]
for bond in [32,64,128]:
 for pair in range(1,11):
  order=list(bins) if pair%2 else list(reversed(bins))
  row={"bond":bond,"pair":pair}
  for name in order:
   out=subprocess.check_output(["taskset","-c","0",bins[name],"10",str(bond),"5","mpo-mps","3","false","32"],env=env,text=True)
   row[name]=float(re.search(r'per_run=([0-9.]+)s',out).group(1))*1000
  rows.append(row);print(json.dumps(row),flush=True)
for bond in [32,64,128]:
 a=statistics.median(r['nary'] for r in rows if r['bond']==bond);b=statistics.median(r['explicit'] for r in rows if r['bond']==bond)
 print(json.dumps({"bond":bond,"nary_ms":a,"explicit_ms":b,"change_pct":(b/a-1)*100,"delta_ms":b-a}))
json.dump(rows,open('/tmp/pr694_probe_order_ab.json','w'),indent=2)
