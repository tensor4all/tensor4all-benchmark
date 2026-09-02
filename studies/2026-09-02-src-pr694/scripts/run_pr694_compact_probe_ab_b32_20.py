import json, os, re, statistics, subprocess
binary='/tmp/t4a-pr694/target/release/examples/benchmark_src'
base=os.environ|{'RAYON_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','T4A_BENCH_ALGORITHM':'src-adaptive','T4A_BENCH_WARMUP':'1'}
pat=re.compile(r'case=mpo-mps/src-adaptive .* per_run=([0-9.]+)s .*relative_error=([0-9.eE+-]+)')
rows=[]
for bond in [32]:
  for pair in range(1,21):
    row={'bond':bond,'pair':pair}
    for variant in (['compact','baseline'] if pair%2 else ['baseline','compact']):
      env=base.copy()
      if variant=='compact':env['T4A_AB_COMPACT_DENSE_ANY']='1'
      out=subprocess.check_output(['taskset','-c','63',binary,'10',str(bond),'5','mpo-mps','3','false','32'],env=env,text=True)
      m=pat.search(out); assert m,out
      row[variant+'_ms']=float(m.group(1))*1000;row[variant+'_error']=float(m.group(2))
    row['change_pct']=(row['compact_ms']/row['baseline_ms']-1)*100
    rows.append(row);print(json.dumps(row),flush=True)
for bond in [32]:
  xs=[r for r in rows if r['bond']==bond]
  b=statistics.median(r['baseline_ms'] for r in xs);c=statistics.median(r['compact_ms'] for r in xs)
  print(json.dumps({'bond':bond,'baseline_median_ms':b,'compact_median_ms':c,'ratio_change_pct':(c/b-1)*100,'paired_change_median_pct':statistics.median(r['change_pct'] for r in xs),'max_error':max(max(r['baseline_error'],r['compact_error']) for r in xs)}),flush=True)
open('/tmp/pr694_compact_probe_ab_b32_20.json','w').write(json.dumps(rows,indent=2))
