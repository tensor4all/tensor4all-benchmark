import json,os,re,statistics,subprocess
binary='/tmp/t4a-pr694/target/release/examples/benchmark_src';base=os.environ|{'RAYON_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','T4A_BENCH_ALGORITHM':'src-adaptive','T4A_BENCH_WARMUP':'1','T4A_AB_DIRECT_REAL_PROBE':'1'};pat=re.compile(r'case=mpo-mps/src-adaptive .* per_run=([0-9.]+)s .*relative_error=([0-9.eE+-]+)');rows=[]
for bond in [32,64,128]:
 for pair in range(1,11):
  row={'bond':bond,'pair':pair}
  for v in (['untracked','leaf'] if pair%2 else ['leaf','untracked']):
   env=base.copy()
   if v=='untracked':env['T4A_AB_UNTRACKED_RESULT']='1'
   out=subprocess.check_output(['taskset','-c','63',binary,'10',str(bond),'5','mpo-mps','3','false','32'],env=env,text=True);m=pat.search(out);assert m,out;row[v+'_ms']=float(m.group(1))*1000;row[v+'_error']=float(m.group(2))
  row['change_pct']=(row['untracked_ms']/row['leaf_ms']-1)*100;rows.append(row);print(json.dumps(row),flush=True)
for bond in [32,64,128]:
 xs=[r for r in rows if r['bond']==bond];b=statistics.median(r['leaf_ms'] for r in xs);d=statistics.median(r['untracked_ms'] for r in xs);print(json.dumps({'bond':bond,'leaf_ms':b,'untracked_ms':d,'saved_ms':b-d,'ratio_change_pct':(d/b-1)*100,'paired_change_median_pct':statistics.median(r['change_pct'] for r in xs),'max_error':max(max(r['leaf_error'],r['untracked_error']) for r in xs)}),flush=True)
open('/tmp/pr694_untracked_result_ab.json','w').write(json.dumps(rows,indent=2))
