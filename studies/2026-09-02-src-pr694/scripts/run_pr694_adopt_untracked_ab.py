import json,os,re,statistics,subprocess
bins={'baseline':'/tmp/benchmark_src.pr694-concrete-session-warm','candidate':'/tmp/benchmark_src.pr694-adopt-untracked'};env=os.environ|{'RAYON_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','T4A_BENCH_ALGORITHM':'src-adaptive','T4A_BENCH_WARMUP':'1'};pat=re.compile(r'case=mpo-mps/src-adaptive .* per_run=([0-9.]+)s .*relative_error=([0-9.eE+-]+)');rows=[]
for bond in [32,64,128]:
 for pair in range(1,11):
  row={'bond':bond,'pair':pair}
  for v in (['candidate','baseline'] if pair%2 else ['baseline','candidate']):
   out=subprocess.check_output(['taskset','-c','63',bins[v],'10',str(bond),'5','mpo-mps','3','false','32'],env=env,text=True);m=pat.search(out);assert m,out;row[v+'_ms']=float(m.group(1))*1000;row[v+'_error']=float(m.group(2))
  row['change_pct']=(row['candidate_ms']/row['baseline_ms']-1)*100;rows.append(row);print(json.dumps(row),flush=True)
for bond in [32,64,128]:
 xs=[r for r in rows if r['bond']==bond];b=statistics.median(r['baseline_ms'] for r in xs);c=statistics.median(r['candidate_ms'] for r in xs);print(json.dumps({'bond':bond,'baseline_ms':b,'candidate_ms':c,'saved_ms':b-c,'ratio_change_pct':(c/b-1)*100,'paired_change_median_pct':statistics.median(r['change_pct'] for r in xs),'max_error':max(max(r['baseline_error'],r['candidate_error']) for r in xs)}),flush=True)
open('/tmp/pr694_adopt_untracked_ab.json','w').write(json.dumps(rows,indent=2))
