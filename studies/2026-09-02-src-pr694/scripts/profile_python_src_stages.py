import os,sys,types,time,json
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
matplotlib=types.ModuleType('matplotlib');matplotlib.pyplot=types.ModuleType('matplotlib.pyplot');sys.modules['matplotlib']=matplotlib;sys.modules['matplotlib.pyplot']=matplotlib.pyplot;sys.modules['torch']=types.ModuleType('torch');sys.modules['seaborn']=types.ModuleType('seaborn');sys.path.insert(0,'/tmp/RandomMPOMPS-pr694/code')
import numpy as np
from tensornetwork.MPO import MPO
from tensornetwork.MPS import MPS
from tensornetwork.contraction import random_contraction_inc
from tensornetwork.stopping import Cutoff
bond=int(sys.argv[1]);np.random.seed(13);mps=MPS.random_mps(10,bond,d=2,dtype=float);mpo=MPO.random_mpo(10,bond,d=2,d2=2,dtype=float);stop=Cutoff(1e-4,mindim=2,maxdim=32)
def run():
 np.random.seed(1234);return random_contraction_inc(mpo,mps,stop=stop,sketchdim=2,sketchincrement=3,finalround=None,accuracychecks=False,cpp=False,debug=False)
run();target=random_contraction_inc.__code__;totals={};events=0;state={}
def stage(line):
 if line<428:return 'initialization'
 if line<458:return 'environment'
 if line<492:return 'sketch'
 if line<514:return 'qr'
 if line<526:return 'error_estimator'
 if line<578:return 'projection'
 if line<582:return 'correction'
 return 'finalization'
def trace(frame,event,arg):
 global events
 if frame.f_code is not target:return trace
 now=time.perf_counter_ns();key=id(frame)
 if key in state:
  prev,t0=state[key];totals[stage(prev)]=totals.get(stage(prev),0)+now-t0
 if event=='line':state[key]=(frame.f_lineno,now);events+=1
 elif event=='return':state.pop(key,None)
 return trace
sys.settrace(trace);start=time.perf_counter_ns();result=run();elapsed=time.perf_counter_ns()-start;sys.settrace(None)
print(json.dumps({'elapsed_ms':elapsed/1e6,'events':events,'stages_ms':{k:v/1e6 for k,v in totals.items()}}))
