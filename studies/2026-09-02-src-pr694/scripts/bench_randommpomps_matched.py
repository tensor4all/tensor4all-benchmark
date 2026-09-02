import json
import os
import sys
import time
import types

os.environ.update(
    OPENBLAS_NUM_THREADS="1",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
)

matplotlib = types.ModuleType("matplotlib")
matplotlib.pyplot = types.ModuleType("matplotlib.pyplot")
sys.modules["matplotlib"] = matplotlib
sys.modules["matplotlib.pyplot"] = matplotlib.pyplot
sys.modules["torch"] = types.ModuleType("torch")
sys.modules["seaborn"] = types.ModuleType("seaborn")
sys.path.insert(0, "/tmp/RandomMPOMPS-pr694/code")

import numpy as np
from tensornetwork.MPO import MPO
from tensornetwork.MPS import MPS
from tensornetwork.contraction import random_contraction_inc
from tensornetwork.stopping import Cutoff

bond = int(sys.argv[1])
reps = int(sys.argv[2])
np.random.seed(13)
mps = MPS.random_mps(10, bond, d=2, dtype=float)
mpo = MPO.random_mpo(10, bond, d=2, d2=2, dtype=float)
stop = Cutoff(1e-4, mindim=2, maxdim=32)

def run():
    np.random.seed(1234)
    return random_contraction_inc(
        mpo,
        mps,
        stop=stop,
        sketchdim=2,
        sketchincrement=3,
        finalround=None,
        accuracychecks=False,
        cpp=False,
        debug=False,
    )

run()
times = []
result = None
for _ in range(reps):
    start = time.perf_counter()
    result = run()
    times.append(time.perf_counter() - start)
bonds = [result[0].shape[1]] + [result[j].shape[2] for j in range(1, result.N - 1)]
print(json.dumps({"bond": bond, "times_s": times, "output_bonds": bonds}))
