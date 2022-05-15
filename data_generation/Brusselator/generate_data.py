'''
Manuscript Associated: Deep transfer operator learning for partial differential equations under conditional shift
Authors: Katiana Kontolati, PhD Candidate, Johns Hopkins University
         Somdatta Goswami, Postdoctoral Researcher, Brown University
This script generates the data employed in the Brusselator application (TL5, TL6)
'''

import numpy as np
from matplotlib import pyplot as plt
from pde import PDE, FieldCollection, PlotTracker, ScalarField, UnitGrid, MemoryStorage, movie_scalar
import os

file = np.load('./inputs_KLE_lx_0.3_ly_0.4_v_0.15.npz') 

samples = file['inputs']

n_samples = 1000 # both train and test
print('Number of total samples:', n_samples)

nx, ny = 28, 28
samples_r = samples[:n_samples].reshape(n_samples, int(nx*ny)) # reshape

# Run Brusselator model 

# Define the PDE
a, b = 1, 1.7   # b=2.2, b=3.0
d0, d1 = 1, 0.5

eq = PDE(
    {
        "u": f"{d0} * laplace(u) + {a} - ({b} + 1) * u + u**2 * v",
        "v": f"{d1} * laplace(v) + {b} * u - u**2 * v",
    }
)

# Run experiments
import time

start = time.time() # save CPU time
storage_outer = []
for i in range(n_samples):
    
    print('Iteration: {}'.format(i+1))
    get = []
    storage = MemoryStorage()
    
    # initialize state
    grid = UnitGrid([nx,ny])
    u = ScalarField(grid, a, label="Field $u$")
    
    # use as initial field, GRF realizations
    v = ScalarField.random_normal(grid, label="Field $v$")
    v._data = samples[i]
    
    state = FieldCollection([u, v])

    # simulate the pde
    tracker = [storage.tracker(interval=0.3)]
    t_range, dt = 10, 1e-2
    sol = eq.solve(state, t_range=t_range, dt=dt, tracker=tracker)
    
    output = np.array((storage.data))[:,1,:,:]   # v
    storage_outer.append(output)
    
finish = time.time() - start
print('Total time: ',finish)

outputs_all = np.array(storage_outer) # np.array

# Save results for specific time steps
snap = 20 # keep a given number of snapshots 
time_steps = list(np.round(np.linspace(0, outputs_all.shape[1]-1, snap)))
time_steps = [int(x) for x in time_steps]
time_steps[0] = 1

outputs = outputs_all[:,time_steps,:,:].reshape(n_samples,int(len(time_steps)*nx*ny))

datafile = 'Brusselator_data_KLE_lx_0.3_ly_0.4_v_0.15.npz' 

datadir = './'
np.savez(os.path.join(datadir, datafile), nx=nx, ny=ny, n_samples=n_samples, inputs=samples_r, outputs=outputs)

