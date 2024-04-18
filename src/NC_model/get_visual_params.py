import h5py
import numpy as np
from neuron.units import mV, ms
from IdenTools import *

DATAFILE = h5py.File('home/g6234203723/Spiking_patterns/src/NC_model/dataset/twoparams.hdf5','r')
SAVEFILE = h5py.File('home/g6234203723/Spiking_patterns/src/NC_model/dataset/twoparams_prop.hdf5','r+')

conds = ['borgkdr', 'cagk', 'cal', 'can', 'hd', 'kad', 'kahp', 'kap', 'nahh']
ts = np.linspace(0,1000,40001)
for cond in conds:
    for i in range(100):
        for j in range(100):
            SAVEFILE[cond][i,j] = GetProp(ts, DATAFILE[cond][i,j])
SAVEFILE.close()
DATAFILE.close()