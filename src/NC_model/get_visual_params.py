import h5py
import numpy as np
from neuron.units import mV, ms
from IdenTools import *
'''
DATAFILE = h5py.File('/home/g6234203723/Spiking_patterns/src/NC_model/dataset/twoparam.hdf5','r')
SAVEFILE = h5py.File('/home/g6234203723/Spiking_patterns/src/NC_model/dataset/twoparam_prop.hdf5','w')
ts = np.linspace(0,1000,40001)
conds = ['borgkdr', 'cagk', 'cal', 'can', 'hd', 'kad', 'kahp', 'kap', 'nahh']
for cond in conds:
    SAVEFILE.create_dataset(cond, (100,100,1000), dtype="float64")
    for i in range(100):
        for j in range(100):
            SAVEFILE[cond][i,j] = GetProp(ts, DATAFILE[cond][i,j])
    print(cond + " DONE!")

'''
DATAFILE = h5py.File('/home/g6234203723/Spiking_patterns/src/NC_model/dataset/threeparams.hdf5','r')
SAVEFILE = h5py.File('/home/g6234203723/Spiking_patterns/src/NC_model/dataset/threeparams_prop.hdf5','w')
ts = np.linspace(0,1000,40001)
count=0
for cond in DATAFILE.keys():
    SAVEFILE.create_dataset(cond, (20,20,20,1000), dtype='float64')
    for i in range(20):
        for j in range(20):
            for k in range(20):
                SAVEFILE[cond][i,j,k] = GetProp(ts, DATAFILE[cond][i,j,k])
        print("run for {} progressed by {}/20".format(cond, i+1))
    count+=1
    print("just {} more to go!!!".format(45-count))
SAVEFILE.close()
DATAFILE.close()
