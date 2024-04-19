import h5py
import numpy as np
from neuron.units import mV, ms
from IdenTools import *
'''
DATAFILE = h5py.File('home/g6234203723/Spiking_patterns/src/NC_model/dataset/twoparams.hdf5','r')
SAVEFILE = h5py.File('home/g6234203723/Spiking_patterns/src/NC_model/dataset/twoparams_prop.hdf5','r+')

conds = ['borgkdr', 'cagk', 'cal', 'can', 'hd', 'kad', 'kahp', 'kap', 'nahh']
ts = np.linspace(0,1000,40001)
for cond in conds:
    for i in range(100):
        for j in range(100):
            SAVEFILE[cond][i,j] = GetProp(ts, DATAFILE[cond][i,j])
'''

DATAFILE = h5py.File('home/g6234203723/Spiking_patterns/src/NC_model/dataset/threeparams.hdf5','r')
SAVEFILE = h5py.File('home/g6234203723/Spiking_patterns/src/NC_model/dataset/threeparams_prop.hdf5','r+')

conds = ['borgkdrvborgkdr', 'borgkdrvcagk', 'borgkdrvcal', 'borgkdrvcan', 'borgkdrvcat', 'borgkdrvhd', \
        'borgkdrvkad', 'borgkdrvkahp', 'borgkdrvkap', 'borgkdrvnahh', 'cagkvborgkdr', 'cagkvcagk', 'cagkvcal', \
        'cagkvcan', 'cagkvcat', 'cagkvhd', 'cagkvkad', 'cagkvkahp', 'cagkvkap', 'cagkvnahh', 'calvborgkdr', \
        'calvcagk', 'calvcal', 'calvcan', 'calvcat', 'calvhd', 'calvkad', 'calvkahp', 'calvkap', 'calvnahh', \
        'canvborgkdr', 'canvcagk', 'canvcal', 'canvcan', 'canvcat', 'canvhd', 'canvkad', 'canvkahp', 'canvkap', \
        'canvnahh', 'catvborgkdr', 'catvcagk', 'catvcal', 'catvcan', 'catvcat', 'catvhd', 'catvkad', 'catvkahp', \
        'catvkap', 'catvnahh', 'hdvborgkdr', 'hdvcagk', 'hdvcal', 'hdvcan', 'hdvcat', 'hdvhd', 'hdvkad', 'hdvkahp', \
        'hdvkap', 'hdvnahh', 'kadvborgkdr', 'kadvcagk', 'kadvcal', 'kadvcan', 'kadvcat', 'kadvhd', 'kadvkad', 'kadvkahp', \
        'kadvkap', 'kadvnahh', 'kahpvborgkdr', 'kahpvcagk', 'kahpvcal', 'kahpvcan', 'kahpvcat', 'kahpvhd', 'kahpvkad', \
        'kahpvkahp', 'kahpvkap', 'kahpvnahh', 'kapvborgkdr', 'kapvcagk', 'kapvcal', 'kapvcan', 'kapvcat', 'kapvhd', 'kapvkad', \
        'kapvkahp', 'kapvkap', 'kapvnahh', 'nahhvborgkdr', 'nahhvcagk', 'nahhvcal', 'nahhvcan', 'nahhvcat', 'nahhvhd', 'nahhvkad', \
        'nahhvkahp', 'nahhvkap', 'nahhvnahh']
conds = ['catvkahp', 'catvnahh']
ts = np.linspace(0,1000,40001)
for cond in conds:
    for i in range(20):
        for j in range(20):
            for k in range(20):
                SAVEFILE[cond][i,j,k] = GetProp(ts, DATAFILE[cond][i,j,k])

SAVEFILE.close()
DATAFILE.close()