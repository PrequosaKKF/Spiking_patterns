import h5py
import numpy as np
from neuron.units import mV, ms
from IdenTools import *

DATAFILE = h5py.File('home/g6234203723/Spiking_patterns/src/NC_model/dataset/twoparams.hdf5','r')
SAVEFILE = h5py.File('home/g6234203723/Spiking_patterns/src/NC_model/dataset/twoparams_prop.hdf5','r+')

conds = ['borgkdr', 'cagk', 'cal', 'cat', 'can', 'hd', 'kad', 'kahp', 'kap', 'nahh']
ts = np.linspace(0,1000,40001)
for cond in conds:
    for i in range(100):
        for j in range(100):
            SAVEFILE[cond][i,j] = GetProp(ts, DATAFILE[cond][i,j])

'''
DATAFILE = h5py.File('home/g6234203723/Spiking_patterns/src/NC_model/dataset/threeparams.hdf5','r')
SAVEFILE = h5py.File('home/g6234203723/Spiking_patterns/src/NC_model/dataset/threeparams_prop.hdf5','r+')

conds = [['borgkdr_cagk',
 'borgkdr_cal',
 'borgkdr_cat',
 'borgkdr_can',
 'borgkdr_hd',
 'borgkdr_kad',
 'borgkdr_kahp',
 'borgkdr_kap',
 'borgkdr_nahh',
 'cagk_cal',
 'cagk_cat',
 'cagk_can',
 'cagk_hd',
 'cagk_kad',
 'cagk_kahp',
 'cagk_kap',
 'cagk_nahh',
 'cal_cat',
 'cal_can',
 'cal_hd',
 'cal_kad',
 'cal_kahp',
 'cal_kap',
 'cal_nahh',
 'cat_can',
 'cat_hd',
 'cat_kad',
 'cat_kahp',
 'cat_kap',
 'cat_nahh',
 'can_hd',
 'can_kad',
 'can_kahp',
 'can_kap',
 'can_nahh',
 'hd_kad',
 'hd_kahp',
 'hd_kap',
 'hd_nahh',
 'kad_kahp',
 'kad_kap',
 'kad_nahh',
 'kahp_kap',
 'kahp_nahh',
 'kap_nahh']]
ts = np.linspace(0,1000,40001)
for cond in conds:
    for i in range(20):
        for j in range(20):
            for k in range(20):
                SAVEFILE[cond][i,j,k] = GetProp(ts, DATAFILE[cond][i,j,k])
'''
SAVEFILE.close()
DATAFILE.close()