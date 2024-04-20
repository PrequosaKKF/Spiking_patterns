from IdenTools import *
import h5py
'''
DATAFILE = h5py.File('/home/g6234203723/Spiking_patterns/src/NC_model/dataset/twoparam_prop.hdf5', 'r')
SAVEFILE = h5py.File('/home/g6234203723/Spiking_patterns/src/NC_model/dataset/twoparam_pattern.hdf5', 'w')
str_dtype = h5py.special_dtype(vlen=str)
for cond in DATAFILE.keys():
    SAVEFILE.create_dataset(cond, (100,100,1), dtype=str_dtype)
    for i in range(100):
        for j in range(100):    
            data = DATAFILE[cond][i,j]
            fsl = data[0]
            pss = data[1]
            swa = data[2]
            isis = data[3:]
            isis = isis[isis != 0]
            SAVEFILE[cond][i,j] = IdentifyPattern(fsl,swa,pss,isis)
        if (i%10 == 0):
            print("run for {} progressed by {}/100".format(cond, i+1))
'''
DATAFILE = h5py.File('/home/g6234203723/Spiking_patterns/src/NC_model/dataset/threeparams_prop.hdf5', 'r')
SAVEFILE = h5py.File('/home/g6234203723/Spiking_patterns/src/NC_model/dataset/threeparams_pattern.hdf5', 'w')
NUM = 20
str_dtype = h5py.special_dtype(vlen=str)
for cond in DATAFILE.keys():
    SAVEFILE.create_dataset(cond, (NUM,NUM,NUM,1), dtype=str_dtype)
    for i in range(NUM):
        for j in range(NUM):
            for k in range(NUM):
                data = DATAFILE[cond][i,j]
                fsl = data[0]
                pss = data[1]
                swa = data[2]
                isis = data[3:]
                isis = isis[isis != 0]
                SAVEFILE[cond][i,j] = IdentifyPattern(fsl,swa,pss,isis)
        print("run for {} progressed by {}/{}".format(cond, i+1, NUM))

DATAFILE.close()
SAVEFILE.close()