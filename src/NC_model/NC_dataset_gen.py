import numpy as np
import neuron as nr
from neuron.units import ms, mV
import pygad
import os
import sys
import h5py

#units: [v] = [v_inf] = mV, [tau] = ms, [g] = uS/mm2, [A] = mm2, [i_ext] = nA, [c] = nF/mm2
f = h5py.File('./dataset/threeparams.hdf5', 'r+')
delay = 200
duration = 600

v_init = -65 * mV

gna=0.025
gkdr=0.03
gkm=0.0001
gcal=0.0025
gcan=0.0028
gcat=.00025
gkahp=0.0003
gcagk=0.0008
gh=1e-5

Ek = -91
Ena = 50
Eca = 75

Ra = 200
cm = 0.75

soma = nr.h.Section(name="soma")
soma.Ra = Ra
soma.cm = cm
soma.diam=35
soma.L=20
soma.insert('pas')
soma.e_pas=v_init
soma.g_pas=1/6e4
soma.insert('cadifus')
soma.insert('can')
soma.gcanbar_can=gcan
soma.insert('cat')
soma.gcatbar_cat=gcat
soma.insert('kahp')
soma.gkahpbar_kahp=gkahp
soma.insert('cagk')
soma.gkbar_cagk=gcagk
soma.insert('nahh')
soma.gnabar_nahh=gna
soma.insert('borgkdr')
soma.gkdrbar_borgkdr=gkdr
soma.insert('hd')
soma.ghdbar_hd=gh*(1+1.75*soma.L/100)
soma.insert('cal')
soma.gcalbar_cal=gcal
soma.insert('kap')
soma.gbar_kap=1e-3*(7+11*soma.L/100)
soma.insert('borgkm')
soma.gkmbar_borgkm=gkm
soma.insert('kad')
soma.gbar_kad=1e-3*(7+11*soma.L/100)
soma.ek = Ek
soma.ena = Ena
soma.eca = Eca

iclamp = nr.h.IClamp(soma(0.5))
iclamp.delay = delay
iclamp.dur = duration
nr.h.load_file("stdrun.hoc")
v = nr.h.Vector().record(soma(0.5)._ref_v)
t = nr.h.Vector().record(nr.h._ref_t)

gbars = np.logspace(-4,0,20)
amps = np.logspace(-2,2,20)
ion_ch_list = ['cat','can','kahp','cagk','nahh','borgkdr','hd','cal','kap','kad']

for G in tuple(ion_ch_list):
    ion_ch_list.pop(0)
    for Gc in ion_ch_list:
        for i in range(len(gbars)):
            for j in range(len(gbars)):
                for k in range(len(amps)):
                    soma.gcanbar_can=gbars[i] if G == 'can' else gcan
                    soma.gkahpbar_kahp=gbars[i] if G == 'kahp' else gkahp
                    soma.gkbar_cagk=gbars[i] if G == 'cagk' else gcagk
                    soma.gnabar_nahh=gbars[i] if G == 'nahh' else gna
                    soma.gkdrbar_borgkdr=gbars[i] if G == 'borgkdr' else gkdr
                    soma.ghdbar_hd=gbars[i] if G == 'hd' else gh*(1+1.75*soma.L/100)
                    soma.gcalbar_cal=gbars[i] if G == 'cal' else gcal
                    soma.gbar_kap=gbars[i] if G == 'kap' else 1e-3*(7+11*soma.L/100)
                    soma.gbar_kad=gbars[i] if G == 'kad' else 1e-3*(7+11*soma.L/100)
                    
                    soma.gcanbar_can=gbars[j] if Gc == 'can' else gcan
                    soma.gkahpbar_kahp=gbars[j] if Gc == 'kahp' else gkahp
                    soma.gkbar_cagk=gbars[j] if Gc == 'cagk' else gcagk
                    soma.gnabar_nahh=gbars[j] if Gc == 'nahh' else gna
                    soma.gkdrbar_borgkdr=gbars[j] if Gc == 'borgkdr' else gkdr
                    soma.ghdbar_hd=gbars[j] if Gc == 'hd' else gh*(1+1.75*soma.L/100)
                    soma.gcalbar_cal=gbars[j] if Gc == 'cal' else gcal
                    soma.gbar_kap=gbars[j] if Gc == 'kap' else 1e-3*(7+11*soma.L/100)
                    soma.gbar_kad=gbars[j] if Gc == 'kad' else 1e-3*(7+11*soma.L/100)

                    iclamp.amp=amps[k]
                    nr.h.finitialize(-65 * mV)
                    nr.h.continuerun(1000 * ms)
                    f["{}v{}".format(G,Gc)][i,j,k] = v.as_numpy()
        print("run for {}v{} ion channel, prog:{}/100".format(G,Gc,i))
f.close()
