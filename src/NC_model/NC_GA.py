import numpy as np
import neuron as nr
from neuron.units import ms, mV
import pygad
import os

spike_type = 'SB'
#units: [v] = [v_inf] = mV, [tau] = ms, [g] = uS/mm2, [A] = mm2, [i_ext] = nA, [c] = nF/mm2

delay = 10
duration = 230

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

vs_tg = np.fromfile("{}.csv".format(spike_type)).reshape((10,10001))

def fitness_func(ga_instance, solution, solution_idx):
  soma.gcatbar_cat=solution[0]
  iclamp.amp=solution[1]
  nr.h.finitialize(-65 * mV)
  nr.h.continuerun(250 * ms)
  vs_nc = v.as_numpy()
  fitness = 1e3-np.min(np.mean(np.abs(vs_nc - vs_tg), axis=1))
  return fitness
    
num_generations = 10
num_parents_mating = 60

fitness_function = fitness_func

sol_per_pop = 100
initial_population = np.abs(np.random.randn(100,2))
num_genes = 2

params_limit = {
   'gcatbar'  : np.linspace(0,2.5,5000).tolist(), #um
   'amp'      : np.linspace(0,50,5000).tolist() #um
}
gene_space = np.array(list(params_limit.values()))

parent_selection_type = "sus"
keep_parents = 32
keep_elitism = 16

crossover_type = "single_point"

mutation_type = "random"

def on_stop(ga_instance, last_population_fitness):
  print(ga_instance.best_solution())
  ga_instance.last_generation_parents.tofile('parents_{}.csv'.format(spike_type))
  ga_instance.last_generation_elitism.tofile('elites_{}.csv'.format(spike_type))

ga_instance = pygad.GA(num_generations=num_generations,
                      num_parents_mating=num_parents_mating,
                      fitness_func=fitness_function,
                      sol_per_pop=sol_per_pop,
                      random_mutation_min_val=0,
                      num_genes=num_genes,
                      initial_population=initial_population,
                      parent_selection_type=parent_selection_type,
                      keep_parents=keep_parents,
                      crossover_type=crossover_type,
                      mutation_type=mutation_type,
                      gene_space=gene_space,
                      on_stop=on_stop,
                      keep_elitism=keep_elitism,
                      parallel_processing=['process', 8],
                      allow_duplicate_genes=False
                      )
ga_instance.run()

if not os.path.exists('log.txt'):
  log_file = open('log.txt', 'a')
  log_file.write("|\tno. of generations\t|\tminimum fitness \t|\t50th percentile fitness\t|\tmaximum fitness \t|\n")
  log_file.write("-----------------------------------------------------------------------------------------------------------------------\n")
else:
  log_file = open('log.txt', 'a')

fitnesses = ga_instance.last_generation_fitness
log_file.write("|\t\t{}\t\t|\t{:.2E}\t\t|\t{:.2E}\t\t|\t{:.2E}\t\t|\n".format(ga_instance.generations_completed, fitnesses.min(), np.percentile(fitnesses, 50), fitnesses.max()))
log_file.close()

while ga_instance.last_generation_fitness.mean() < 995 :
  ga_instance.run()
  fitnesses = ga_instance.last_generation_fitness

  log_file = open('log.txt', 'a')
  log_file.write("|\t\t{}\t\t|\t{:.2E}\t\t|\t{:.2E}\t\t|\t{:.2E}\t\t|\n".format(ga_instance.generations_completed, fitnesses.min(), np.percentile(fitnesses, 50), fitnesses.max()))
  log_file.close()
print(ga_instance.last_generation_parents)
log_file = open('log.txt', 'a')
log_file.write("{}".format(ga_instance.last_generation_parents))
log_file.close()
ga_instance.plot_fitness()
  #ga_instance.save(filename='{}.pkl'.format(spike_type))