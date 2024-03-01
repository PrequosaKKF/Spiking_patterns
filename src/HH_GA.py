import numpy as np
from model import *
import neuron as nr
from neuron.units import ms, mV
import pygad
import json
import os

spike_type = 'FS_NoAdaptation'
#units: [v] = [v_inf] = mV, [tau] = ms, [g] = uS/mm2, [A] = mm2, [i_ext] = nA, [c] = nF/mm2

delay = 10
duration = 230

with open("src/Izhikevich_Model_Params.json") as json_file:
    types_params = json.load(json_file)
iz = IzhikevichModel(types_params[spike_type])
iz.dur = duration
iz.delay = delay
iz.Initialize(-65*mV)
iz.ContinueRun(250*ms)

soma = nr.h.Section(name="soma")
soma.insert('hh')
iclamp = nr.h.IClamp(soma(0.5))
iclamp.delay = delay
iclamp.dur = duration
nr.h.load_file("stdrun.hoc")
v = nr.h.Vector().record(soma(0.5)._ref_v)
t = nr.h.Vector().record(nr.h._ref_t)

vs_tg = np.fromfile("src/RS.csv")

def fitness_func(ga_instance, solution, solution_idx):
  soma.L              = solution[0]
  soma.diam           = solution[1]
  soma(0.5).hh.gkbar  = solution[2]
  soma(0.5).hh.gnabar = solution[3]
  soma(0.5).hh.gl     = solution[4]
  soma.cm             = solution[5]
  iclamp.amp          = solution[6]
  nr.h.finitialize(-65 * mV)
  nr.h.continuerun(250 * ms)
  vs_hh = v.as_numpy()
  fitness = 1e3-np.mean(np.abs(vs_hh - vs_tg))
  return fitness
    
num_generations = 2
num_parents_mating = 5

fitness_function = fitness_func

sol_per_pop = 10
solution = [20, 20, .036, .12, .0003, 1, 0]
initial_population = [solution for i in range(sol_per_pop)]
num_genes = len(solution)

params_limit = {
   'L'      : np.linspace(4,100,100).tolist(), #um
   'diam'   : np.linspace(5,140,100).tolist(), #um
   'gkbar'  : np.linspace(0,100,100).tolist(),
   'gnabar' : np.linspace(0,100,100).tolist(),
   'gl'     : np.linspace(0,.01,100).tolist(),
   'cm'     : np.linspace(0,10,100).tolist(),
   'amp'    : np.linspace(0,10,100).tolist(),
}
gene_space = np.array(list(params_limit.values()))

parent_selection_type = "sus"
keep_parents = 3
keep_elitism = 2

crossover_type = "single_point"

mutation_type = "random"

def on_gen(ga_instance):
  print(ga_instance.best_solution())

def on_stop(ga_instance, last_population_fitness):
  print(ga_instance.best_solution())
  ga_instance.last_generation_parents.tofile('parents.csv')
  print(ga_instance.last_generation_fitness)

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
                      parallel_processing=['process', 8]
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