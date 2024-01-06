import numpy as np
from neuron_models import *
from neuron.units import ms, mV
import pygad
import json

spike_type = 'FS_NoAdaptation'
#units: [v] = [v_inf] = mV, [tau] = ms, [g] = uS/mm2, [A] = mm2, [i_ext] = nA, [c] = nF/mm2

delay = 10
duration = 230

with open("./Izhikevich_Model_Params.json") as json_file:
    types_params = json.load(json_file)
iz = IzhikevichModel(types_params[spike_type])
iz.dur = duration
iz.delay = delay
iz.Initialize(-65*mV)
iz.ContinueRun(250*ms)

solution = np.zeros(3)
def fitness_func(ga_instance, solution, solution_idx):
  h = HodgkinHuxleyModel()
  h.dur = duration
  h.delay = delay
  h.amp = abs(solution[0])
  h.area = abs(solution[1])
  h.cm = abs(solution[2])
  h.Initialize(-65*mV)
  h.ContinueRun(250*ms)
  vs_hh = h.ys[:,0]
  fitness = 1.0/(0.001+np.sum(np.abs(vs_hh - iz.vs)))
  return fitness
    
num_generations = 1
num_parents_mating = 6

fitness_function = fitness_func

sol_per_pop = 10
num_genes = len(solution)

init_range_low = 0
init_range_high = 100

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "two_points"

mutation_type = "random"

def on_gen(ga_instance):
    ga_instance.best_solution()

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_by_replacement=True,
                       mutation_type=mutation_type,
                       mutation_num_genes=2,
                       random_mutation_min_val=0,
                       random_mutation_max_val=100,
                      )
print("|\tno. of generations\t|\tminimum fitness \t|")
print("-----------------------------------------------------------------")
ga_instance.run()
print("|\t\t{}\t\t|\t{:.2E}\t\t|".format(ga_instance.generations_completed, ga_instance.last_generation_fitness.min()))
while ga_instance.last_generation_fitness.min() < 0.1 :
  ga_instance.run()
  print("|\t\t{}\t\t|\t{:.2E}\t\t|".format(ga_instance.generations_completed, ga_instance.last_generation_fitness.min()))
  ga_instance.save(filename=spike_type)