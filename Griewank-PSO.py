import numpy as np
import random


# Parámetros del PSO
num_particles = 250
max_iterations = 2000
inertia_weight = 0.729
cognitive_parameter = 1.494
social_parameter = 1.494


# Parámetros del problema
dimension = 10
search_space_min = -600
search_space_max = 600


# Fitness function (Griewank function)
def fitness(vector):
   sum_part = sum(x**2 / 4000 for x in vector)
   prod_part = np.prod([np.cos(x / np.sqrt(i + 1)) for i, x in enumerate(vector)])
   return sum_part - prod_part + 1


# Inicialización de partículas
particles = [{'position': [random.uniform(search_space_min, search_space_max) for _ in range(dimension)],
             'velocity': [random.uniform(-1, 1) for _ in range(dimension)],
             'best_position': [random.uniform(search_space_min, search_space_max) for _ in range(dimension)],
             'best_fitness': float('inf')} for _ in range(num_particles)]


# Mejor posición global
global_best_position = [random.uniform(search_space_min, search_space_max) for _ in range(dimension)]
global_best_fitness = float('inf')


# Función de actualización de partículas
def update_particles():
   global global_best_position, global_best_fitness


   for particle in particles:
       # Actualización de posición y velocidad
       for i in range(dimension):
           inertia_term = inertia_weight * particle['velocity'][i]
           cognitive_term = cognitive_parameter * random.random() * (particle['best_position'][i] - particle['position'][i])
           social_term = social_parameter * random.random() * (global_best_position[i] - particle['position'][i])


           particle['velocity'][i] = inertia_term + cognitive_term + social_term
           particle['position'][i] += particle['velocity'][i]


       # Evaluación del fitness
       current_fitness = fitness(particle['position'])


       # Actualización del mejor conocido de la partícula
       if current_fitness < particle['best_fitness']:
           particle['best_fitness'] = current_fitness
           particle['best_position'] = particle['position']


       # Actualización del mejor global si es necesario
       if current_fitness < global_best_fitness:
           global_best_fitness = current_fitness
           global_best_position = particle['position']


# Ejecución del PSO
for iteration in range(max_iterations):
   update_particles()


# Resultados del PSO
print("Resultados del PSO:")
print("Fitness:", global_best_fitness)
print("Optimal vector:", global_best_position)
