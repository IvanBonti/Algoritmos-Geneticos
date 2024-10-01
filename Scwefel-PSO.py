import numpy as np
import random


# Definir la función Schwefel
def schwefel_function(x):
   return 4189.829101 * len(x) - sum(xi * np.sin(np.sqrt(np.abs(xi))) for xi in x)


# Clase Particle
class Particle:
   def __init__(self, dimension, search_space_min, search_space_max):
       self.position = [random.uniform(search_space_min, search_space_max) for _ in range(dimension)]
       self.velocity = [random.uniform(-1, 1) for _ in range(dimension)]
       self.best_position = self.position.copy()
       self.fitness = schwefel_function(self.position)
       self.best_fitness = self.fitness


# Función PSO
def pso(dimension, population_size, search_space_min, search_space_max, w, c1, c2, max_iterations):
   particles = [Particle(dimension, search_space_min, search_space_max) for _ in range(population_size)]
   global_best_particle = min(particles, key=lambda particle: particle.best_fitness)


   for _ in range(max_iterations):
       for particle in particles:
           for i in range(dimension):
               r1, r2 = random.random(), random.random()
               cognitive_term = c1 * r1 * (particle.best_position[i] - particle.position[i])
               social_term = c2 * r2 * (global_best_particle.best_position[i] - particle.position[i])
               particle.velocity[i] = w * particle.velocity[i] + cognitive_term + social_term
               particle.position[i] += particle.velocity[i]


               # Asegurar que la posición esté dentro del rango especificado
               particle.position[i] = max(search_space_min, min(search_space_max, particle.position[i]))


           # Actualizar la aptitud
           particle.fitness = schwefel_function(particle.position)


           # Actualizar el mejor personal
           if particle.fitness < particle.best_fitness:
               particle.best_fitness = particle.fitness
               particle.best_position = particle.position.copy()


       # Actualizar el mejor global
       global_best_particle = min(particles, key=lambda particle: particle.best_fitness)


   return global_best_particle.best_fitness, global_best_particle.best_position


# Parámetros PSO
dimension = 10
population_size = 50
search_space_min = -500
search_space_max = 500
w = 0.729
c1 = c2 = 1.494
max_iterations = 100


# Ejecutar PSO
best_fitness, best_position = pso(dimension, population_size, search_space_min, search_space_max, w, c1, c2, max_iterations)


# Imprimir resultados
print("Valor mínimo de la función Schwefel:")
print("Aptitud:", best_fitness)
print("Vector óptimo:", best_position)