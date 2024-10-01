import numpy as np
import random


# Define the sphere function
def sphere_function(x):
   return sum(xi**2 for xi in x)


# Particle class
class Particle:
   def __init__(self, dimension, search_space_min, search_space_max):
       self.position = [random.uniform(search_space_min, search_space_max) for _ in range(dimension)]
       self.position = [self.position[0], self.position[1]]  # Limit the vector to two values
       self.velocity = [random.uniform(-1, 1) for _ in range(dimension)]
       self.velocity = [self.velocity[0], self.velocity[1]]  # Limit the vector to two values
       self.best_position = self.position.copy()
       self.fitness = sphere_function(self.position)
       self.best_fitness = self.fitness


# PSO function
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


               # Ensure the position is within the specified range
               particle.position[i] = max(search_space_min, min(search_space_max, particle.position[i]))


           # Update fitness
           particle.fitness = sphere_function(particle.position)


           # Update personal best
           if particle.fitness < particle.best_fitness:
               particle.best_fitness = particle.fitness
               particle.best_position = particle.position.copy()


       # Update global best
       global_best_particle = min(particles, key=lambda particle: particle.best_fitness)


   return global_best_particle.best_fitness, global_best_particle.best_position


# PSO parameters
dimension = 2  # Set dimension to 2
population_size = 50
search_space_min = 1
search_space_max = 2
w = 0.729
c1 = c2 = 1.494
max_iterations = 100


# Run PSO
best_fitness, best_position = pso(dimension, population_size, search_space_min, search_space_max, w, c1, c2, max_iterations)


# Print results
print("Minimum value of the Sphere function:")
print("Fitness:", best_fitness)
print("Optimal vector:", best_position)
