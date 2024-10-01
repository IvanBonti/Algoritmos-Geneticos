import random
from deap import base, creator, tools, algorithms
import numpy as np

# Definir minimización
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Crear individuos
def create_individual():
    return [random.uniform(-5, 5) for _ in range(2)]

# Evaluar la función Sphere
def evaluate_sphere(individual):
    return (sum(x**2 for x in individual),)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_sphere)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Configuración del AGS
def genetic_algorithm():
    population = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    # Algoritmo evolutivo
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=50, 
                                              stats=stats, halloffame=hof, verbose=True)
    return hof[0]

# Ejecutar algoritmo
best_solution = genetic_algorithm()
print(f"Best solution: {best_solution}, Fitness: {evaluate_sphere(best_solution)}")
