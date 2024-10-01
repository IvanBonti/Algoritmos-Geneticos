import random
import numpy as np
from deap import base, creator, tools

# Crear el tipo de Fitness (minimizar) y el tipo de Individuo
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimizar
creator.create("Individual", list, fitness=creator.FitnessMin)  # Individuo es una lista

# Configurar el Toolbox
toolbox = base.Toolbox()

# Función para generar un individuo (valores dentro de los límites de la función Schwefel)
def create_individual_schwefel():
    return [random.uniform(-500, 500) for _ in range(10)]  # Schwefel se evalúa en 10 dimensiones

# Registrar la función de creación de individuos en el toolbox
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual_schwefel)

# Registrar la población
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Registrar la función de evaluación (función Schwefel)
def schwefel(individual):
    return 4189.829101 * len(individual) - sum(x * np.sin(np.sqrt(abs(x))) for x in individual),

# Registrar la evaluación en el toolbox
toolbox.register("evaluate", schwefel)

# Operadores genéticos
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=100, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Configuración final (para un algoritmo evolutivo básico)
def main():
    random.seed(64)

    # Crear la población
    population = toolbox.population(n=100)

    # Configuración de parámetros evolutivos
    NGEN = 50   # Número de generaciones
    CXPB = 0.5  # Probabilidad de cruce
    MUTPB = 0.2 # Probabilidad de mutación

    # Evaluar la población inicial
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Evolución
    for gen in range(NGEN):
        # Seleccionar la siguiente generación de individuos
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Aplicar cruce y mutación
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluar individuos con fitness no calculado
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Reemplazar la población con la nueva generación
        population[:] = offspring

        # Imprimir la mejor solución de la generación
        fits = [ind.fitness.values[0] for ind in population]
        length = len(population)
        mean = sum(fits) / length
        best_ind = tools.selBest(population, 1)[0]
        print(f"Gen {gen}: Best individual is {best_ind}, {best_ind.fitness.values[0]}")

if __name__ == "__main__":
    main()
