import numpy as np
import random
from scipy.stats import cauchy

def optimize(func, lb, ub, ndim, max_evaluations=10000, initial_pop_size=100, archive_size=100):
    """
    NL-SHADE-RSP-MID: Non-Linear Success-History based Adaptive Differential Evolution
    with Random Scale Parameter and Midpoint-based Crossover
    
    Parameters:
    - func: The CEC2022 function object to minimize
    - lb: Lower bounds of the search space
    - ub: Upper bounds of the search space
    - ndim: Number of dimensions
    - max_evaluations: Maximum number of function evaluations
    - initial_pop_size: Initial population size
    - archive_size: Size of the external archive
    
    Returns:
    - best_solution: Best found solution
    - best_fitness: Fitness of the best solution
    """
    # Initialize parameters
    pop_size = initial_pop_size
    min_pop_size = 4
    memory_size = 5
    
    # Initialize memory for CR and F
    memory_cr = np.ones(memory_size) * 0.5
    memory_f = np.ones(memory_size) * 0.5
    memory_pos = 0
    
    # Initialize population
    population = np.random.uniform(lb, ub, (pop_size, ndim))
    fitness = np.array([func.evaluate(x) for x in population])
    evaluations = pop_size
    
    # Initialize archive
    archive = []
    
    # Initialize best solution
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    # Main optimization loop
    while evaluations < max_evaluations:
        # Non-linear population size reduction
        new_pop_size = int(round(((min_pop_size - initial_pop_size) / max_evaluations) * evaluations + initial_pop_size))
        new_pop_size = max(new_pop_size, min_pop_size)
        
        # If population size decreased, remove worst individuals
        if new_pop_size < pop_size:
            sorted_indices = np.argsort(fitness)[:new_pop_size]
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            pop_size = new_pop_size
        
        # Initialize arrays for new generation
        new_population = np.empty_like(population)
        new_fitness = np.empty_like(fitness)
        
        # Generate CR and F for each individual
        cr = np.zeros(pop_size)
        f = np.zeros(pop_size)
        
        for i in range(pop_size):
            # Select random memory position
            r = random.randint(0, memory_size-1)
            
            # Generate CR from normal distribution
            cr[i] = np.clip(np.random.normal(memory_cr[r], 0.1), 0, 1)
            
            # Generate F from Cauchy distribution
            f[i] = np.clip(cauchy.rvs(loc=memory_f[r], scale=0.1), 0, 1)
            while f[i] <= 0:
                f[i] = np.clip(cauchy.rvs(loc=memory_f[r], scale=0.1), 0, 1)
        
        # Mutation and crossover
        for i in range(pop_size):
            # Current target vector
            x_i = population[i]
            
            # Select three distinct random vectors
            candidates = [idx for idx in range(pop_size) if idx != i]
            r1, r2, r3 = random.sample(candidates, 3)
            
            # Select additional vector from archive if not empty
            if len(archive) > 0 and random.random() < 0.5:
                r2 = random.choice(archive)  # Select random individual from archive
            
            # Mutation: current-to-pbest/1
            p_best_size = max(2, int(pop_size * 0.1))
            p_best_indices = np.argpartition(fitness, p_best_size)[:p_best_size]
            p_best_idx = int(random.choice(p_best_indices))  # Ensure integer index
            
            # Random scale parameter
            rsp = random.uniform(0.5, 1.5)
            
            # Mutation
            v_i = x_i + f[i] * rsp * (population[p_best_idx] - x_i) + f[i] * (population[r1] - population[r2])
            
            # Crossover: midpoint-based
            j_rand = random.randint(0, ndim-1)
            u_i = np.zeros(ndim)
            for j in range(ndim):
                if random.random() < cr[i] or j == j_rand:
                    # Midpoint crossover
                    u_i[j] = (x_i[j] + v_i[j]) / 2
                else:
                    u_i[j] = x_i[j]
            
            # Boundary handling
            u_i = np.clip(u_i, lb, ub)
            
            # Selection
            new_fitness_i = func.evaluate(u_i)
            evaluations += 1
            
            if new_fitness_i <= fitness[i]:
                new_population[i] = u_i
                new_fitness[i] = new_fitness_i
                
                # Update archive
                if len(archive) < archive_size:
                    archive.append(population[i].copy())
                elif len(archive) > 0:
                    # Replace random individual in archive
                    replace_idx = random.randint(0, len(archive)-1)
                    archive[replace_idx] = population[i].copy()
                
                # Update success memory
                memory_cr[memory_pos] = cr[i]
                memory_f[memory_pos] = f[i]
                memory_pos = (memory_pos + 1) % memory_size
            else:
                new_population[i] = population[i]
                new_fitness[i] = fitness[i]
            
            # Update best solution
            if new_fitness[i] < best_fitness:
                best_solution = new_population[i].copy()
                best_fitness = new_fitness[i]
            
            if evaluations >= max_evaluations:
                break
        
        # Update population
        population = new_population
        fitness = new_fitness
    
    return best_solution, best_fitness