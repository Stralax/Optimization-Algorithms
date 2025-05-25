import numpy as np

def optimize(func, lb, ub, ndim, pop_size=50, max_iter=1000, F=0.5, CR=0.9):
    # Initialize population
    pop = np.random.uniform(lb, ub, (pop_size, ndim))
    fitness = np.array([func.evaluate(ind) for ind in pop])
    
    # Track best solution
    best_idx = np.argmin(fitness)
    best_solution = pop[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    for _ in range(max_iter):
        for i in range(pop_size):
            # Mutation: Select three random distinct indices
            indices = [idx for idx in range(pop_size) if idx != i]
            a, b, c = pop[np.random.choice(indices, 3, replace=False)]
            mutant = a + F * (b - c)
            mutant = np.clip(mutant, lb, ub)
            
            # Crossover
            trial = np.copy(pop[i])
            j_rand = np.random.randint(0, ndim)
            for j in range(ndim):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = mutant[j]
            
            # Selection
            trial_fitness = func.evaluate(trial)
            if trial_fitness <= fitness[i]:
                pop[i] = trial
                fitness[i] = trial_fitness
                if trial_fitness < best_fitness:
                    best_solution = trial.copy()
                    best_fitness = trial_fitness
    
    return best_solution, best_fitness