import numpy as np

def optimize(func, lb, ub, ndim, pop_size=30, max_iter=1000):
    # Initialize population
    pop = np.random.uniform(lb, ub, (pop_size, ndim))
    fitness = np.array([func.evaluate(ind) for ind in pop])
    best_idx = np.argmin(fitness)
    best_solution = pop[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    for t in range(max_iter):
        a = 2 * (1 - t / max_iter)  # Linearly decrease a from 2 to 0
        for i in range(pop_size):
            r = np.random.rand()
            A = 2 * a * np.random.rand() - a
            C = 2 * np.random.rand()
            
            if np.random.rand() < 0.5:
                # Encircling prey
                if abs(A) < 1:
                    D = np.abs(C * best_solution - pop[i])
                    pop[i] = best_solution - A * D
                else:
                    # Search for prey (exploration)
                    rand_idx = np.random.randint(0, pop_size)
                    D = np.abs(C * pop[rand_idx] - pop[i])
                    pop[i] = pop[rand_idx] - A * D
            else:
                # Bubble-net attacking (spiral update)
                D = np.abs(best_solution - pop[i])
                l = np.random.uniform(-1, 1)
                pop[i] = D * np.exp(l) * np.cos(2 * np.pi * l) + best_solution
            
            pop[i] = np.clip(pop[i], lb, ub)
            fitness[i] = func.evaluate(pop[i])
            
            if fitness[i] < best_fitness:
                best_solution = pop[i].copy()
                best_fitness = fitness[i]
    
    return best_solution, best_fitness