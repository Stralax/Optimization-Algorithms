import numpy as np

def optimize(func, lb, ub, ndim, n_ants=50, max_iter=1000, alpha=1.0, beta=2.0, rho=0.1):
    # Initialize pheromone trails
    pheromone = np.ones((ndim, 100)) * 0.1  # Discretized solution space
    tau_min, tau_max = 0.01, 1.0
    delta = (ub - lb) / 99
    
    # Initialize population
    pop = np.random.uniform(lb, ub, (n_ants, ndim))
    best_solution = pop[0].copy()
    best_fitness = func.evaluate(best_solution)
    
    for _ in range(max_iter):
        for ant in range(n_ants):
            # Construct solution
            solution = np.zeros(ndim)
            for d in range(ndim):
                probs = pheromone[d] ** alpha * (1.0 / (1e-10 + delta[d])) ** beta
                probs /= probs.sum()
                idx = np.random.choice(range(100), p=probs)
                solution[d] = lb[d] + idx * delta[d]
            
            # Evaluate
            fitness = func.evaluate(solution)
            if fitness < best_fitness:
                best_solution = solution.copy()
                best_fitness = fitness
            
            # Update pheromones
            idx = np.clip(((solution - lb) / delta).astype(int), 0, 99)
            for d in range(ndim):
                pheromone[d, idx[d]] += 1.0 / (1e-10 + fitness)
        
        # Evaporation
        pheromone = (1 - rho) * pheromone
        pheromone = np.clip(pheromone, tau_min, tau_max)
    
    return best_solution, best_fitness