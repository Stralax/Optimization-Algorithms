import numpy as np

def optimize(func, lb, ub, ndim, max_iter=10000, step_size=0.001, lambda_=0.1):
    # Initialize solution and penalties
    current = np.random.uniform(lb, ub, ndim)
    penalties = np.zeros(ndim)
    current_fitness = func.evaluate(current)
    best_solution = current.copy()
    best_fitness = current_fitness
    
    def augmented_fitness(x):
        return func.evaluate(x) + lambda_ * np.sum(penalties * (x - lb) / (ub - lb))
    
    for _ in range(max_iter):
        # Generate neighbors
        neighbors = []
        for _ in range(10):
            delta = np.random.normal(0, step_size * (ub - lb), ndim)
            neighbor = np.clip(current + delta, lb, ub)
            neighbors.append(neighbor)
        
        # Select best neighbor based on augmented fitness
        best_neighbor = None
        best_aug_fitness = float('inf')
        for neighbor in neighbors:
            aug_fitness = augmented_fitness(neighbor)
            if aug_fitness < best_aug_fitness:
                best_neighbor = neighbor
                best_aug_fitness = aug_fitness
        
        if best_neighbor is not None:
            current = best_neighbor
            current_fitness = func.evaluate(current)
            if current_fitness < best_fitness:
                best_solution = current.copy()
                best_fitness = current_fitness
            
            # Update penalties for features with high utility
            utilities = [(i, abs(current[i] - lb[i]) / (ub[i] - lb[i]) / (1 + penalties[i])) 
                         for i in range(ndim)]
            max_util_idx = max(utilities, key=lambda x: x[1])[0]
            penalties[max_util_idx] += 1
    
    return best_solution, best_fitness