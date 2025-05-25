import numpy as np
from collections import deque

def optimize(func, lb, ub, ndim, max_iter=1000, tabu_size=50, step_size=0.1):
    # Initialize solution
    current = np.random.uniform(lb, ub, ndim)
    current_fitness = func.evaluate(current)
    best_solution = current.copy()
    best_fitness = current_fitness
    
    # Initialize tabu list
    tabu_list = deque(maxlen=tabu_size)
    tabu_list.append(current.copy())
    
    for _ in range(max_iter):
        # Generate neighbors
        neighbors = []
        for _ in range(10):  # Generate 10 neighbors
            delta = np.random.normal(0, step_size * (ub - lb), ndim)
            neighbor = np.clip(current + delta, lb, ub)
            neighbors.append(neighbor)
        
        # Evaluate neighbors
        best_neighbor = None
        best_neighbor_fitness = float('inf')
        for neighbor in neighbors:
            if not any(np.allclose(neighbor, t, atol=1e-5) for t in tabu_list):
                fitness = func.evaluate(neighbor)
                if fitness < best_neighbor_fitness:
                    best_neighbor = neighbor
                    best_neighbor_fitness = fitness
        
        # Update current solution
        if best_neighbor is not None:
            current = best_neighbor
            current_fitness = best_neighbor_fitness
            tabu_list.append(current.copy())
            if current_fitness < best_fitness:
                best_solution = current.copy()
                best_fitness = current_fitness
    
    return best_solution, best_fitness