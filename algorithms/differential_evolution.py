import numpy as np
from collections import deque

def optimize(func, lb, ub, ndim=20, pop_size=500, max_iter=5000):
    """
    Enhanced Differential Evolution with:
    - Robust direction vector handling
    - Advanced diversity preservation
    - Tabu search with fixed global radius
    - Earthquake mechanism
    """
    # Constants
    GLOBAL_TABU_RADIUS = 0.001 * np.mean(ub - lb)
    INITIAL_DIVERSITY_THRESHOLD = 2.5
    MIN_DIRECTION_NORM = 1e-10  # Minimum norm for direction vectors
    
    # Initialize population
    pop = np.random.uniform(lb, ub, (pop_size, ndim))
    fitness = np.array([func.evaluate(x) for x in pop])
    
    # Track best solution
    best_idx = np.argmin(fitness)
    best_solution = pop[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    # Optimization parameters
    F = 0.5
    CR = 0.9
    tabu_list = deque(maxlen=100)
    memory = deque(maxlen=500)
    memory.append(best_fitness)
    
    # Earthquake parameters
    no_improvement_counter = 0
    earthquake_threshold = 200
    earthquake_strength = 0.7
    
    # Diversity tracking
    diversity_threshold = INITIAL_DIVERSITY_THRESHOLD
    min_observed_diversity = np.mean(ub - lb)

    for generation in range(max_iter):
        current_diversity = np.std(fitness)
        spatial_diversity = np.mean(np.std(pop, axis=0))
        min_observed_diversity = min(min_observed_diversity, current_diversity)
        
        # Enhanced diversity condition
        diversity_alert = (
            (current_diversity < diversity_threshold) or
            (current_diversity < 0.1 * min_observed_diversity) or
            (spatial_diversity < 0.05 * np.mean(ub - lb))
        )
        
        if diversity_alert:
            print(f"Diversity alert at gen {generation}")
            
            # Save current best if improved
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness - 1e-4:
                best_solution = pop[current_best_idx].copy()
                best_fitness = fitness[current_best_idx]
                tabu_list.append(best_solution.copy())
            
            diversity_threshold = max(0.1, 0.9 * diversity_threshold)
            
            # Generate new points with safe direction handling
            for i in range(pop_size):
                if np.random.rand() < 0.8:
                    for attempt in range(5):
                        # Robust direction calculation
                        if len(tabu_list) > 0:
                            distances = [np.linalg.norm(pop[i] - tabu) for tabu in tabu_list]
                            nearest_tabu = tabu_list[np.argmin(distances)]
                            direction = pop[i] - nearest_tabu
                            norm = np.linalg.norm(direction)
                            if norm < MIN_DIRECTION_NORM:
                                direction = np.random.randn(ndim)
                                norm = np.linalg.norm(direction)
                        else:
                            direction = np.random.randn(ndim)
                            norm = np.linalg.norm(direction)
                        
                        if norm > MIN_DIRECTION_NORM:
                            direction /= norm
                        else:
                            direction = np.random.randn(ndim)
                            direction /= np.linalg.norm(direction)
                        
                        distance = (0.5 + 0.5*(1-current_diversity/min_observed_diversity)) * np.mean(ub-lb)
                        new_point = np.clip(pop[i] + distance * direction, lb, ub)
                        
                        # Tabu check
                        in_tabu = any(np.linalg.norm(new_point - tabu) < GLOBAL_TABU_RADIUS for tabu in tabu_list)
                        
                        if not in_tabu:
                            pop[i] = new_point
                            fitness[i] = func.evaluate(pop[i])
                            break
                    else:
                        pop[i] = np.random.uniform(lb, ub, ndim)
                        fitness[i] = func.evaluate(pop[i])
            
            min_observed_diversity = np.mean(ub - lb)
            continue
        
        # Standard DE operations
        for i in range(pop_size):
            # Mutation and crossover
            candidates = [idx for idx in range(pop_size) if idx != i]
            a, b, c = pop[np.random.choice(candidates, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), lb, ub)
            
            trial = np.copy(pop[i])
            cross_points = np.random.rand(ndim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, ndim)] = True
            trial[cross_points] = mutant[cross_points]
            
            # Tabu check
            in_tabu = any(np.linalg.norm(trial - tabu) < tabu_radius for tabu in tabu_list)
            
            if not in_tabu:
                trial_fitness = func.evaluate(trial)
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness - 1e-4:
                        best_solution = trial.copy()
                        best_fitness = trial_fitness
                        tabu_list.append(best_solution.copy())
                        no_improvement_counter = 0
                        memory.append(best_fitness)
        
        # Enhanced earthquake trigger (checks multiple conditions)
        if generation > earthquake_threshold:
            # Condition 1: No improvement in recent generations
            no_improvement = all(abs(memory[-1] - x) < 1e-4 for x in list(memory)[-20:])
            
            # Condition 2: Population clustering
            mean_distance = np.mean([np.linalg.norm(pop[i] - pop[j]) 
                                   for i in range(pop_size) 
                                   for j in range(i+1, min(i+10, pop_size))])
            clustered = mean_distance < 0.1 * np.mean(ub - lb)
            
            if no_improvement or clustered:
                no_improvement_counter += 1
            else:
                no_improvement_counter = max(0, no_improvement_counter - 2)
        
        if no_improvement_counter >= earthquake_threshold:
            print(f"Earthquake triggered at gen {generation} "
                  f"(Diversity: {current_diversity:.4f})")
            
            # Save current best to tabu list
            tabu_list.append(best_solution.copy())
            
            # Strong population reset (keep 10% elites)
            elite_size = pop_size // 10
            elite_indices = np.argsort(fitness)[:elite_size]
            reset_indices = np.random.choice(
                np.setdiff1d(np.arange(pop_size), elite_indices),
                int(pop_size * 0.8),  # Reset 80% of population
                replace=False
            )
            
            for idx in reset_indices:
                mutation = np.random.normal(0, earthquake_strength, ndim) * (ub - lb)
                pop[idx] = np.clip(pop[idx] + mutation, lb, ub)
                fitness[idx] = func.evaluate(pop[idx])
            
            # Adjust earthquake parameters
            no_improvement_counter = 0
            earthquake_threshold = min(500, earthquake_threshold + 100)
            earthquake_strength = max(0.2, earthquake_strength * 0.8)
        
        # Dynamic parameter adaptation
        if generation % 100 == 0:
            # Adjust F and CR based on diversity
            diversity_ratio = current_diversity / INITIAL_DIVERSITY_THRESHOLD
            F = np.clip(0.4 + 0.5 * diversity_ratio, 0.2, 0.9)
            CR = np.clip(0.7 + 0.2 * (1 - diversity_ratio), 0.5, 0.95)
        
        # Progress monitoring
        if generation % 100 == 0 or generation == max_iter-1:
            print(f"Gen {generation}: Best={best_fitness:.6f}, "
                  f"Div={current_diversity:.4f}, "
                  f"Tabu={len(tabu_list)}, "
                  f"F={F:.2f}, CR={CR:.2f}")
    
    return best_solution, best_fitness