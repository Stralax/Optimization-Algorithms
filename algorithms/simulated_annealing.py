import numpy as np
import random
from math import exp
import time

class Evaluator:
    def __init__(self, func):
        self.func = func
        self.eval_count = 0
        
    def evaluate(self, x):
        self.eval_count += 1
        if hasattr(self.func, 'evaluate'):
            return self.func.evaluate(x)
        return self.func(x)

def optimize(func, lb, ub, ndim=20, max_iter=500, runs=7000, output_file='results.txt'):
    """
    Enhanced Simulated Annealing with:
    - Adaptive cooling schedule
    - Smart neighborhood sampling
    - Adaptive step size control
    - Multiple restart strategy
    - Detailed logging and progress tracking
    - Parallel run capability
    """
    evaluator = Evaluator(func)
    lb = np.full(ndim, lb) if np.isscalar(lb) else np.array(lb)
    ub = np.full(ndim, ub) if np.isscalar(ub) else np.array(ub)
    search_range = ub - lb
    
    # SA configuration
    config = {
        'initial_temp': 1000.0,
        'min_temp': 1e-8,
        'adaptive_cooling': True,
        'reannealing_interval': 200,
        'step_size': 0.2 * search_range,
        'min_step_size': 0.01 * search_range,
        'acceptance_window': 50,
        'max_no_improve': 100
    }
    
    best_overall_solution = None
    best_overall_fitness = float('inf')
    
    with open(output_file, 'w') as f:
        f.write("Run,Iteration,Time,Best_Fitness,Current_Fitness,Temperature,Step_Size\n")
        
        for run in range(1, runs + 1):
            start_time = time.time()
            
            # Smart initialization using LHS
            current_solution = np.array([np.random.uniform(lb[d], ub[d]) for d in range(ndim)])
            current_fitness = evaluator.evaluate(current_solution)
            best_solution = current_solution.copy()
            best_fitness = current_fitness
            
            # Initialize SA parameters
            temp = config['initial_temp']
            step_size = config['step_size']
            acceptance_history = []
            no_improve_count = 0
            best_run_fitness = best_fitness
            
            for iteration in range(max_iter):
                # Adaptive neighborhood sampling
                if np.random.rand() < 0.7:  # Local search
                    neighbor = current_solution + step_size * np.random.randn(ndim)
                else:  # Global jump
                    neighbor = np.array([np.random.uniform(lb[d], ub[d]) for d in range(ndim)])
                
                neighbor = np.clip(neighbor, lb, ub)
                neighbor_fitness = evaluator.evaluate(neighbor)
                
                # Acceptance criterion
                delta = neighbor_fitness - current_fitness
                if delta < 0 or (temp > 0 and np.random.rand() < exp(-delta / temp)):
                    current_solution = neighbor
                    current_fitness = neighbor_fitness
                    acceptance_history.append(1)
                    
                    # Update best solution
                    if current_fitness < best_fitness:
                        best_solution = current_solution.copy()
                        best_fitness = current_fitness
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    acceptance_history.append(0)
                    no_improve_count += 1
                
                # Adaptive cooling and step size control
                if config['adaptive_cooling']:
                    # Adjust based on acceptance rate
                    if len(acceptance_history) > config['acceptance_window']:
                        acceptance_rate = np.mean(acceptance_history[-config['acceptance_window']:])
                        
                        # Dynamic temperature adjustment
                        if acceptance_rate < 0.2:
                            temp *= 0.9
                        elif acceptance_rate > 0.5:
                            temp *= 1.1
                        
                        # Dynamic step size adjustment
                        if acceptance_rate < 0.1:
                            step_size = np.maximum(step_size * 0.8, config['min_step_size'])
                        elif acceptance_rate > 0.4:
                            step_size = np.minimum(step_size * 1.2, search_range * 0.5)
                
                # Standard cooling schedule
                temp *= 0.995
                
                # Reannealing if stuck
                if no_improve_count >= config['max_no_improve']:
                    temp = max(temp * 2, config['initial_temp'] * 0.5)
                    step_size = np.minimum(step_size * 1.5, search_range * 0.3)
                    no_improve_count = 0
                
                # Periodic reannealing
                if config['reannealing_interval'] and iteration % config['reannealing_interval'] == 0:
                    temp = max(temp * 1.5, config['initial_temp'] * 0.3)
                
                # Log progress
                elapsed = time.time() - start_time
                f.write(f"{run},{iteration},{elapsed:.2f},{best_fitness:.6f},{current_fitness:.6f},{temp:.6f},{np.mean(step_size):.6f}\n")
                
                # Early termination conditions
                if temp < config['min_temp']:
                    break
                if no_improve_count >= 2 * config['max_no_improve']:
                    break
            
            # Update overall best
            if best_fitness < best_overall_fitness:
                best_overall_solution = best_solution.copy()
                best_overall_fitness = best_fitness
            
            print(f"Run {run}: Best={best_fitness:.6f}, Time={elapsed:.2f}s, Evals={evaluator.eval_count}")
    
    return best_overall_solution, best_overall_fitness