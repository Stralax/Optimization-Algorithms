import numpy as np
from scipy.stats import levy
from sklearn.cluster import KMeans
from collections import deque

class EnhancedWOA:
    """
    Ultra-enhanced Whale Optimization Algorithm featuring:
    - Hybrid swarm intelligence from multiple algorithms (WOA, GWO, PSO)
    - Dynamic exploration-exploitation balance with multiple phases
    - Adaptive neighborhood search with memory
    - Hybrid local search strategies (Levy flights, pattern search)
    - Cluster-based diversity maintenance
    - Opposition-based learning
    - Fitness landscape adaptation
    - Specialized negative region handling
    """
    
    def __init__(self, func, lb, ub, ndim=20, pop_size=8000, max_iter=1500):
        self.func = func
        self.lb = np.full(ndim, lb) if np.isscalar(lb) else np.array(lb)
        self.ub = np.full(ndim, ub) if np.isscalar(ub) else np.array(ub)
        self.ndim = ndim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.search_range = self.ub - self.lb
        self.neg_mask = (self.lb < 0) & (self.ub > 0)  # Dimensions with negative values
        
        # Memory for adaptive parameters
        self.fitness_history = deque(maxlen=50)
        self.diversity_history = deque(maxlen=20)
        self.phase = 'exploration'  # Current optimization phase
        
        # Initialize population
        self.pop = self._smart_initialization()
        self.fitness = np.array([self.func(ind) for ind in self.pop])
        
        # Initialize best solution
        self.best_idx = np.argmin(self.fitness)
        self.best_solution = self.pop[self.best_idx].copy()
        self.best_fitness = self.fitness[self.best_idx]
        
        # Cluster information
        self.cluster_centers = None
        self.cluster_assignments = None
        
    def _smart_initialization(self):
        """Hybrid initialization combining LHS, opposition learning, and negative region focus"""
        pop = np.zeros((self.pop_size, self.ndim))
        
        # Strategy 1: Focused sampling in negative regions (50% of population)
        for j in range(self.ndim):
            if self.neg_mask[j] and np.random.rand() < 0.7:  # 70% chance for negative focus
                # Bimodal distribution with emphasis on negative region
                pop[:, j] = np.concatenate([
                    np.random.normal(-0.5*abs(self.ub[j]-self.lb[j]), 
                                    0.2*self.search_range[j], 
                                    self.pop_size//2),
                    np.random.uniform(self.lb[j], self.ub[j], 
                                    self.pop_size - self.pop_size//2)
                ])
            else:
                # Latin Hypercube Sampling for other dimensions
                pop[:, j] = self.lb[j] + (self.ub[j]-self.lb[j]) * \
                           np.random.permutation(np.linspace(0, 1, self.pop_size))
        
        pop = np.clip(pop, self.lb, self.ub)
        
        # Opposition learning for worst solutions
        fitness = np.array([self.func(ind) for ind in pop])
        worst_idx = np.argsort(fitness)[-self.pop_size//3:]
        opp_pop = self.lb + self.ub - pop[worst_idx]
        opp_fitness = np.array([self.func(ind) for ind in opp_pop])
        
        # Replace only if better
        improvement_mask = opp_fitness < fitness[worst_idx]
        pop[worst_idx[improvement_mask]] = opp_pop[improvement_mask]
        
        return np.clip(pop, self.lb, self.ub)
    
    def _update_clusters(self):
        """Maintain diversity using adaptive clustering"""
        n_clusters = min(5, self.pop_size//20)
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(self.pop)
        self.cluster_centers = kmeans.cluster_centers_
        self.cluster_assignments = kmeans.labels_
        
    def _calculate_diversity(self):
        """Measure population diversity"""
        return np.mean(np.std(self.pop, axis=0) / self.search_range)
    
    def _adaptive_parameters(self, t):
        """Dynamically adjust parameters based on search progress"""
        progress = t / self.max_iter
        
        # Phase transition logic
        if progress < 0.3:
            self.phase = 'exploration'
        elif progress < 0.7:
            self.phase = 'balanced'
        else:
            self.phase = 'exploitation'
        
        # Adaptive a parameter (from WOA)
        a = 2 * (1 - progress**2)  # Quadratic decay
        
        # Adaptive C parameter (from GWO)
        C = 1.5 + np.sin(np.pi*progress)  # Oscillating parameter
        
        # Social factor (from PSO)
        w = 0.9 - 0.5*progress  # Decreasing inertia weight
        
        return a, C, w
    
    def _movement_strategy(self, i, t, a, C, w):
        """Hybrid movement combining WOA, GWO and PSO strategies"""
        r = np.random.rand()
        current = self.pop[i]
        
        # Elite guidance probability increases with time
        if r < 0.4 + 0.4*(t/self.max_iter):  
            # Exploitation phase with multiple strategies
            if np.random.rand() < 0.6:
                # WOA bubble-net attacking (original WOA)
                if np.random.rand() < 0.5:
                    D = np.abs(self.best_solution - current)
                    self.pop[i] = self.best_solution + a * (np.random.rand(self.ndim) - 0.5) * D
                else:
                    # WOA spiral movement
                    distance = np.linalg.norm(self.best_solution - current)
                    theta = 2*np.pi*np.random.rand()
                    self.pop[i] = distance * np.exp(theta) * np.cos(theta) + self.best_solution
            else:
                # GWO hunting mechanism
                alpha = self.pop[np.argmin(self.fitness)]
                beta = self.pop[np.argsort(self.fitness)[1]]
                delta = self.pop[np.argsort(self.fitness)[2]]
                
                D_alpha = np.abs(C*alpha - current)
                D_beta = np.abs(C*beta - current)
                D_delta = np.abs(C*delta - current)
                
                X1 = alpha - a * D_alpha
                X2 = beta - a * D_beta
                X3 = delta - a * D_delta
                
                self.pop[i] = (X1 + X2 + X3) / 3
        else:
            # Exploration phase with directional mutations
            if np.random.rand() < 0.5:
                # PSO-inspired exploration with memory
                if not hasattr(self, 'velocity'):
                    self.velocity = np.zeros_like(self.pop)
                if not hasattr(self, 'personal_best'):
                    self.personal_best = self.pop.copy()
                    self.personal_best_fitness = self.fitness.copy()
                
                # Update personal best
                if self.fitness[i] < self.personal_best_fitness[i]:
                    self.personal_best[i] = self.pop[i].copy()
                    self.personal_best_fitness[i] = self.fitness[i]
                
                # Update velocity
                cognitive = 1.5 * np.random.rand() * (self.personal_best[i] - current)
                social = 1.5 * np.random.rand() * (self.best_solution - current)
                self.velocity[i] = w * self.velocity[i] + cognitive + social
                
                # Apply velocity
                self.pop[i] += self.velocity[i]
            else:
                # Levy flight exploration around cluster centers
                cluster_id = self.cluster_assignments[i]
                levy_step = levy.rvs(size=self.ndim, scale=0.01*self.search_range)
                self.pop[i] = self.cluster_centers[cluster_id] + 0.5*levy_step
        
        # Special handling for negative regions when best fitness is negative
        if self.best_fitness < 0:
            self.pop[i][self.neg_mask] *= (0.9 + 0.2*np.random.rand(np.sum(self.neg_mask)))
    
    def _boundary_handling(self, candidate):
        """Intelligent boundary constraint handling"""
        out_of_bounds = (candidate < self.lb) | (candidate > self.ub)
        
        if np.any(out_of_bounds):
            if np.random.rand() < 0.7:  # Reflect with probability
                candidate = np.where(candidate < self.lb, 
                                   2*self.lb - candidate, 
                                   candidate)
                candidate = np.where(candidate > self.ub, 
                                   2*self.ub - candidate, 
                                   candidate)
            else:  # Random reinitialization with probability
                candidate = np.where(out_of_bounds, 
                                   self.lb + (self.ub-self.lb)*np.random.rand(self.ndim), 
                                   candidate)
        
        return np.clip(candidate, self.lb, self.ub)
    
    def _local_refinement(self):
        """Hybrid local search combining pattern search and gradient information"""
        refined_solution = self.best_solution.copy()
        
        # Multi-directional pattern search
        for _ in range(20):
            candidate = refined_solution.copy()
            for j in range(self.ndim):
                if self.neg_mask[j] and self.best_fitness < 0:
                    # Special handling for negative regions
                    candidate[j] += 0.01 * self.search_range[j] * np.random.choice([-1, 1])
                else:
                    candidate[j] += 0.01 * self.search_range[j] * np.random.randn()
            
            candidate = self._boundary_handling(candidate)
            candidate_fitness = self.func(candidate)
            if candidate_fitness < self.best_fitness:
                refined_solution = candidate.copy()
        
        return refined_solution
    
    def optimize(self):
        """Main optimization loop"""
        convergence = np.zeros(self.max_iter)
        
        for t in range(self.max_iter):
            # Update adaptive parameters
            a, C, w = self._adaptive_parameters(t)
            
            # Cluster-based diversity maintenance (every 50 iterations)
            if t % 50 == 0 or self.cluster_centers is None:
                self._update_clusters()
            
            # Track diversity
            current_diversity = self._calculate_diversity()
            self.diversity_history.append(current_diversity)
            
            # Adaptive movement for each individual
            for i in range(self.pop_size):
                self._movement_strategy(i, t, a, C, w)
                self.pop[i] = self._boundary_handling(self.pop[i])
                
                # Evaluate and update
                new_fitness = self.func(self.pop[i])
                if new_fitness < self.fitness[i]:
                    self.fitness[i] = new_fitness
                    if new_fitness < self.best_fitness:
                        self.best_solution = self.pop[i].copy()
                        self.best_fitness = new_fitness
            
            convergence[t] = self.best_fitness
            self.fitness_history.append(self.best_fitness)
            
            # Phase-based local refinement
            if t % 100 == 0 or t == self.max_iter-1:
                refined = self._local_refinement()
                refined_fitness = self.func(refined)
                if refined_fitness < self.best_fitness:
                    self.best_solution = refined.copy()
                    self.best_fitness = refined_fitness
            
            # Early stopping condition
            if t > 200 and abs(np.mean(list(self.fitness_history)[-50:]) - self.best_fitness) < 1e-12:
                break
        
        # Final intensive search phase
        final_candidates = [
            self.best_solution + 0.01 * self.search_range * np.random.randn(self.ndim),
            self.best_solution * (0.9 + 0.2*np.random.rand(self.ndim)),
            np.where(self.neg_mask, 
                    self.best_solution - 0.05*abs(self.best_solution)*np.random.rand(self.ndim),
                    self.best_solution + 0.05*self.best_solution*np.random.rand(self.ndim))
        ]
        
        for candidate in final_candidates:
            candidate = self._boundary_handling(candidate)
            candidate_fitness = self.func(candidate)
            if candidate_fitness < self.best_fitness:
                self.best_solution = candidate.copy()
                self.best_fitness = candidate_fitness
        
        return self.best_solution, self.best_fitness, convergence[:t+1]


def optimize(cec_func, lb, ub, ndim=20, pop_size=400, max_iter=3000):
    """
    Wrapper specifically for CEC2022 functions
    """
    # Create a callable wrapper
    class FuncWrapper:
        def __init__(self, cec_func):
            self.cec_func = cec_func
            
        def __call__(self, x):
            return self.cec_func.evaluate(x)
    
    # Initialize the wrapper
    func = FuncWrapper(cec_func)
    
    # Create and run optimizer
    optimizer = EnhancedWOA(func, lb, ub, ndim, pop_size, max_iter)
    best_solution, best_fitness, _ = optimizer.optimize()
    
    return best_solution, best_fitness