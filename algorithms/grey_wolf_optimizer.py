import numpy as np
from scipy.spatial.distance import cdist
import random

def optimize(func, lb, ub, ndim=20, pop_size=400, max_iter=3000):
    """
    Enhanced Grey Wolf Optimizer with L-SHADE features
    - Adaptive memory for CR and F parameters
    - Success-history based parameter adaptation
    - Linear population size reduction
    - Archive mechanism for better diversity
    - Improved mutation strategies
    """
    
    # Initialize bounds
    lb = np.full(ndim, lb) if np.isscalar(lb) else np.array(lb)
    ub = np.full(ndim, ub) if np.isscalar(ub) else np.array(ub)
    search_range = ub - lb
    
    # L-SHADE inspired parameters
    memory_size = 10
    archive = []
    m_cr = np.full(memory_size, 0.5)
    m_f = np.full(memory_size, 0.5)
    memory_index = 0
    
    # Enhanced initialization with LHS and opposition points
    pop = np.zeros((pop_size, ndim))
    for j in range(ndim):
        pop[:, j] = np.random.permutation(np.linspace(lb[j], ub[j], pop_size))
    pop += np.random.uniform(-0.1, 0.1, (pop_size, ndim)) * search_range
    
    # Opposition population (50% of pop_size)
    opp_pop = lb + ub - pop[:pop_size//2]
    pop = np.vstack((pop, opp_pop))
    pop = np.clip(pop, lb, ub)
    
    # Evaluate initial population
    fitness = np.array([func.evaluate(ind) for ind in pop])
    evals = len(pop)
    max_evals = max_iter * pop_size  # Convert iterations to evaluations
    
    # Initialize leaders
    sorted_idx = np.argsort(fitness)
    alpha_pos = pop[sorted_idx[0]].copy()
    alpha_score = fitness[sorted_idx[0]]
    beta_pos = pop[sorted_idx[1]].copy()
    beta_score = fitness[sorted_idx[1]]
    delta_pos = pop[sorted_idx[2]].copy()
    delta_score = fitness[sorted_idx[2]]
    
    # Adaptive parameters
    a_initial = 1.5 #2.0
    a_final = 0.01
    
    current_pop_size = pop_size
    
    while evals < max_evals and current_pop_size >= 4:
        progress = evals / max_evals
        
        # L-SHADE inspired linear population reduction
        new_pop_size = int(4 + (pop_size - 4) * (1 - progress))
        if new_pop_size < current_pop_size:
            # Sort and keep best individuals
            sorted_idx = np.argsort(fitness[:current_pop_size])
            pop = pop[sorted_idx]
            fitness = fitness[sorted_idx]
            current_pop_size = new_pop_size
        
        # Dynamic parameter adaptation
        if progress < 0.3:  # Exploration phase
            a = a_initial - (a_initial - 1.5) * (progress / 0.3)
            mutation_scale = 0.5
            local_search_prob = 0.1
        elif progress < 0.7:  # Transition phase
            a = 1.5 - (1.5 - 0.5) * ((progress-0.3)/0.4)
            mutation_scale = 0.3
            local_search_prob = 0.3
        else:  # Exploitation phase
            a = 0.5 - (0.5 - a_final) * ((progress-0.7)/0.3)
            mutation_scale = 0.1
            local_search_prob = 0.5
        
        # Corrected fitness-distance balance weights calculation
        leaders = np.vstack([alpha_pos, beta_pos, delta_pos])  # Shape (3, ndim)
        leader_scores = np.array([alpha_score, beta_score, delta_score])
        
        # Calculate distances (current_pop_size x 3)
        dists = cdist(pop[:current_pop_size], leaders)
        
        # Calculate weights properly
        with np.errstate(divide='ignore'):
            score_weights = 1 / (leader_scores + 1e-100)
            dist_weights = 1 / (dists + 1e-100)
        
        # Normalize weights for each wolf (current_pop_size x 3)
        weights = (score_weights[np.newaxis, :] * dist_weights)
        weights /= weights.sum(axis=1, keepdims=True)
        
        # L-SHADE inspired success tracking
        sf, scr, diffs = [], [], []
        
        for i in range(current_pop_size):
            if evals >= max_evals:
                break
                
            # L-SHADE parameter generation
            r = np.random.randint(0, memory_size)
            cr = np.clip(np.random.normal(m_cr[r], 0.1), 0, 1)
            f = 0
            while f <= 0:
                f = np.random.standard_cauchy() * 0.1 + m_f[r]
            f = np.clip(f, 0, 1)
            
            # Enhanced mutation strategy combining GWO and DE
            if np.random.rand() < 0.5:  # GWO update with adaptive weights
                r1, r2 = np.random.rand(2)
                A = 2 * a * r1 - a
                C = 2 * r2
                
                # Calculate D for all leaders at once (3 x ndim)
                D = np.abs(C * leaders - pop[i])
                
                # Calculate X for all leaders (3 x ndim)
                X = leaders - A * D
                
                # Weighted average using pre-calculated weights
                new_pos = np.sum(weights[i, :, np.newaxis] * X, axis=0)
                
                # Add Gaussian mutation
                mutation = mutation_scale * search_range * np.random.randn(ndim)
                new_pos += mutation * (1 - progress)
                
            else:  # L-SHADE inspired DE mutation
                indices = list(range(current_pop_size))
                indices.remove(i)
                
                if len(indices) >= 2:
                    a_idx, b_idx = np.random.choice(indices, 2, replace=False)
                    a_vec, b_vec = pop[a_idx], pop[b_idx]
                    
                    # Use archive if available
                    if archive and np.random.rand() < 0.5:
                        c_vec = archive[np.random.randint(len(archive))]
                    else:
                        c_idx = np.random.choice(indices)
                        c_vec = pop[c_idx]
                    
                    # DE/current-to-pbest/1 mutation
                    p = 0.1  # Top 10% for pbest
                    p_size = max(1, int(p * current_pop_size))
                    pbest_idx = np.random.choice(p_size)
                    pbest = pop[pbest_idx]
                    
                    new_pos = pop[i] + f * (pbest - pop[i]) + f * (a_vec - c_vec)
                else:
                    new_pos = pop[i] + mutation_scale * search_range * np.random.randn(ndim)
            
            new_pos = np.clip(new_pos, lb, ub)
            
            # L-SHADE inspired crossover
            cross_points = np.random.rand(ndim) < cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, ndim)] = True
            trial = np.where(cross_points, new_pos, pop[i])
            
            # Local search around best solution
            if np.random.rand() < local_search_prob:
                candidate = alpha_pos + 0.1*search_range*np.random.randn(ndim)
                candidate = np.clip(candidate, lb, ub)
                candidate_score = func.evaluate(candidate)
                evals += 1
                if candidate_score < alpha_score:
                    alpha_score = candidate_score
                    alpha_pos = candidate.copy()
                    pop[i] = candidate
                    fitness[i] = candidate_score
                    continue
            
            # Evaluate trial solution
            trial_score = func.evaluate(trial)
            evals += 1
            
            # Selection and success tracking
            if trial_score < fitness[i]:
                # Track successful parameters for L-SHADE memory update
                sf.append(f)
                scr.append(cr)
                diffs.append(fitness[i] - trial_score)
                
                # Add old solution to archive
                archive.append(pop[i].copy())
                
                # Update population
                pop[i] = trial
                fitness[i] = trial_score
            
            # Archive size management
            if len(archive) > current_pop_size:
                archive = random.sample(archive, current_pop_size)
        
        # L-SHADE memory update
        if diffs:
            diffs = np.array(diffs)
            weights_mem = diffs / np.sum(diffs)
            
            # Weighted Lehmer mean for CR
            if np.sum(weights_mem * np.array(scr)) > 0:
                m_cr[memory_index] = np.sum(weights_mem * (np.array(scr) ** 2)) / np.sum(weights_mem * np.array(scr))
            
            # Weighted Lehmer mean for F
            if np.sum(weights_mem * np.array(sf)) > 0:
                m_f[memory_index] = np.sum(weights_mem * (np.array(sf) ** 2)) / np.sum(weights_mem * np.array(sf))
            
            memory_index = (memory_index + 1) % memory_size
        
        # Update leaders
        best_idx = np.argmin(fitness[:current_pop_size])
        if fitness[best_idx] < alpha_score:
            delta_score = beta_score
            delta_pos = beta_pos.copy()
            beta_score = alpha_score
            beta_pos = alpha_pos.copy()
            alpha_score = fitness[best_idx]
            alpha_pos = pop[best_idx].copy()
        elif fitness[best_idx] < beta_score:
            delta_score = beta_score
            delta_pos = beta_pos.copy()
            beta_score = fitness[best_idx]
            beta_pos = pop[best_idx].copy()
        elif fitness[best_idx] < delta_score:
            delta_score = fitness[best_idx]
            delta_pos = pop[best_idx].copy()
        
        # Opposition-based restart (10% of population)
        if evals % (50 * current_pop_size) == 0 and progress > 0.3:
            n_restart = max(1, current_pop_size // 10)
            restart_indices = np.random.choice(current_pop_size, n_restart, replace=False)
            pop[restart_indices] = lb + ub - pop[restart_indices]
            pop[restart_indices] = np.clip(pop[restart_indices], lb, ub)
            for idx in restart_indices:
                if evals < max_evals:
                    fitness[idx] = func.evaluate(pop[idx])
                    evals += 1
    
    # Final intensive local search
    remaining_evals = max_evals - evals
    for _ in range(min(100, remaining_evals)):
        if evals >= max_evals:
            break
        candidate = alpha_pos + 0.01*search_range*np.random.randn(ndim)
        candidate = np.clip(candidate, lb, ub)
        candidate_score = func.evaluate(candidate)
        evals += 1
        if candidate_score < alpha_score:
            alpha_score = candidate_score
            alpha_pos = candidate.copy()
    
    return alpha_pos, alpha_score