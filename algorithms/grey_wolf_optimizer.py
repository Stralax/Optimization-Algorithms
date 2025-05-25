import numpy as np

def optimize(func, lb, ub, ndim, pop_size=30, max_iter=1000):
    # Initialize population
    pop = np.random.uniform(lb, ub, (pop_size, ndim))
    fitness = np.array([func.evaluate(ind) for ind in pop])
    
    # Initialize alpha, beta, delta
    alpha_idx = np.argmin(fitness)
    alpha_pos = pop[alpha_idx].copy()
    alpha_score = fitness[alpha_idx]
    beta_idx = np.argsort(fitness)[1]
    beta_pos = pop[beta_idx].copy()
    beta_score = fitness[beta_idx]
    delta_idx = np.argsort(fitness)[2]
    delta_pos = pop[delta_idx].copy()
    delta_score = fitness[delta_idx]
    
    for t in range(max_iter):
        a = 2 * (1 - t / max_iter)  # Linearly decrease a from 2 to 0
        for i in range(pop_size):
            # Update position based on alpha, beta, delta
            r1, r2 = np.random.rand(2)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = np.abs(C1 * alpha_pos - pop[i])
            X1 = alpha_pos - A1 * D_alpha
            
            r1, r2 = np.random.rand(2)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = np.abs(C2 * beta_pos - pop[i])
            X2 = beta_pos - A2 * D_beta
            
            r1, r2 = np.random.rand(2)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = np.abs(C3 * delta_pos - pop[i])
            X3 = delta_pos - A3 * D_delta
            
            # New position
            pop[i] = (X1 + X2 + X3) / 3
            pop[i] = np.clip(pop[i], lb, ub)
            
            # Evaluate fitness
            fitness[i] = func.evaluate(pop[i])
            
            # Update alpha, beta, delta
            if fitness[i] < alpha_score:
                alpha_score = fitness[i]
                alpha_pos = pop[i].copy()
            elif fitness[i] < beta_score:
                beta_score = fitness[i]
                beta_pos = pop[i].copy()
            elif fitness[i] < delta_score:
                delta_score = fitness[i]
                delta_pos = pop[i].copy()
    
    return alpha_pos, alpha_score