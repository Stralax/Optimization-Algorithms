import numpy as np
from opfunu.cec_based.cec2022 import (
    F12022, F22022, F32022, F42022, F52022, F62022,
    F72022, F82022, F92022, F102022, F112022, F122022
)

class PSO:
    def __init__(self, obj_func, num_particles=50, max_evals=100000, w=0.7, c1=1.5, c2=1.5):
        self.func = obj_func
        self.D = obj_func.ndim
        self.lower = np.array(obj_func.lb)
        self.upper = np.array(obj_func.ub)
        self.num_particles = num_particles
        self.max_evals = max_evals
        self.evals = 0
        self.w = w      
        self.c1 = c1    
        self.c2 = c2    

        self.positions = np.random.uniform(self.lower, self.upper, (num_particles, self.D))
        self.velocities = np.random.uniform(-abs(self.upper - self.lower), abs(self.upper - self.lower), (num_particles, self.D))
        
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.array([obj_func.evaluate(p) for p in self.positions])
        self.evals += num_particles

        best_idx = np.argmin(self.personal_best_scores)
        self.global_best_position = self.personal_best_positions[best_idx]
        self.global_best_score = self.personal_best_scores[best_idx]

    def optimize(self):
        while self.evals < self.max_evals:
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.D), np.random.rand(self.D)
                
                cognitive = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social = self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower, self.upper)

                score = self.func.evaluate(self.positions[i])
                self.evals += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i].copy()
                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = self.positions[i].copy()
        return self.global_best_position, self.global_best_score


cec_functions = [F12022, F22022, F32022, F42022, F52022, F62022,
                 F72022, F82022, F92022, F102022, F112022, F122022]

results = {}
np.random.seed(42)
for i, FuncClass in enumerate(cec_functions, start=1):
    f = FuncClass(ndim=20)
    optimizer = PSO(f, num_particles=500, max_evals=1_000_000)
    best_sol, best_fit = optimizer.optimize()
    results[f"F{i}"] = {"fitness": best_fit, "coordinates": best_sol.tolist()}
    print(results)

print(results)
