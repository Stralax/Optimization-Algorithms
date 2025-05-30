import numpy as np
from opfunu.cec_based.cec2022 import (
    F12022, F22022, F32022, F42022, F52022, F62022,
    F72022, F82022, F92022, F102022, F112022, F122022
)
import random

# GLS-augmented L-SHADE
class GLS_LSHADE:
    def __init__(self, obj_func, max_evals=100000, pop_init=100, pop_min=4, memory_size=5, penalty_weight=0.1):
        self.func = obj_func
        self.D = self.func.ndim
        self.max_evals = max_evals
        self.pop_init = pop_init
        self.pop_min = pop_min
        self.memory_size = memory_size
        self.penalty_weight = penalty_weight
        
        self.lower = self.func.lb
        self.upper = self.func.ub

        self.archive = []
        self.m_cr = np.full(memory_size, 0.5)
        self.m_f = np.full(memory_size, 0.5)
        self.evals = 0

        # Penalty vector and feature usage count
        self.penalty = np.zeros(self.D)
        self.feature_usage = np.zeros(self.D)

    def penalized_fitness(self, solution):
        base = self.func.evaluate(solution)
        penalty_term = self.penalty_weight * np.sum(self.penalty * np.abs(solution))
        return base + penalty_term

    def update_penalty(self, solution):
        used_features = np.abs(solution) > 1e-5
        self.feature_usage[used_features] += 1
        max_usage = np.max(self.feature_usage)
        if max_usage > 0:
            self.penalty = self.feature_usage / max_usage

    def optimize(self):
        NP = self.pop_init
        population = np.random.uniform(self.lower, self.upper, (NP, self.D))
        fitness = np.array([self.penalized_fitness(sol) for sol in population])
        self.evals += NP
        best = population[np.argmin(fitness)]

        memory_index = 0

        while self.evals < self.max_evals and NP >= self.pop_min:
            sf, scr, diffs = [], [], []
            new_pop = []

            for i in range(NP):
                r = np.random.randint(0, self.memory_size)
                cr = np.clip(np.random.normal(self.m_cr[r], 0.1), 0, 1)
                f = 0
                while f <= 0:
                    f = np.random.standard_cauchy() * 0.1 + self.m_f[r]
                f = np.clip(f, 0, 1)

                indices = list(range(NP))
                indices.remove(i)
                a, b = population[np.random.choice(indices, 2, replace=False)]
                if self.archive:
                    c = self.archive[np.random.randint(len(self.archive))]
                else:
                    c = population[np.random.choice(indices)]

                mutant = population[i] + f * (a - population[i]) + f * (b - c)
                mutant = np.clip(mutant, self.lower, self.upper)

                trial = np.where(np.random.rand(self.D) < cr, mutant, population[i])
                trial_fit = self.penalized_fitness(trial)
                self.evals += 1

                if trial_fit < fitness[i]:
                    sf.append(f)
                    scr.append(cr)
                    diffs.append(fitness[i] - trial_fit)
                    new_pop.append(trial)
                    self.archive.append(population[i].copy())
                    fitness[i] = trial_fit
                    self.update_penalty(trial)
                else:
                    new_pop.append(population[i])

            if len(self.archive) > NP:
                self.archive = random.sample(self.archive, NP)

            if diffs:
                diffs = np.array(diffs)
                weights = diffs / np.sum(diffs)
                self.m_cr[memory_index] = np.sum(weights * (np.array(scr) ** 2)) / np.sum(weights * scr)
                self.m_f[memory_index] = np.sum(weights * (np.array(sf) ** 2)) / np.sum(weights * sf)
                memory_index = (memory_index + 1) % self.memory_size

            population = np.array(new_pop)
            NP_new = round(self.pop_min + (self.pop_init - self.pop_min) * (1 - self.evals / self.max_evals))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                population = population[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new

            best = population[np.argmin(fitness)]

        final_fit = self.func.evaluate(best)
        return best, final_fit


# Run GLS+LSHADE on all 12 CEC2022 functions
cec_functions = [
    F12022, F22022, F32022, F42022, F52022, F62022,
    F72022, F82022, F92022, F102022, F112022, F122022
]

results = {}
np.random.seed(42)

for i, FuncClass in enumerate(cec_functions, start=1):
    f = FuncClass(ndim=20)
    optimizer = GLS_LSHADE(f, max_evals=250000, pop_init=200, pop_min=4, memory_size=10, penalty_weight=0.1)
    best_sol, best_fit = optimizer.optimize()
    results[f"F{i}"] = {
        "fitness": best_fit,
        "coordinates": best_sol.tolist()
    }

print(results)
