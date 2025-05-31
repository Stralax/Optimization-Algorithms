"""
Izboljšan genetski algoritam za iskanje globalnega minimuma v 20D funkcijah
Optimiziran za CEC funkcije z boljšimi parametri in strategijami
"""

import numpy as np
import random

class GeneticAlgorithm:
    def __init__(self, population_size=200, max_generations=3000, 
                 crossover_rate=0.95, mutation_rate=0.05, elite_size=20,
                 tournament_size=5, F=0.5, CR=0.9):
        """
        Hibridni genetski algoritam z DE operatorji
        
        Parameters:
        - population_size: velikost populacije (povečana za boljše pokrivanje)
        - max_generations: maksimalno število generacij
        - crossover_rate: verjetnost križanja
        - mutation_rate: začetna verjetnost mutacije (zmanjšana)
        - elite_size: število najboljših (povečano)
        - tournament_size: velikost turnirja (povečana)
        - F: faktor mutacije za DE operatorje
        - CR: verjetnost križanja za DE
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.initial_mutation_rate = mutation_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.F = F
        self.CR = CR
        
        # Parametri za adaptacijo
        self.fitness_history = []
        self.diversity_history = []
        self.stagnation_counter = 0
        self.max_stagnation = 100
        
    def initialize_population(self, ndim, lb, ub):
        """Inicializacija populacije z različnimi strategijami"""
        population = []
        
        # 50% populacije - naključno
        for _ in range(self.population_size // 2):
            individual = np.random.uniform(lb, ub, ndim)
            population.append(individual)
        
        # 25% populacije - okoli centra
        center = (lb + ub) / 2
        std = (ub - lb) / 6  # Standard deviation
        for _ in range(self.population_size // 4):
            individual = np.random.normal(center, std)
            individual = np.clip(individual, lb, ub)
            population.append(individual)
        
        # 25% populacije - Latin Hypercube Sampling za boljše pokrivanje
        remaining = self.population_size - len(population)
        for i in range(remaining):
            individual = np.zeros(ndim)
            for j in range(ndim):
                # LHS pristop
                segment = (ub[j] - lb[j]) / remaining
                individual[j] = lb[j] + segment * (i + np.random.random())
            np.random.shuffle(individual)  # Premešaj dimenzije
            population.append(individual)
        
        return np.array(population)
    
    def evaluate_population(self, population, func):
        """Ocenjevanje celotne populacije z robustnim obravnavanjem napak"""
        fitness_values = []
        for individual in population:
            try:
                # Preveri, če je func CEC funkcija (ima evaluate metodo)
                if hasattr(func, 'evaluate'):
                    fitness = func.evaluate(individual)
                else:
                    fitness = func(individual)
                
                # Preveri veljavnost
                if np.isnan(fitness) or np.isinf(fitness):
                    fitness = 1e12
                fitness_values.append(fitness)
            except Exception as e:
                fitness_values.append(1e12)
        return np.array(fitness_values)
    
    def calculate_diversity(self, population):
        """Izračunaj diverziteto populacije"""
        center = np.mean(population, axis=0)
        distances = [np.linalg.norm(ind - center) for ind in population]
        return np.mean(distances)
    
    def differential_evolution_mutation(self, population, fitness_values, target_idx):
        """DE/rand/1 mutacija"""
        # Izberi tri naključne različne posameznike
        candidates = list(range(len(population)))
        candidates.remove(target_idx)
        
        if len(candidates) < 3:
            return population[target_idx].copy()
        
        r1, r2, r3 = random.sample(candidates, 3)
        
        # DE mutacija: v = x_r1 + F * (x_r2 - x_r3)
        mutant = population[r1] + self.F * (population[r2] - population[r3])
        return mutant
    
    def adaptive_crossover(self, target, mutant, lb, ub):
        """Adaptivno križanje z DE strategijo"""
        ndim = len(target)
        trial = target.copy()
        
        # Zagotovi vsaj eno spremenjeno dimenzijo
        j_rand = random.randint(0, ndim - 1)
        
        for j in range(ndim):
            if random.random() < self.CR or j == j_rand:
                trial[j] = mutant[j]
        
        # Popravi meje
        trial = np.clip(trial, lb, ub)
        return trial
    
    def gaussian_mutation(self, individual, lb, ub, sigma=0.1):
        """Gaussova mutacija z adaptivno standardno deviacijo"""
        mutated = individual.copy()
        ndim = len(individual)
        
        for i in range(ndim):
            if random.random() < self.mutation_rate:
                # Adaptivna standardna deviacija glede na obseg
                adaptive_sigma = sigma * (ub[i] - lb[i])
                
                # Gaussova mutacija
                noise = np.random.normal(0, adaptive_sigma)
                mutated[i] = individual[i] + noise
                
                # Popravi meje z odbojom
                if mutated[i] < lb[i]:
                    mutated[i] = lb[i] + (lb[i] - mutated[i])
                elif mutated[i] > ub[i]:
                    mutated[i] = ub[i] - (mutated[i] - ub[i])
                
                # Končno omeji
                mutated[i] = np.clip(mutated[i], lb[i], ub[i])
        
        return mutated
    
    def tournament_selection(self, population, fitness_values, num_selected=None):
        """Izboljšana turnirska selekcija"""
        if num_selected is None:
            num_selected = self.population_size - self.elite_size
        
        selected = []
        for _ in range(num_selected):
            # Naključno izberi posameznikov za turnir
            tournament_indices = np.random.choice(
                len(population), self.tournament_size, replace=False
            )
            tournament_fitness = fitness_values[tournament_indices]
            
            # Izberi najboljšega iz turnirja
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
        return np.array(selected)
    
    def adapt_parameters(self, generation, best_fitness):
        """Adaptacija parametrov glede na napredek"""
        # Shrani trenutno najboljšo prilagojenost
        self.fitness_history.append(best_fitness)
        
        # Preveri stagnacijo
        if len(self.fitness_history) > 50:
            recent_improvement = abs(self.fitness_history[-50] - self.fitness_history[-1])
            if recent_improvement < 1e-10:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
        
        # Adaptacija parametrov pri stagnaciji
        if self.stagnation_counter > 20:
            self.F = min(0.9, self.F * 1.1)  # Povečaj mutacijski faktor
            self.mutation_rate = min(0.2, self.mutation_rate * 1.5)
            self.CR = max(0.1, self.CR * 0.9)  # Zmanjšaj križanje
        else:
            # Postopno zmanjšuj parametre
            progress = generation / self.max_generations
            self.F = 0.5 + 0.3 * (1 - progress)
            self.mutation_rate = self.initial_mutation_rate * (1 - progress * 0.8)
            self.CR = 0.9 - 0.4 * progress
    
    def local_search(self, individual, func, lb, ub, max_iter=20):
        """Enostavno lokalno iskanje za izboljšanje rešitev"""
        current = individual.copy()
        
        try:
            # Preveri, če je func CEC funkcija (ima evaluate metodo)
            if hasattr(func, 'evaluate'):
                current_fitness = func.evaluate(current)
            else:
                current_fitness = func(current)
        except:
            return individual, 1e12
        
        for _ in range(max_iter):
            # Naredi majhno spremembo
            candidate = current + np.random.normal(0, 0.01 * (ub - lb))
            candidate = np.clip(candidate, lb, ub)
            
            try:
                # Preveri, če je func CEC funkcija (ima evaluate metodo)
                if hasattr(func, 'evaluate'):
                    candidate_fitness = func.evaluate(candidate)
                else:
                    candidate_fitness = func(candidate)
                    
                if candidate_fitness < current_fitness:
                    current = candidate
                    current_fitness = candidate_fitness
            except:
                continue
        
        return current, current_fitness
    
    def optimize(self, func, lb, ub, ndim):
        """Glavna optimizacijska funkcija"""
        # Inicializacija
        population = self.initialize_population(ndim, lb, ub)
        best_fitness = float('inf')
        best_solution = None
        
        # Evaluacija začetne populacije
        fitness_values = self.evaluate_population(population, func)
        
        for generation in range(self.max_generations):
            # Posodobitev najboljše rešitve
            current_best_idx = np.argmin(fitness_values)
            current_best_fitness = fitness_values[current_best_idx]
            
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_solution = population[current_best_idx].copy()
            
            # Adaptacija parametrov
            self.adapt_parameters(generation, best_fitness)
            
            # Nova populacija
            new_population = []
            new_fitness = []
            
            # Elitizem - obdrži najboljše
            elite_indices = np.argsort(fitness_values)[:self.elite_size]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
                new_fitness.append(fitness_values[idx])
            
            # Generiraj nove posameznike z hibridnim pristopom
            for i in range(len(population)):
                if i < self.elite_size:
                    continue  # Elita je že dodana
                
                # 70% DE operacije, 30% klasični GA
                if random.random() < 0.7:
                    # Differential Evolution pristop
                    mutant = self.differential_evolution_mutation(population, fitness_values, i)
                    trial = self.adaptive_crossover(population[i], mutant, lb, ub)
                else:
                    # Klasični GA pristop
                    # Selekcija
                    if random.random() < 0.5:
                        parent1 = self.tournament_selection(population, fitness_values, 1)[0]
                    else:
                        parent1 = population[i].copy()
                    
                    # Mutacija
                    trial = self.gaussian_mutation(parent1, lb, ub)
                
                # Evaluacija
                try:
                    # Preveri, če je func CEC funkcija (ima evaluate metodo)
                    if hasattr(func, 'evaluate'):
                        trial_fitness = func.evaluate(trial)
                    else:
                        trial_fitness = func(trial)
                        
                    if np.isnan(trial_fitness) or np.isinf(trial_fitness):
                        trial_fitness = 1e12
                except:
                    trial_fitness = 1e12
                
                # Selekcija preživetja
                if trial_fitness <= fitness_values[i]:
                    new_population.append(trial)
                    new_fitness.append(trial_fitness)
                else:
                    new_population.append(population[i].copy())
                    new_fitness.append(fitness_values[i])
            
            # Posodobi populacijo
            population = np.array(new_population)
            fitness_values = np.array(new_fitness)
            
            # Lokalno iskanje na najboljših vsako 100 generacij
            if generation % 100 == 0 and generation > 0:
                best_idx = np.argmin(fitness_values)
                improved_sol, improved_fit = self.local_search(
                    population[best_idx], func, lb, ub
                )
                if improved_fit < fitness_values[best_idx]:
                    population[best_idx] = improved_sol
                    fitness_values[best_idx] = improved_fit
                    
                    if improved_fit < best_fitness:
                        best_fitness = improved_fit
                        best_solution = improved_sol.copy()
            
            # Zgodnji izhod
            if best_fitness < 1e-12:
                break
            
            # Restart pri preveliki stagnaciji
            if self.stagnation_counter > self.max_stagnation:
                print(f"  Restart at generation {generation} (stagnation)")
                # Obdrži le najboljših 10%
                keep_indices = np.argsort(fitness_values)[:self.population_size//10]
                new_pop = self.initialize_population(ndim, lb, ub)
                for i, idx in enumerate(keep_indices):
                    new_pop[i] = population[idx].copy()
                population = new_pop
                fitness_values = self.evaluate_population(population, func)
                self.stagnation_counter = 0
            
            # Izpis napredka
            if generation % 200 == 0:
                diversity = self.calculate_diversity(population)
                print(f"  Generation {generation}: Best = {best_fitness:.6e}, "
                      f"F = {self.F:.3f}, MR = {self.mutation_rate:.3f}, "
                      f"Div = {diversity:.3f}")
        
        return best_solution, best_fitness

def optimize(func, lb, ub, ndim):
    """
    Optimizacijska funkcija kompatibilna z glavno funkcijo
    
    Parameters:
    - func: ciljna funkcija za minimiziranje
    - lb: spodnje meje (lahko array ali scalar)
    - ub: zgornje meje (lahko array ali scalar)
    - ndim: število dimenzij
    
    Returns:
    - best_solution: najboljša rešitev
    - best_fitness: najboljša prilagojenost
    """
    # Pripravi meje
    if np.isscalar(lb):
        lb = np.full(ndim, lb)
    if np.isscalar(ub):
        ub = np.full(ndim, ub)
    
    # Inicializiraj hibridni genetski algoritam
    # Parametri so optimizirani za CEC funkcije
    ga = GeneticAlgorithm(
        population_size=300,      # Večja populacija za boljše iskanje
        max_generations=5000,     # Več generacij
        crossover_rate=0.95,      # Visoka verjetnost križanja
        mutation_rate=0.02,       # Nizka začetna mutacija
        elite_size=30,            # Več elitnih rešitev
        tournament_size=7,        # Večji turnir
        F=0.6,                    # DE mutacijski faktor
        CR=0.8                    # DE križanje
    )
    
    print(f"  Starting Genetic Algorithm optimization...")
    result_solution, result_fitness = ga.optimize(func, lb, ub, ndim)
    print(f"  Final result: {result_fitness:.6e}")
    
    return result_solution, result_fitness

# Test funkcija
if __name__ == "__main__":
    # Test z enostavno funkcijo
    def sphere_function(x):
        return np.sum(x**2)
    
    # Test optimizacije
    lb = np.full(20, -5.12)
    ub = np.full(20, 5.12)
    ndim = 20
    
    best_sol, best_fit = optimize(sphere_function, lb, ub, ndim)
    print(f"Test - Best fitness: {best_fit:.6e}")
    print(f"Test - Best solution norm: {np.linalg.norm(best_sol):.6e}")