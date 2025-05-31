import numpy as np
import argparse
from opfunu.cec_based import cec2022
from algorithms import differential_evolution as de
from algorithms import ant_colony_optimization as aco
from algorithms import grey_wolf_optimizer as gwo
from algorithms import tabu_search as ts
from algorithms import whale_optimization_algorithm as woa
from algorithms import guided_local_search as gls
from algorithms import simulated_annealing as sa  # Add new module
from algorithms import nl_shade_rsp_mid
from algorithms import genetic_algorithm as ga

# List of CeC 2022 functions
functions = [
    cec2022.F12022, cec2022.F22022, cec2022.F32022,
    cec2022.F42022,
    cec2022.F52022,
    cec2022.F62022, cec2022.F72022, cec2022.F82022,
    cec2022.F92022, 
    cec2022.F102022,
    cec2022.F112022,
    cec2022.F122022
]

# List of optimization algorithms
algorithms = [
    ("Differential Evolution", de.optimize),
    ("Ant Colony Optimization", aco.optimize),
    ("Grey Wolf Optimizer", gwo.optimize),
    ("Tabu Search", ts.optimize),
    ("Whale Optimization Algorithm", woa.optimize),
    ("Guided Local Search", gls.optimize),
    ("Simulated Annealing", sa.optimize),  # Add Simulated Annealing
    ("NL-SHADE-RSP-MID", nl_shade_rsp_mid.optimize),
    ("Genetic Algorithm", ga.optimize)
]

def main(algo_index=None):
    # Determine which algorithms to run
    algo_indices = range(1, len(algorithms) + 1) if algo_index is None else [algo_index]
    
    for idx in algo_indices:
        # Validate algorithm index
        if not (1 <= idx <= len(algorithms)):
            print(f"Invalid algorithm index: {idx}. Skipping.")
            continue
        
        algo_name, algo_func = algorithms[idx - 1]

        # Open file to store results
        with open(f"optimization_results_{algo_name.replace(' ', '_')}.txt", "w") as result_file:
            result_file.write(f"Results for {algo_name}:\n")
            result_file.write("=" * 50 + "\n")

            # Iterate over each CeC 2022 function
            for f in functions:
                try:
                    # Initialize function with 20 dimensions
                    func = f(ndim=20)
                    lb = func.lb
                    ub = func.ub
                    ndim = 20

                    # Run the selected algorithm
                    try:
                        best_solution, best_fitness = algo_func(func, lb, ub, ndim)
                        result_file.write(f"\n{f.__name__}:\n")
                        result_file.write(f"  Best Fitness: {best_fitness:.6f}\n")
                        result_file.write(f"  Best Solution: {np.round(best_solution, 6)}\n")
                        print(f"{f.__name__} - {algo_name}: Best Fitness = {best_fitness:.6f}")
                    except Exception as e:
                        error_msg = f"Error in {algo_name} for {f.__name__}: {str(e)}"
                        result_file.write(error_msg + "\n")
                        print(error_msg)
                except Exception as e:
                    error_msg = f"Error initializing {f.__name__}: {str(e)}"
                    result_file.write(error_msg + "\n")
                    print(error_msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run optimization algorithm(s) on CeC 2022 functions.")
    parser.add_argument(
        "--algo",
        type=int,
        choices=range(1, 10),  # Update to include 8
        help="Algorithm index (1: DE, 2: ACO, 3: GWO, 4: TS, 5: WOA, 6: GLS, 7: SA, 8: NL-SHADE-RSP-MID). If not provided, runs all algorithms."
    )
    args = parser.parse_args()
    main(args.algo)