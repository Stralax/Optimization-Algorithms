# CeC 2022 Optimization Assignment

## Project Overview
This project addresses the optimization of the 12 CeC 2022 benchmark functions provided by the `opfunu` package, focusing on minimization in a 20-dimensional space with default bounds of [-100, 100] per dimension. The objective is to implement and evaluate six optimization algorithms, including two local search methods (Tabu Search and Guided Local Search) and four metaheuristic approaches (Differential Evolution, Ant Colony Optimization, Grey Wolf Optimizer, and Whale Optimization Algorithm), all coded from scratch without using pre-built optimizers. The goal is to find the minimum values for each function, with results saved in `.txt` files and a detailed report submitted in `.pdf` format by June 1, 2025.

## Optimization Algorithms
The following algorithms were implemented to solve the CeC 2022 functions:

1. **Differential Evolution (DE)**: A population-based algorithm that uses mutation, crossover, and selection to explore the search space, effective for continuous optimization problems.
2. **Ant Colony Optimization (ACO)**: Adapted for continuous domains, this algorithm mimics ant pheromone trails to construct solutions, balancing exploration and exploitation.
3. **Grey Wolf Optimizer (GWO)**: Inspired by the social hierarchy and hunting behavior of grey wolves, it uses leader-guided updates to optimize solutions.
4. **Tabu Search (TS)**: A local search method that employs a tabu list to avoid revisiting solutions, enhancing escape from local optima.
5. **Whale Optimization Algorithm (WOA)**: Models the bubble-net hunting strategy of humpback whales, combining encircling and spiral updates for global search.
6. **Guided Local Search (GLS)**: A local search method that uses penalties to escape local optima, guided by an augmented objective function.

## Features
- **Custom Implementations**: All algorithms are implemented from scratch, adhering to assignment requirements.
- **Command-Line Interface**: The main script allows selecting an algorithm via a command-line argument (`--algo 1-6`).
- **Robust Output**: Results are saved in algorithm-specific `.txt` files, including best fitness values and solutions for each function.
- **Error Handling**: Comprehensive error handling ensures robust execution across all functions and algorithms.

## Requirements
- Python 3.6+
- Libraries: `opfunu`, `numpy`
- Install dependencies:
  ```bash
  pip install opfunu numpy
