import numpy as np
import argparse

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
algorithms_path = os.path.join(parent_dir, '.')
sys.path.append(algorithms_path)

from opfunu.cec_based import cec2022
from algorithms import differential_evolution as de
from algorithms import ant_colony_optimization as aco
from algorithms import grey_wolf_optimizer as gwo
from algorithms import tabu_search as ts
from algorithms import whale_optimization_algorithm as woa
from algorithms import guided_local_search as gls

print("ALO")