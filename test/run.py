# primer pokretanja:
# python3 run.py --algo 2

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
import best as best

# Parsaj argumente iz terminala
parser = argparse.ArgumentParser(description="Zaženi optimizacijski algoritem.")
parser.add_argument('--algo', type=int, choices=range(1, 7), required=True,
                    help='Izberi algoritem (1-6): 1=DE, 2=ACO, 3=GWO, 4=TS, 5=WOA, 6=GLS')
args = parser.parse_args()

def runDEF():

    # logic
    # run x times
    # with diferent parametars
    # if (result better than best.minimalne_vrednosti[x-1]), x represents Fx
    # save result and parameters

    print("DEF")

def runACO():
    print("ACO")

def runGWO():
    print("GWO")

def runTS():
    print("TS")

def runWOA():
    print("WOA")

def runGLS():
    print("GLS")


# Switch za izbiro algoritma
match args.algo:
    case 1:
        print("Izbran algoritem: Differential Evolution")
        runDEF()  # ali klic tvoje funkcije iz modula
    case 2:
        print("Izbran algoritem: Ant Colony Optimization")
        runACO()
    case 3:
        print("Izbran algoritem: Grey Wolf Optimizer")
        runGWO()
    case 4:
        print("Izbran algoritem: Tabu Search")
        runTS()
    case 5:
        print("Izbran algoritem: Whale Optimization Algorithm")
        runWOA()
    case 6:
        print("Izbran algoritem: Guided Local Search")
        runGLS()
    case _:
        print("Neveljavna izbira algoritma.")


