"""
RUN_RANDOM_FOREST.py
====================
Standalone runner to execute, evaluate, and demonstrate Random Forest alone.
Usage: python RUN_RANDOM_FOREST.py
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from RUN_INDIVIDUAL_MODEL import run_single_model

if __name__ == "__main__":
    run_single_model("random_forest")

