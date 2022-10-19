import os
import sys

MIFF_PATH, _ = os.path.split(os.path.abspath(__file__))
PM_PATH, _ = os.path.split(MIFF_PATH)
sys.path.append(PM_PATH)