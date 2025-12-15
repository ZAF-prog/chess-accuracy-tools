
import sys
print("Starting imports...")
try:
    import multiprocessing
    print("multiprocessing ok")
    import chess
    print("chess ok")
    import chess.engine
    print("chess.engine ok")
    import chess.pgn
    print("chess.pgn ok")
    import numpy as np
    print("numpy ok")
    import math
    print("math ok")
    import csv
    print("csv ok")
    import argparse
    print("argparse ok")
    import pickle
    print("pickle ok")
    import time
    print("time ok")
    import os
    print("os ok")
    from pathlib import Path
    print("pathlib ok")
    from collections import defaultdict
    print("collections ok")
    from scipy.optimize import minimize
    print("scipy.optimize ok")
    try:
        import psutil
        print("psutil ok")
    except:
        print("psutil missing (caught)")
except Exception as e:
    print(f"CRASH during import: {e}")
    import traceback
    traceback.print_exc()
print("Imports done.")
