import g2o
import numpy as np

def optimize(frame, points, local_window, fix_points=False, verbose=False, iterations=50):
    """optimize using g2o

    :frame: TODO
    :points: TODO
    :local_window: TODO
    :fix_points: TODO
    :verbose: TODO
    :iterations: TODO
    :returns: TODO

    """
    if local_window is not None
        frames = frames[-local_window:]

    optimizer = g2o.SparseOptimizer()
    solver = g2o.

