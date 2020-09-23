from pathlib import Path
import os
import numpy as np

cmaps = ["winter",
    "cool",
    "summer",
    "spring",
    "Purples",
    "YlOrBr"]


colors = ["blue",
    "green",
    "orange",
    "red",
    "purple",
    "yellow"]

markers = ['o', 'v', '^', '<', '>', 's', 'D', 'd']

surface_color = 'viridis'
surface_alpha_0 = 0.2
surface_alpha_1 = 0.4

marker_size_0 = 14

labelsize = 9


def f__file__(f):
    """format __file__ to appropriate base path"""
    return os.path.basename(f).split('.')[0]
