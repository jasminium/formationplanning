"""
FormationPlanning
Copyright (C) 2020 : Northumbria University
            Author : John Hartley

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import pathlib
import sys

import numpy as np

from formationplanning.sqp import minimise_flux
from formationplanning.ls import solve_constraints as minimise_flux_wang
from formationplanning import ls
from formationplanning.plotting.plot_flux_minimisation_data_hemipshere_single_comparison import plot
from formationplanning.plotting import plot_flux_minimisation_data_hemipshere_single_comparison as plotter
from formationplanning.plotting import animate
from formationplanning import trajectory
from formationplanning.trajectory import map_to_time
from formationplanning.trajectory import interpolate_trajectory
from formationplanning.trajectory import path_length

# Target to track
target = np.array((50, 50, 50))
targets = target.reshape((1, 3))

# initial drone separation
d = 5
surface = np.array([
    [0, 0, 0],
    [0, d, 0],
    [0, d, d],
    [0, 0, d]])

# target radius
r = 2.5

# target formation sidelength
l = r * 2

# title for the animation
ani_title = 'Figure 2'
# output directory
directory = pathlib.Path('example_a_1')
# evaluate using the flux guided approach
x_t, phi_t = minimise_flux(target, surface, l)
ls.xtol = 7
ls.beta = 0
# evaluate using LS approach
x_t_wang, phi_t = minimise_flux_wang(target, surface, l)

ls.xtol = 1
# increase the weighting of the regularisation term
ls.beta = 400
# evaluate using LS approach
x_t_wang_con, phi_t = minimise_flux_wang(target, surface, l)

# evalute the total length of the trajectories
pl_w = path_length(x_t_wang)
pl_wc = path_length(x_t_wang_con)
pl_fg = path_length(x_t)

print('path length, beta=0,', pl_w)
print('path length, beta=400,', pl_wc)
print('path length, FG,', pl_fg)

# plot the trajectories
colors = ['#009E73', '#56B4E9', '#E69F00']
plotter.colors = colors
plotter.labels = ['FG', r'LS $\beta=0$', r'LS $\beta$=400']
tr = [x_t[::10], x_t_wang, x_t_wang_con]
plot(tr, target, 1, directory=directory, phi=phi_t)

# section 1 b
target = np.array((-50, 50, 50))
directory = pathlib.Path('example_b_1')
x_t, phi_t = minimise_flux(target, surface, l)
ls.xtol = 7
ls.beta = 0
x_t_wang, phi_t = minimise_flux_wang(target, surface, l)

ls.xtol = 1
ls.beta = 400
x_t_wang_con, phi_t = minimise_flux_wang(target, surface, l)

pl_w = path_length(x_t_wang)
pl_wc = path_length(x_t_wang_con)
pl_fg = path_length(x_t)

print('path length, beta=0,', pl_w)
print('path length, beta=400,', pl_wc)
print('path length, FG,', pl_fg)

# plot the trajectories
plot([x_t[::10], x_t_wang, x_t_wang_con], target, 1, directory=directory, phi=phi_t)