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

import numpy as np

from formationplanning.sqp import minimise_flux
from formationplanning.ls import solve_constraints as minimise_flux_wang
from formationplanning import ls
import formationplanning.plotting.plot_flux_minimisation_data_hemipshere_single_3_c as plotter
from formationplanning.plotting.plot_flux_minimisation_data_hemipshere_single_3_c import plot
from formationplanning.plotting.plot_flux_minimisation_data_hemipshere_single_3_c import plot_2d
from formationplanning.plotting import animate
from formationplanning import trajectory
from formationplanning.trajectory import map_to_time
from formationplanning.trajectory import interpolate_trajectory
from formationplanning.trajectory import preprocess, reparametrise
from formationplanning.dynamics import dynamics_sim

a_max = 5
v_max = 10

# locate a target swarm in a random location
np.random.seed(20)
mean = np.random.rand() * 200
mu, sigma = 200, 100
targets = np.random.normal(mu, sigma, (10, 3))

# initial drone separation
d = 5

# initial drone separation
d = 5
surface = np.array([
    [0, 0, 0],
    [0, d, 0],
    [0, d, d],
    [0, 0, d]])

# target formation sidelength
# target radius
r = 2.5

# target formation sidelength
l = r  * 2

directory = pathlib.Path('example_c_1')
x, phi_t = minimise_flux(targets, surface)
# time optimal path parameterisation

x = preprocess(x)
x_t, v_t, a_t, t = reparametrise(x, -v_max, v_max, -a_max, a_max)
#x_t, v_t, a_t, t = retime(x, [-10, 10], [-10, 10])
x_t = x_t.reshape((x_t.shape[0], 4, 3))
v_t = v_t.reshape((v_t.shape[0], 4, 3))
a_t = a_t.reshape((a_t.shape[0], 4, 3))

# follow the trajectory using 3d particle simulation with p controller on the applied force
dt = t[1] - t[0]
x1, xd1, xdd1 = dynamics_sim([0, 0, 0], x_t[:, 0, :], dt, u_mod=a_max) # drone 1
x2, xd2, xdd2 = dynamics_sim([0, 5, 0], x_t[:, 1, :], dt, u_mod=a_max) # drone 2
x3, xd3, xdd3 = dynamics_sim([0, 5, 5], x_t[:, 2, :], dt, u_mod=a_max) # drone 3
x4, xd4, xdd4 = dynamics_sim([0, 0, 5], x_t[:, 3, :], dt, u_mod=a_max) # drone 4

x_t = np.stack((x1, x2, x3, x4), axis=1)
xd_t = np.stack((xd1, xd2, xd3, xd4), axis=1)
xdd_t = np.stack((xdd1, xdd2, xdd3, xdd4), axis=1)

# plot the trajectories
plotter.colors = ['#009E73', '#56B4E9', '#E69F00', '#D55E00']
plotter.labels = ['Path', 'Simulation']
plotter.linestyles = ['--', '-']
plotter.skip_formation = 2000
plotter.elev = 10
plotter.azim = 180 - 40 + 10
plot([x, x_t], targets, 10, directory=directory, phi=phi_t)
plot_2d(xd_t, xdd_t, t, directory=directory)
#plot_2d(v_t, a_t, t)

animate.plot(x_t, targets, 10, phi=phi_t, directory=directory,
        cm=True, title='Figure 8', colors=plotter.colors, show_followers=False)