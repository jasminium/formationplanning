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
import formationplanning.plotting.plot_flux_minimisation_data_hemipshere_single_3_d as plotter
from formationplanning.plotting.plot_flux_minimisation_data_hemipshere_single_3_d import plot
from formationplanning.plotting.plot_flux_minimisation_data_hemipshere_single_3_d import plot_2d
from formationplanning.plotting import animate
from formationplanning import trajectory
from formationplanning.trajectory import map_to_time
from formationplanning.trajectory import interpolate_trajectory
from formationplanning.trajectory import preprocess, reparametrise
from formationplanning.trajectory import generate_follower_path
from formationplanning.dynamics import dynamics_sim

a_max = 10
v_max = 10

target = np.array((10, 10, 10))
targets = target.reshape(1, 3)

# initial drone separation
d = 2
surface = np.array([
    [0, 0, 0],
    [0, d, 0],
    [0, d, d],
    [0, 0, d]])

# target formation sidelength
# target radius
r = 3

# target formation sidelength
l = r  * 2

directory = pathlib.Path('example_e_1')
x, phi_t = minimise_flux(target, surface, l)
# time optimal path parameterisation
f = generate_follower_path(x)

x = np.concatenate((x, f), axis=1)

# leaders
x = preprocess(x)
x_t, v_t, a_t, t = reparametrise(x, -v_max, v_max, -a_max, a_max)
x_t = x_t.reshape((x_t.shape[0], 9, 3))
v_t = v_t.reshape((v_t.shape[0], 9, 3))
a_t = a_t.reshape((a_t.shape[0], 9, 3))

# follow the trajectory using 3d particle simulation with p controller on the applied force
dt = t[1] - t[0]
x1, xd1, xdd1 = dynamics_sim([0, 0, 0], x_t[:, 0, :], dt, u_mod=a_max) # drone 1
x2, xd2, xdd2 = dynamics_sim([0, 5, 0], x_t[:, 1, :], dt, u_mod=a_max) # drone 2
x3, xd3, xdd3 = dynamics_sim([0, 5, 5], x_t[:, 2, :], dt, u_mod=a_max) # drone 3
x4, xd4, xdd4 = dynamics_sim([0, 0, 5], x_t[:, 3, :], dt, u_mod=a_max) # drone 4

f1, fd1, fdd1 = dynamics_sim(x_t[0, 4, :], x_t[:, 4, :], dt, u_mod=a_max) # drone 5
f2, fd2, fdd2 = dynamics_sim(x_t[0, 5, :], x_t[:, 5, :], dt, u_mod=a_max) # drone 6
f3, fd3, fdd3 = dynamics_sim(x_t[0, 6, :], x_t[:, 6, :], dt, u_mod=a_max) # drone 7
f4, fd4, fdd4 = dynamics_sim(x_t[0, 7, :], x_t[:, 7, :], dt, u_mod=a_max) # drone 8
f5, fd5, fdd5 = dynamics_sim(x_t[0, 8, :], x_t[:, 8, :], dt, u_mod=a_max) # drone 9

x_t = np.stack((x1, x2, x3, x4, f1, f2, f3, f4, f5), axis=1)
xd_t = np.stack((xd1, xd2, xd3, xd4, fd1, fd2, fd3, fd4, fd5), axis=1)
xdd_t = np.stack((xdd1, xdd2, xdd3, xdd4, fdd1, fdd2, fdd3, fdd4, fdd5), axis=1)

# plot the trajectories
plotter.colors = ['#009E73', '#56B4E9', '#E69F00', '#D55E00']
plotter.labels = ['Path', 'Simulation']
plotter.linestyles = ['--', '-']
plotter.formation_slices = [0, -1, 300, 230]
plotter.skip_formation = 200
plotter.elev = 15
plotter.azim = 180 + 90  + 10
plot([x, x_t], target, r, directory=directory, phi=phi_t)
plot_2d(xd_t, xdd_t, t, directory=directory)

animate.plot(x_t, targets, r, phi=phi_t, directory=directory,
        cm=False, title='Figure', colors=plotter.colors, show_followers=True)

# next

target = np.array((-10, 10, 20))
targets = target.reshape(1, 3)

# initial drone separation
d = 20
surface = np.array([
    [0, 0, 0],
    [0, d, 0],
    [0, d, d],
    [0, 0, d]])

# target radius
r = 2.5

# target formation sidelength
l = r  * 2

directory = pathlib.Path('example_e_2')
x, phi_t = minimise_flux(target, surface, l)
# time optimal path parameterisation

# followers
from formationplanning.trajectory import generate_follower_path
f = generate_follower_path(x)

x = np.concatenate((x, f), axis=1)

# leaders
x = preprocess(x)
x_t, v_t, a_t, t = reparametrise(x, -v_max, v_max, -a_max, a_max)
x_t = x_t.reshape((x_t.shape[0], 9, 3))
v_t = v_t.reshape((v_t.shape[0], 9, 3))
a_t = a_t.reshape((a_t.shape[0], 9, 3))

# follow the trajectory using 3d particle simulation with p controller on the applied force
dt = t[1] - t[0]
x1, xd1, xdd1 = dynamics_sim([0, 0, 0], x_t[:, 0, :], dt, u_mod=a_max) # drone 1
x2, xd2, xdd2 = dynamics_sim([0, 5, 0], x_t[:, 1, :], dt, u_mod=a_max) # drone 2
x3, xd3, xdd3 = dynamics_sim([0, 5, 5], x_t[:, 2, :], dt, u_mod=a_max) # drone 3
x4, xd4, xdd4 = dynamics_sim([0, 0, 5], x_t[:, 3, :], dt, u_mod=a_max) # drone 4

f1, fd1, fdd1 = dynamics_sim(x_t[0, 4, :], x_t[:, 4, :], dt, u_mod=a_max) # drone 5
f2, fd2, fdd2 = dynamics_sim(x_t[0, 5, :], x_t[:, 5, :], dt, u_mod=a_max) # drone 6
f3, fd3, fdd3 = dynamics_sim(x_t[0, 6, :], x_t[:, 6, :], dt, u_mod=a_max) # drone 7
f4, fd4, fdd4 = dynamics_sim(x_t[0, 7, :], x_t[:, 7, :], dt, u_mod=a_max) # drone 8
f5, fd5, fdd5 = dynamics_sim(x_t[0, 8, :], x_t[:, 8, :], dt, u_mod=a_max) # drone 9

x_t = np.stack((x1, x2, x3, x4, f1, f2, f3, f4, f5), axis=1)
xd_t = np.stack((xd1, xd2, xd3, xd4, fd1, fd2, fd3, fd4, fd5), axis=1)
xdd_t = np.stack((xdd1, xdd2, xdd3, xdd4, fdd1, fdd2, fdd3, fdd4, fdd5), axis=1)

# plot the trajectories
plotter.colors = ['#009E73', '#56B4E9', '#E69F00', '#D55E00']
plotter.labels = ['Path', 'Simulation']
plotter.linestyles = ['--', '-']
plotter.formation_slices = [0, -1, 450]
plotter.skip_formation = 200
plotter.elev = 20
plotter.azim = 180 + 90  + 10
plot([x, x_t], target, r, directory=directory, phi=phi_t)
plot_2d(xd_t, xdd_t, t, directory=directory)

animate.plot(x_t, targets, r, phi=phi_t, directory=directory,
        cm=False, title='Figure', colors=plotter.colors, show_followers=True)