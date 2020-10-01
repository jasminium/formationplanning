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

import numpy as np
from formationplanning.minimise_flux_q import minimise_flux
from formationplanning.plot_flux_minimisation_data_hemipshere_obstacles import plot
from formationplanning.trajectory import map_to_time
from formationplanning.trajectory import interpolate_trajectory
from formationplanning import trajectory
import pathlib
import sys

# View the problem.
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
fig = plt.figure()
ax = Axes3D(fig)
d = 9

# basic d length plate
x = [0,0,0,0]
y = [0,d,d,0]
z = [0,0,14,14]
verts = [np.stack([x, y, z], axis=1)]

# translation vectors for the plate
x_t = [10, 10, 10, 10]
y_t = [5, 5, 5, 5]
z_t = [0, 0, 0, 0]

# translate the plate
verts_t = [np.stack([x_t, y_t, z_t], axis=1)]
verts = np.add(verts, verts_t)
obstacles = [verts]

obstacle  = Poly3DCollection(verts)
obstacle.set_color('orange')
ax.add_collection3d(obstacle)
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)
ax.set_zlim(0, 20)
plt.title('Scene')
plt.show()

# simple raw charge case
q = np.zeros((21, 21, 21))
q[20, 20, 20] = 1
q[10, 5:15, 5:15] = -1

# initial drone separation
d = 5
surface = np.array([
    [0, 0, 0],
    [0, d, 0],
    [0, d, d],
    [0, 0, d]])

# hostile radius
r = 2.5

# target formation sidelength
l = r * 2

directory = pathlib.Path('target_right_up_q_simple_test')
x_t, phi_t = minimise_flux(q, surface, dx=1, l=l)
# map the solution to time domain
t = map_to_time(x_t)
# interpolate the trajectory at every 0.1 seconds.
t_int, x_t_int = interpolate_trajectory(t, x_t)
# generate the follower drone trajectories
f_t_int = trajectory.generate_hemisphere_followers(x_t_int)
# export the trajectories for DJI virtual sticks
v = np.concatenate((x_t_int, f_t_int), axis=1)
trajectory.export_trajectory(v, directory)
# plot the trajectories
plot(x_t_int, np.array([np.array((20, 20, 20))]), r, obstacles, directory=directory, phi=phi_t)

print('flux final', phi_t[-1])
print('flux proportion', phi_t[-1] / 4 / np.pi)
