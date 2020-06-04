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
from formationplanning.minimise_flux import minimise_flux
from formationplanning.plot_flux_minimisation_data_hemipshere_single import plot
from formationplanning.trajectory import map_to_time
from formationplanning.trajectory import interpolate_trajectory
from formationplanning import trajectory
import pathlib

target = np.array((50, 50, 50))

# initial drone separation
d = 5
surface = np.array([
    [0, 0, 0],
    [0, d, 0],
    [0, d, d],
    [0, 0, d]])

# hostile radius
r = 20

# target formation sidelength
l = r * 2

directory = pathlib.Path('target_right_up')
x_t, phi_t = minimise_flux(target, surface, l)
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
plot(x_t_int[::10], target, r, directory=directory, phi=phi_t)

print('flux final', phi_t[-1])
print('flux proportion', phi_t[-1] / 4 / np.pi)
