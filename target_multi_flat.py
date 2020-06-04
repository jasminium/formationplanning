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
from formationplanning.plot_flux_minimisation_data_hemipshere import plot
from formationplanning.trajectory import map_to_time
from formationplanning.trajectory import interpolate_trajectory
from formationplanning import trajectory
import pathlib
import time

def target_multi_right_large():
    # locate a target swarm in a random location
    np.random.seed(20)
    mean = np.random.rand() * 200
    mu, sigma = 200, 100
    targets = np.random.normal(mu, sigma, (10, 3))

    # initial drone separation
    d = 5

    # formation of the leaders
    surface = np.array([
        [0, 0, 0],
        [0, d, 0],
        [0, d, d],
        [0, 0, d]])

    directory = pathlib.Path('target_multi_flat_right_large')
    # find the 3d path solution
    x_t, phi_t = minimise_flux(targets, surface)
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
    plot(x_t_int[::50], targets, 5, directory=directory, phi=phi_t)

    print('flux final', phi_t[-1])
    print('flux proportion', phi_t[-1] / 4 / np.pi)


def target_multi_left_small():
    # locate a target swarm in a random location
    np.random.seed(22)
    mu, sigma = 200, 10
    targets = np.random.normal(mu, sigma, (10, 3))

        # initial drone separation
    d = 5

    # formation of the leaders
    surface = np.array([
        [0, 0, 0],
        [0, d, 0],
        [0, d, d],
        [0, 0, d]])

    # flip the x components
    targets[:, 0]  = targets[:, 0] * -1
    directory = pathlib.Path('target_multi_flat_left_small')
    # find the 3d path solution
    x_t, phi_t = minimise_flux(targets, surface)
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
    plot(x_t_int[::50], targets, 5, directory=directory, phi=phi_t)
    print('flux final', phi_t[-1])
    print('flux proportion', phi_t[-1] / 4 / np.pi)


if __name__ == '__main__':

    target_multi_left_small()
    target_multi_right_large()