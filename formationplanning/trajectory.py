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
from scipy import interpolate
from pathlib import Path

# max velocity of particle
v_max = 3
# sampling interval
s_i = 0.1

def map_to_time(x_i):
    l = x_i.shape[0]
    t = np.zeros((l))

    for i in range(1, l):
        # points at last iteration
        x0 = x_i[i - 1]
        # points at current iteration
        x1 = x_i[i]
        d = x0 - x1
        d2 = d * d
        ds = np.sqrt(np.sum(d2, axis=1))
        # longest arc length over the interval between any 1 position
        ds_max = np.amax(ds)
        # delta required to travel at vmax over the interval
        dt = ds_max / v_max
        t[i] = dt
    
    return np.cumsum(t)

def interpolate_trajectory(t, x_t):
    # interpolate to give value every 0.1 seconds
    interp = interpolate.interp1d(t, x_t, axis=0)
    t = np.arange(t[0], t[-1], s_i)
    x_t_in = interp(t)
    return t, x_t_in

def generate_hemisphere_followers(x_t):
    r2 = x_t[:, 0, :] 
    r6 = x_t[:, 1, :] 
    r7 = x_t[:, 2, :] 

    # generate the A B D basis

    c = 0.5 * (r2 + r7)
    
    a = c - r7
    a_l = np.linalg.norm(a, axis=1)
    a_l = np.stack((a_l, a_l, a_l), axis=1)
    a = np.multiply(a, 1 / a_l)
    
    b = c - r6
    b_l = np.linalg.norm(b, axis=1)
    b_l = np.stack((b_l, b_l, b_l), axis=1)
    b = np.multiply(b, 1 / b_l)
    
    d = np.cross(a, b)
    d_l = np.linalg.norm(d, axis=1)
    d_l = np.stack((d_l, d_l, d_l), axis=1)
    d = np.multiply(d, 1 / d_l)

    # radius of the formation
    r = np.linalg.norm(c - r7, axis=1)

    # now construct any points we like in this basis.

    # distance along d to the centre of the circle
    d_2 = r / 2
    d_2 = np.stack((d_2, d_2, d_2), axis=1)
    r = np.stack((r, r, r), axis=1)
    r_2 = np.sqrt(r**2 - d_2**2)

    # vector of first follower
    f1 = (d_2 * d + r_2 * a) + c
    f2 = (d_2 * d - r_2 * a) + c
    f3 = (d_2 * d + r_2 * b) + c
    f4 = (d_2 * d - r_2 * b) + c
    f5 = (r * d) + c

    followers = np.stack((f1, f2, f3, f4, f5), axis=1)
    return followers

def export_trajectory(x_t, directory):

    # save the multi trajectory data
    Path(directory).mkdir(parents=True, exist_ok=True)

    # iterate over each drones trajectories.
    for i in range(0, x_t.shape[1]):
        # save each drone trajectory by component
        for j, c in enumerate(['x', 'y', 'z']):
            fp = "{}{}_t.csv".format(c, i)
            t = x_t[:, i, j]
            t.tofile(directory / fp, sep="\n")
