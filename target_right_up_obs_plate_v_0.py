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
from formationplanning.q_solve import q_solve
from formationplanning.q_solve import v_r
from formationplanning.minimise_flux_q import minimise_flux
from formationplanning.plot_flux_minimisation_data_hemipshere_obstacles import plot
from formationplanning.trajectory import map_to_time
from formationplanning.trajectory import interpolate_trajectory
from formationplanning import trajectory
import pathlib
from pathlib import Path
import sys

test_name = __file__.split('.')[0]

try:
    q_path = Path('q') / test_name
    q_path = q_path.with_suffix('.npy')
    print('Try to read Read q dist, ', str(q_path))
    q = np.load(q_path)

except IOError:
    print('Create q dist')

    # simulate 20 x 20 x 20 m domain
    nx = 21
    ny = 21
    nz = 21
    # discretisation
    dl = 1

    # potential
    v = np.zeros((nx, nx, nz))

    # potential field due to point charge
    h = np.array((20, 20, 20))

    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            for k in range(v.shape[2]):
                # potential
                vijk = v_r(h[0] * dl, h[1] * dl, h[2] * dl,  i * dl, j * dl, k * dl)
                # position
                v[i, j, k] = vijk

    # uniform box
    v[10, 5:15, 5:15] = 0

    print('Solve for q')
    q, a = q_solve(v, dl)

    np.save(q_path, q)
    #np.save('q/20x20x20_t_box_a', a)

    v_f = a.dot(q.flatten())
    v_f = v_f.reshape(v.shape)

    slice_index = 2
    import matplotlib.pyplot as plt
    fig = plt.figure(2)
    plt.title('Potential')
    plt.imshow(v[:,:, slice_index], cmap='coolwarm', origin='lower')
    plt.colorbar()

    fig = plt.figure(3)
    plt.imshow(q[:, :, slice_index], cmap='coolwarm', origin='lower')
    plt.colorbar()
    plt.title('inverse q')

    fig = plt.figure(4)
    plt.imshow(v_f[:, :, slice_index], cmap='coolwarm', origin='lower')
    plt.colorbar()
    plt.title('Potential inversion')

    fig = plt.figure(5)
    plt.imshow((v_f - v)[:, :, slice_index], cmap='coolwarm', origin='lower')
    plt.colorbar()
    plt.title('Potential diff')

    plt.show()


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
x_t_1 = [10, 10, 10, 10]
y_t = [5, 5, 5, 5]
z_t = [0, 0, 0, 0]

# translate the plate
verts_t1 = [np.stack([x_t_1, y_t, z_t], axis=1)]
verts_p1 = np.add(verts, verts_t1)

for obstacle in [verts_p1]:
    obstacle  = Poly3DCollection(obstacle)
    obstacle.set_color('orange')
    ax.add_collection3d(obstacle)
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)
ax.set_zlim(0, 20)
plt.title('Scene')

fig = plt.figure(2)
plt.imshow(q[:, :, 10], cmap='coolwarm', origin='lower')
plt.colorbar()
plt.title('Charge dist')

plt.show()

# starting position of the drones in the front face of the formation
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

directory = pathlib.Path(test_name)
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
plot(x_t_int, np.array([np.array((20, 20, 20))]), r, [verts_p1], directory=directory, phi=phi_t)

print('flux final', phi_t[-1])
print('flux proportion', phi_t[-1] / 4 / np.pi)
