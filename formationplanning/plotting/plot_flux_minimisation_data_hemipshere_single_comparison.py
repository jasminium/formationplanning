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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.cm as cm
import numpy as np

colors = ['mediumseagreen', 'blue', 'tomato']
labels = ['FG', 'Wang', 'Wang with constraints']
target_color = '#0072B2'

def plot(x, target, target_r, phi=None, directory=None):

    SMALL_SIZE = 8
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    width = 4.98
    height = width / 1.2
    plt.rcParams['figure.figsize'] = width, height

    fig = plt.figure(1, constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')

    for i, x_i in enumerate(x):
        # particle order
        # r2, r6, r7, r3

        # generate the follower trajectories from the parameterised vertex trajectories.

        x_j = x_i[::1]
        c_s = np.arange(0, x_j.shape[0])

        r2 = x_j[:, 0, :]
        r6 = x_j[:, 1, :]
        r7 = x_j[:, 2, :]
        r3 = x_j[:, 3, :]

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

        #boo = ax.scatter(x_j[:, 0, 0], x_j[:, 0, 1], x_j[:, 0, 2], c=c_s, cmap='winter', marker='o', depthshade=False)
        #ax.scatter(x_j[:, 1, 0], x_j[:, 1, 1], x_j[:, 1, 2], c=c_s, cmap='cool', marker='o', depthshade=False)
        #ax.scatter(x_j[:, 2, 0], x_j[:, 2, 1], x_j[:, 2, 2], c=c_s, cmap='summer', marker='o', depthshade=False)
        #ax.scatter(x_j[:, 3, 0], x_j[:, 3, 1], x_j[:, 3, 2], c=c_s, cmap='spring', marker='o', depthshade=False)

        ax.plot(x_j[:, 0, 0], x_j[:, 0, 1], x_j[:, 0, 2], c=colors[i], label=labels[i], linewidth=1)
        ax.plot(x_j[:, 1, 0], x_j[:, 1, 1], x_j[:, 1, 2], c=colors[i], linewidth=1)
        ax.plot(x_j[:, 2, 0], x_j[:, 2, 1], x_j[:, 2, 2], c=colors[i], linewidth=1)
        ax.plot(x_j[:, 3, 0], x_j[:, 3, 1], x_j[:, 3, 2], c=colors[i], linewidth=1)

    # draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = target_r * np.cos(u) * np.sin(v) + target[0]
    y = target_r * np.sin(u) * np.sin(v) + target[1]
    z = target_r * np.cos(v) + target[2]
    ax.plot_surface(x, y, z, color=target_color, alpha=0.5)

    #plt.colorbar(boo)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')

    plt.legend()
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    #ax.set_xlim(-10, 10, auto=True)
    #ax.set_ylim(-10, 10, auto=True)
    #ax.set_zlim(-10, 10, auto=True)

    if directory is not None:
        directory.mkdir(exist_ok=True)
        plt.savefig(directory / 'path.png', dpi=330)

    plt.show()