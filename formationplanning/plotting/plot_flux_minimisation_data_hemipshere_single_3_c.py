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

colors = None
labels = ['FG', 'Wang', 'Wang with constraints']
target_color = '#0072B2'
targets_color = 'grey'
linestyles = None
skip_formation = 200
linecolor = '#0072B2'
elev = 0
azim = 0

SMALL_SIZE = 8
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#plt.rc('text', usetex = True)
plt.rc('font', **{'family':"sans-serif"})

params = {'text.latex.preamble': [r'\usepackage{siunitx}', 
    r'\usepackage{sfmath}', r'\sisetup{detect-family = true}',
    r'\usepackage{amsmath}']}   
plt.rcParams.update(params)

width = 4.98
height = width / 1.2
plt.rcParams['figure.figsize'] = width, height

def plot_2d(v, a, t, directory=None):
    """
    plt.figure(2, constrained_layout=True)
    plt.plot(t, v[:, :, 0], label=r'$x_x$', linewidth=1, color=colors[0])
    plt.plot(t, v[:, :, 1], label=r'$x_y$', linewidth=1, color=colors[1])
    plt.plot(t, v[:, :, 2], label=r'$x_z$', linewidth=1, color=colors[2])
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [s]')
    plt.title('Velocity')

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=colors[0], lw=1),
                    Line2D([0], [0], color=colors[1], lw=1),
                    Line2D([0], [0], color=colors[2], lw=1)]

    plt.gca().legend(custom_lines, [r'$\dot{x}_x$', r'$\dot{x}_y$', r'$\dot{x}_z$'])

    if directory is not None:
        directory.mkdir(exist_ok=True)
        plt.savefig(directory / 'velocity.png', dpi=330)

    plt.figure(3, constrained_layout=True)
    plt.plot(t, a[:, :, 0], label='x', linewidth=1, color=colors[0])
    plt.plot(t, a[:, :, 1], label='y', linewidth=1, color=colors[1])
    plt.plot(t, a[:, :, 2], label='z', linewidth=1, color=colors[2])
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [s]')
    plt.title('Acceleration')

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=colors[0], lw=1),
                    Line2D([0], [0], color=colors[1], lw=1),
                    Line2D([0], [0], color=colors[2], lw=1)]

    plt.gca().legend(custom_lines, [r'$\ddot{x}_x$', r'$\ddot{x}_y$', r'$\ddot{x}_z$'])

    if directory is not None:
        directory.mkdir(exist_ok=True)
        plt.savefig(directory / 'acceleration.png', dpi=330)
    
    """
    

    plt.figure(4, constrained_layout=True)
    plt.plot(t, v[:, :, 0], linewidth=1, color=colors[0], linestyle=linestyles[0])
    plt.plot(t, v[:, :, 1], linewidth=1, color=colors[0], linestyle=linestyles[0])
    plt.plot(t, v[:, :, 2], linewidth=1, color=colors[0], linestyle=linestyles[0])
    plt.plot(t, a[:, :, 0], linewidth=1, color=colors[1], linestyle=linestyles[1])
    plt.plot(t, a[:, :, 1], linewidth=1, color=colors[1], linestyle=linestyles[1])
    plt.plot(t, a[:, :, 2], linewidth=1, color=colors[1], linestyle=linestyles[1])

    plt.xlabel('Time [s]')

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=colors[0], lw=1, linestyle=linestyles[0]),
                    Line2D([0], [0], color=colors[1], lw=1, linestyle=linestyles[1])]

    plt.gca().legend(custom_lines, ['$\dot{x}$ [\si{\metre\per\second}]', '$\ddot{x}$ [\si{\metre\per\square\second}]'])

    if directory is not None:
        directory.mkdir(exist_ok=True)
        plt.savefig(directory / 'v_a.png', dpi=330)

    plt.show()

def plot(x, targets, target_r, phi=None, directory=None):
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

        ax.plot(x_j[:, 0, 0], x_j[:, 0, 1], x_j[:, 0, 2], c=colors[0], label=labels[i], linewidth=1, linestyle=linestyles[i])
        ax.plot(x_j[:, 1, 0], x_j[:, 1, 1], x_j[:, 1, 2], c=colors[1], linewidth=1, linestyle=linestyles[i])
        ax.plot(x_j[:, 2, 0], x_j[:, 2, 1], x_j[:, 2, 2], c=colors[2], linewidth=1, linestyle=linestyles[i])
        ax.plot(x_j[:, 3, 0], x_j[:, 3, 1], x_j[:, 3, 2], c=colors[3], linewidth=1, linestyle=linestyles[i])

    try:
        x1 = x[1]

        # plot the formation. first last and every 200th point
        x1 = x1[0:-1:skip_formation, :, :]
        xlast = x[1][-1, :, :]
        x1 = np.append(x1, [xlast], axis=0)
        ax.scatter(x1[:, 0, 0], x1[:, 0, 1], x1[:, 0, 2], c=colors[0], depthshade=False, s=8)
        ax.scatter(x1[:, 1, 0], x1[:, 1, 1], x1[:, 1, 2], c=colors[1], depthshade=False, s=8)
        ax.scatter(x1[:, 2, 0], x1[:, 2, 1], x1[:, 2, 2], c=colors[2], depthshade=False, s=8)
        ax.scatter(x1[:, 3, 0], x1[:, 3, 1], x1[:, 3, 2], c=colors[3], depthshade=False, s=8)

        # plot lines connecting the formation points

        for i in range(x1.shape[0]):
            xi = x1[i]

            l1 = np.stack((xi[0, :], xi[1, :]))
            l2 = np.stack((xi[1, :], xi[2, :]))
            l3 = np.stack((xi[2, :], xi[3, :]))
            l4 = np.stack((xi[3, :], xi[0, :]))

            ax.plot(l1[:, 0], l1[:, 1], l1[:, 2], color=linecolor, linewidth=0.5)
            ax.plot(l2[:, 0], l2[:, 1], l2[:, 2], color=linecolor, linewidth=0.5)
            ax.plot(l3[:, 0], l3[:, 1], l3[:, 2], color=linecolor, linewidth=0.5)
            ax.plot(l4[:, 0], l4[:, 1], l4[:, 2], color=linecolor, linewidth=0.5)
    except IndexError:
        pass

    # find the centre of mass of the targets
    n = targets.shape[0]
    centre = np.zeros((3))
    for t in targets:
        centre += t

    centre = centre / n

    for target in targets:
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = target_r * np.cos(u) * np.sin(v) + target[0]
        y = target_r * np.sin(u) * np.sin(v) + target[1]
        z = target_r * np.cos(v) + target[2]
        ax.plot_surface(x, y, z, color=targets_color, alpha=0.5)

    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = target_r * np.cos(u) * np.sin(v) + centre[0]
    y = target_r * np.sin(u) * np.sin(v) + centre[1]
    z = target_r * np.cos(v) + centre[2]
    ax.plot_surface(x, y, z, color=target_color, alpha=0.5)

    #plt.colorbar(boo)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')

    from matplotlib.lines import Line2D
    from matplotlib import markers
    custom_lines = [Line2D([0], [0], color='black', lw=1, linestyle=linestyles[0]),
                    Line2D([0], [0], color='black', lw=1, linestyle=linestyles[1]),
                    Line2D([0], [0], color=targets_color, marker='o', linestyle='None'),
                    Line2D([0], [0], color=target_color, marker='o', linestyle='None')]

    plt.gca().legend(custom_lines, ['Desired path', 'Simulated trajectory', 'Targets', 'Centre of charge'])
    
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    #ax.set_xlim(-10, 10, auto=True)
    #ax.set_ylim(-10, 10, auto=True)
    #ax.set_zlim(-10, 10, auto=True)

    ax.grid(True)

    ax.elev = elev
    ax.azim = azim

    if directory is not None:
        directory.mkdir(exist_ok=True)
        plt.savefig(directory / 'path.png', dpi=330)

    
    plt.show()