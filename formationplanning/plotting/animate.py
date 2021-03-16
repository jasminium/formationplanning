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
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

def plot(x_j, targets, target_r, phi=None, directory=None,
        cm=True, title=None, colors=None, show_followers=False):

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

    # interpolate data to view motion smoothly at 60 fps
    fps = 24
    n = x_j.shape[0]
    # duration
    d = n * 0.01
    # time steps
    t = np.linspace(0, d, n)
    # interpolated time steps
    ti = np.linspace(0, d, int(d * fps))
    from scipy.interpolate import interp1d
    inter = interp1d(t, x_j, axis=0)
    # interpolate trajectories
    x_j = inter(ti)

    # each particle
    # particle order
    # r2, r6, r7, r3
    r2 = x_j[:, 0, :]
    r6 = x_j[:, 1, :]
    r7 = x_j[:, 2, :]
    r3 = x_j[:, 3, :]

    # find the centre of mass of the targets
    centre = np.average(targets, axis=0)

    # exclude first iteration
    vertices = x_j[1:]

    # animation
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')

    xmin = vertices[:, :, 0].min()
    xmax = vertices[:, :, 0].max()
    ymin = vertices[:, :, 1].min()
    ymax = vertices[:, :, 1].max()
    zmin = vertices[:, :, 2].min()
    zmax = vertices[:, :, 2].max()

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    def update_graph(j):
        ax.clear()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')

        #if title:
        #    ax.set_title(title)

        for target in targets:
            x, y, z = sphere(target, target_r)
            ax.plot_surface(x, y, z, color="#CC79A7")
        if cm:
            x, y, z = sphere(centre, target_r)
            ax.plot_surface(x, y, z, color="#D55E00")

        ax.scatter(vertices[j, 0, 0], vertices[j, 0, 1], vertices[j, 0, 2], color=colors[0])
        ax.scatter(vertices[j, 1, 0], vertices[j, 1, 1], vertices[j, 1, 2], color=colors[1])
        ax.scatter(vertices[j, 2, 0], vertices[j, 2, 1], vertices[j, 2, 2], color=colors[2])
        ax.scatter(vertices[j, 3, 0], vertices[j, 3, 1], vertices[j, 3, 2], color=colors[3])

        # plot a surface which intersections the front face of the formation
        #ax.plot_trisurf(vertices[j, :, 0], vertices[j, :, 1], vertices[j, :, 2], color='blue')

        if show_followers:
            ax.scatter(vertices[j, 4, 0], vertices[j, 4, 1], vertices[j, 4, 2], color='grey')
            ax.scatter(vertices[j, 5, 0], vertices[j, 5, 1], vertices[j, 5, 2], color='grey')
            ax.scatter(vertices[j, 6, 0], vertices[j, 6, 1], vertices[j, 6, 2], color='grey')
            ax.scatter(vertices[j, 7, 0], vertices[j, 7, 1], vertices[j, 7, 2], color='grey')
            ax.scatter(vertices[j, 8, 0], vertices[j, 8, 1], vertices[j, 8, 2], color='grey')

        # show the total flux through the surface
        #ax.text(1, 0.5, 0.5, s='flux ' + str(round(phi[j], 3)), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    ani = animation.FuncAnimation(fig, update_graph, vertices.shape[0],
                                interval=1/fps * 1000, blit=False)
    plt.show()


    # save animation as GIF and mp4
    print('Write simulated trajectory GIF and MP4')
    fn = title.strip() + '.gif'
    ani.save(directory / fn, writer='imagemagick', fps=fps)

    Writer = matplotlib.animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=-1, extra_args=['-vcodec', 'libx264'])
    fn = title.strip() + '.mp4'
    ani.save(directory / fn, writer=writer)

def sphere(centre, r):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = r * np.cos(u) * np.sin(v) + centre[0]
    y = r * np.sin(u) * np.sin(v) + centre[1]
    z = r * np.cos(v) + centre[2]
    return x, y, z
