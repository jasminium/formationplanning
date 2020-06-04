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

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import matplotlib.animation as animation

def plot(x_j, targets, target_r, phi=None, directory=None):

    
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

    # particle order
    # r2, r6, r7, r3

    # generate the follower trajectories from the parameterised vertex trajectories.

    x_j = x_j[::1]
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
    
    # find the centre of mass of the targets
    n = targets.shape[0]
    centre = np.zeros((3))
    for t in targets:
        centre += t
    
    centre = centre / n

    boo = ax.scatter(x_j[:, 0, 0], x_j[:, 0, 1], x_j[:, 0, 2], c=c_s, cmap='winter', marker='o', depthshade=False)
    ax.scatter(x_j[:, 1, 0], x_j[:, 1, 1], x_j[:, 1, 2], c=c_s, cmap='cool', marker='o', depthshade=False)
    ax.scatter(x_j[:, 2, 0], x_j[:, 2, 1], x_j[:, 2, 2], c=c_s, cmap='summer', marker='o', depthshade=False)
    ax.scatter(x_j[:, 3, 0], x_j[:, 3, 1], x_j[:, 3, 2], c=c_s, cmap='spring', marker='o', depthshade=False)

    ax.plot(x_j[:, 0, 0], x_j[:, 0, 1], x_j[:, 0, 2], c='blue', linewidth='0.5')
    ax.plot(x_j[:, 1, 0], x_j[:, 1, 1], x_j[:, 1, 2], c='blue', linewidth='0.5')
    ax.plot(x_j[:, 2, 0], x_j[:, 2, 1], x_j[:, 2, 2], c='blue', linewidth='0.5')
    ax.plot(x_j[:, 3, 0], x_j[:, 3, 1], x_j[:, 3, 2], c='blue', linewidth='0.5')

    # draw sphere
    for target in targets:
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = target_r * np.cos(u) * np.sin(v) + target[0]
        y = target_r * np.sin(u) * np.sin(v) + target[1]
        z = target_r * np.cos(v) + target[2]
        ax.plot_surface(x, y, z, color="blue")

    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = target_r * np.cos(u) * np.sin(v) + centre[0]
    y = target_r * np.sin(u) * np.sin(v) + centre[1]
    z = target_r * np.cos(v) + centre[2]
    ax.plot_surface(x, y, z, color="red")
    #plt.colorbar(boo)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')

    if directory is not None:
        directory.mkdir(exist_ok=True)
        plt.savefig(directory / 'trajectory.png', dpi=330)

    # trisurface doesnt allow for flat surface
    vertices = x_j[1:]
    followers = np.stack((f1, f2, f3, f4, f5), axis=1)[1:]

    # animation
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')

    fv = np.concatenate((vertices, followers), axis=1)
    xmin = fv[:, :, 0].min()
    xmax = fv[:, :, 0].max()
    ymin = fv[:, :, 1].min()
    ymax = fv[:, :, 1].max()
    zmin = fv[:, :, 2].min()
    zmax = fv[:, :, 2].max()

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.plot_trisurf(vertices[0, :, 0], vertices[0, :, 1], vertices[0, :, 2], color='blue')
    
    ax.plot_surface(x, y, z, color="r")


    def update_graph(j):
        ax.clear()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        
        for target in targets:
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = target_r * np.cos(u) * np.sin(v) + target[0]
            y = target_r * np.sin(u) * np.sin(v) + target[1]
            z = target_r * np.cos(v) + target[2]
            ax.plot_surface(x, y, z, color="blue")
        
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = target_r * np.cos(u) * np.sin(v) + centre[0]
        y = target_r * np.sin(u) * np.sin(v) + centre[1]
        z = target_r * np.cos(v) + centre[2]
        ax.plot_surface(x, y, z, color="red")
 
        #graph_1 = ax.plot_trisurf(vertices[j, :, 0], vertices[j, :, 1], vertices[j, :, 2], color='blue')
        #graph_2 = ax.plot_trisurf(followers[j, :, 0], followers[j, :, 1], followers[j, :, 2], color='green')

        ax.scatter(vertices[j, 0, 0], vertices[j, 0, 1], vertices[j, 0, 2], color='springgreen')
        ax.scatter(vertices[j, 1, 0], vertices[j, 1, 1], vertices[j, 1, 2], color='springgreen')
        ax.scatter(vertices[j, 2, 0], vertices[j, 2, 1], vertices[j, 2, 2], color='springgreen')
        ax.scatter(vertices[j, 3, 0], vertices[j, 3, 1], vertices[j, 3, 2], color='springgreen')

        ax.scatter(followers[j, 0, 0], followers[j, 0, 1], followers[j, 0, 2], color='fuchsia')
        ax.scatter(followers[j, 1, 0], followers[j, 1, 1], followers[j, 1, 2], color='fuchsia')
        ax.scatter(followers[j, 2, 0], followers[j, 2, 1], followers[j, 2, 2], color='fuchsia')
        ax.scatter(followers[j, 3, 0], followers[j, 3, 1], followers[j, 3, 2], color='fuchsia')
        ax.scatter(followers[j, 4, 0], followers[j, 4, 1], followers[j, 4, 2], color='orange')

        #ax.text(1, 0.5, 0.5, s='flux ' + str(round(phi[j], 3)), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    ani = animation.FuncAnimation(fig, update_graph, vertices.shape[0],
                                interval=100, blit=False)

    fig = plt.figure(3)

    plt.plot(phi)

    plt.xlabel('Iterations')
    plt.ylabel('Flux [E/Vm^2]')


    plt.show()


    width = 4.98
    height = width / 1.2
    plt.rcParams['figure.figsize'] = width, height

    gap = vertices.shape[0] // 9

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    #plt.subplots(constrained_layout=True)

    # last image is always the final image
    for i, index in enumerate([0, 1, 2, 3, 4, 5, 6, 7, -1]):
        fig = plt.figure(4 + i, constrained_layout=True)
        ax = fig.add_subplot(111, projection='3d')
        #graph_1 = ax.plot_trisurf(vertices[index * gap, :, 0], vertices[index * gap, :, 1], vertices[index * gap, :, 2], color='blue')
        #graph_2 = ax.plot_trisurf(followers[index * gap, :, 0], followers[index * gap, :, 1], followers[index * gap, :, 2], color='green')
        if index == -1:
            gap = 1
        ax.scatter(vertices[index * gap, 0, 0], vertices[index * gap, 0, 1], vertices[index * gap, 0, 2], color='springgreen', marker='o', s=40)
        ax.scatter(vertices[index * gap, 1, 0], vertices[index * gap, 1, 1], vertices[index * gap, 1, 2], color='springgreen', marker='o', s=40)
        ax.scatter(vertices[index * gap, 2, 0], vertices[index * gap, 2, 1], vertices[index * gap, 2, 2], color='springgreen', marker='o', s=40)
        ax.scatter(vertices[index * gap, 3, 0], vertices[index * gap, 3, 1], vertices[index * gap, 3, 2], color='springgreen', marker='o', s=40)

        ax.scatter(followers[index * gap, 0, 0], followers[index * gap, 0, 1], followers[index * gap, 0, 2], color='fuchsia', marker='o', s=40)
        ax.scatter(followers[index * gap, 1, 0], followers[index * gap, 1, 1], followers[index * gap, 1, 2], color='fuchsia', marker='o', s=40)
        ax.scatter(followers[index * gap, 2, 0], followers[index * gap, 2, 1], followers[index * gap, 2, 2], color='fuchsia', marker='o', s=40)
        ax.scatter(followers[index * gap, 3, 0], followers[index * gap, 3, 1], followers[index * gap, 3, 2], color='fuchsia', marker='o', s=40)
        ax.scatter(followers[index * gap, 4, 0], followers[index * gap, 4, 1], followers[index * gap, 4, 2], color='orange', marker='o', s=40)
        
        for target in targets:
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = target_r * np.cos(u) * np.sin(v) + target[0]
            y = target_r * np.sin(u) * np.sin(v) + target[1]
            z = target_r * np.cos(v) + target[2]
            ax.plot_surface(x, y, z, color="blue", alpha=0.5)

        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = target_r * np.cos(u) * np.sin(v) + centre[0]
        y = target_r * np.sin(u) * np.sin(v) + centre[1]
        z = target_r * np.cos(v) + centre[2]
        ax.plot_surface(x, y, z, color="red", alpha=0.5)

        xmin = fv[index*gap, :, 0].min()
        xmax = fv[index*gap, :, 0].max()
        ymin = fv[index*gap, :, 1].min()
        ymax = fv[index*gap, :, 1].max()
        zmin = fv[index*gap, :, 2].min()
        zmax = fv[index*gap, :, 2].max()

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')

        if directory is not None:
            directory.mkdir(exist_ok=True)
            plt.savefig(directory / 'plate_{}.png'.format(i), dpi=330)

    # save animation
    #Writer = matplotlib.animation.writers['ffmpeg']
    #writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
    #ani.save('3d-scatted-animated.mp4', writer=writer)

    return fv