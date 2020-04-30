import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.lines import Line2D

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

width = 12.65 / 2.54
height = width / 1.2
plt.rcParams['figure.figsize'] = width, height

def plot(directory, save=False):

    phi_t = np.load(directory / 'phi_t.npy')
    phi_ref_t = np.load(directory / 'phi_ref_t.npy')
    d_phi_t = np.load(directory / 'd_phi_t.npy')
    vertices_2_t = np.load(directory / 'vertices_2_t.npy')
    followers_2_t = np.load(directory / 'followers_2_t.npy')
    index = int(np.load(directory / 'index.npy'))
    print(index)
    p = np.load(directory / 'p.npy')

    print('Expected flux for boundary box', 4 * np.pi * 6 / 6)
    print('t[0] flux', phi_t[0])

    step = 20

    t = np.arange(0, index)[::step]# / 50000 * 20

    it_time = np.arange(0, vertices_2_t.shape[0]) * 0.05

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(vertices_init[0][0], vertices_init[0][1], vertices_init[0][2], label='starting r2', color='red', alpha=0.3)
    ax.scatter(vertices_init[1][0], vertices_init[1][1], vertices_init[1][2], label='starting r6', color='red', alpha=0.3)
    ax.scatter(vertices_init[2][0], vertices_init[2][1], vertices_init[2][2], label='starting r7', color='red', alpha=0.3)
    ax.scatter(vertices_init[3][0], vertices_init[3][1], vertices_init[3][2], label='starting r3', color='red', alpha=0.3)

    ax.scatter(vertices_2[0][0], vertices_2[0][1], vertices_2[0][2], label='ending r2', color='red')
    ax.scatter(vertices_2[1][0], vertices_2[1][1], vertices_2[1][2], label='ending r6', color='red')
    ax.scatter(vertices_2[2][0], vertices_2[2][1], vertices_2[2][2], label='ending r7', color='red')
    ax.scatter(vertices_2[3][0], vertices_2[3][1], vertices_2[3][2], label='ending r3', color='red')

    ax.scatter(vertices_init_back[0][0], vertices_init_back[0][1], vertices_init_back[0][2], label='starting r1', color='green', alpha=0.3)
    ax.scatter(vertices_init_back[1][0], vertices_init_back[1][1], vertices_init_back[1][2], label='starting r4', color='green', alpha=0.3)
    ax.scatter(vertices_init_back[2][0], vertices_init_back[2][1], vertices_init_back[2][2], label='starting r9', color='green', alpha=0.3)
    ax.scatter(vertices_init_back[3][0], vertices_init_back[3][1], vertices_init_back[3][2], label='starting r5', color='green', alpha=0.3)

    for i in range(0, 4):
        ax.scatter(followers_2_t[index, i, 0], followers_2_t[index, i, 1], followers_2_t[index, i, 2], color='green')

    ax.scatter(p[0], p[1], p[2], marker='x', label='Source')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    fig.legend()
    #python plt.savefig('starting_and_finishing_points.png', dpi=330)




    plt.figure(3)
    plt.plot(phi_ref_t, label='ref phi')
    plt.plot(d_phi_t, label='total derivative phi')
    plt.title('Comparison of total derivative and phi')
    plt.xlabel('iterations')
    plt.ylabel('total flux')
    plt.savefig('comparison_phi_total_derivative.png', dpi=330)
    plt.legend()
    """

    plt.figure(1)
    plt.plot(phi_t)
    plt.title('Total Flux')
    plt.xlabel('Iterations')
    plt.ylabel('Total flux')
    ##python plt.savefig('total_flux.png', dpi=330)

    #-------------------------------------------------------------------
    # 3d time history of the vertices
    #-------------------------------------------------------------------
    custom_lines = [Line2D([0], [0], color='orange', lw=1),
                    Line2D([0], [0], color='blue', lw=1),
                    Line2D([0], [0], marker='o', color='w',
                            markerfacecolor='g')]

    fig = plt.figure(2)
    plt.subplots(constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')

    colors = ['Greens', 'Reds', 'Blues', 'Purples']
    colors2 = ['GnBu', 'OrRd', 'PuBu', 'YlGn']

    for i in range(0, 4):
        #ax.scatter(vertices_2_t[:, i, 0], vertices_2_t[:, i, 1], vertices_2_t[:, i, 2], cmap=colors[i], c=np.arange(0, iterations), s=2)
        #ax.scatter(followers_2_t[:, i, 0], followers_2_t[:, i, 1], followers_2_t[:, i, 2], cmap=colors2[i], c=np.arange(0, iterations), s=2)
        ax.plot3D(vertices_2_t[:, i, 0][:index], vertices_2_t[:, i, 1][:index], vertices_2_t[:, i, 2][:index], color='orange')
        ax.plot3D(followers_2_t[:, i, 0][:index], followers_2_t[:, i, 1][:index], followers_2_t[:, i, 2][:index], color='blue'), 

    ax.scatter(p[0], p[1], p[2], marker='o', label='Source', color='green')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.legend(custom_lines, ['Lead', 'Follow', 'Hostile'])
    if save:
        fig.savefig(directory / 'flux_3d_positions.png', dpi=330)

    # plot trajectories
    fig = plt.figure(3, constrained_layout=True)
    plt.title('Position')

    for i in range(0, 4):
        #ax.scatter(vertices_2_t[:, i, 0], vertices_2_t[:, i, 1], vertices_2_t[:, i, 2], cmap=colors[i], c=np.arange(0, iterations), s=2)
        #ax.scatter(followers_2_t[:, i, 0], followers_2_t[:, i, 1], followers_2_t[:, i, 2], cmap=colors2[i], c=np.arange(0, iterations), s=2)
        plt.plot(it_time[::step], vertices_2_t[:, i, 0][::step], label="drone #{} x".format(i))
        plt.plot(it_time[::step], vertices_2_t[:, i, 1][::step], label="drone #{} y".format(i))
        plt.plot(it_time[::step], vertices_2_t[:, i, 2][::step], label="drone #{} z".format(i))

    #plt.xlim(0, 21)
    plt.xlabel('iterations')
    plt.ylabel('component displacement [m]')
    plt.legend()
    #python plt.savefig('component_positions.png', dpi=330)


    fig = plt.figure(4, constrained_layout=True)
    plt.title('Velocity')

    time = t / 50000 * 10

    for i in range(0, 4):
        #ax.scatter(vertices_2_t[:, i, 0], vertices_2_t[:, i, 1], vertices_2_t[:, i, 2], cmap=colors[i], c=np.arange(0, iterations), s=2)
        #ax.scatter(followers_2_t[:, i, 0], followers_2_t[:, i, 1], followers_2_t[:, i, 2], cmap=colors2[i], c=np.arange(0, iterations), s=2)
        plt.plot(time, np.gradient(vertices_2_t[:, i, 0][::step], time[1] - time[0]), label="drone #{} x".format(i))
        plt.plot(time, np.gradient(vertices_2_t[:, i, 1][::step], time[1] - time[0]), label="drone #{} y".format(i))
        plt.plot(time, np.gradient(vertices_2_t[:, i, 2][::step], time[1] - time[0]), label="drone #{} z".format(i))

    #plt.xlim(0, 21)
    plt.ylabel('speed [ms-1]')
    plt.xlabel('time [s]')
    plt.legend()
    #python plt.savefig('velocity.png', dpi=330)


    # plot trajectories
    fig = plt.figure(5, constrained_layout=True)
    plt.title('Position')

    for i in range(0, 4):
        #ax.scatter(vertices_2_t[:, i, 0], vertices_2_t[:, i, 1], vertices_2_t[:, i, 2], cmap=colors[i], c=np.arange(0, iterations), s=2)
        #ax.scatter(followers_2_t[:, i, 0], followers_2_t[:, i, 1], followers_2_t[:, i, 2], cmap=colors2[i], c=np.arange(0, iterations), s=2)
        plt.plot(phi_t[::step] * 20, vertices_2_t[:, i, 0][::step], label="drone #{} x".format(i))
        plt.plot(phi_t[::step] * 20, vertices_2_t[:, i, 1][::step], label="drone #{} y".format(i))
        plt.plot(phi_t[::step] * 20, vertices_2_t[:, i, 2][::step], label="drone #{} z".format(i))

    #plt.xlim(0, 21)
    plt.xlabel('iterations')
    plt.ylabel('component displacement [m]')
    plt.legend()


    """
    print('Do antimation')

    # animation
    fig = plt.figure(9)
    ax = fig.add_subplot(111, projection='3d')

    #ax.set_xlim(-1, 1)
    #ax.set_ylim(-1, 1)
    #ax.set_zlim(-1, 1)

    ax.scatter(p[0], p[1], p[2], marker='x', label='Source')

    ax.set_xlim(-200, 10)
    ax.set_ylim(-15, 15)
    ax.set_zlim(0, 40)


    print(vertices_2_t.shape)
    vertices_2_t = vertices_2_t[::100]
    followers_2_t = followers_2_t[::100]
    print(vertices_2_t.shape)

    graph_1 = ax.scatter(vertices_2_t[0, 0, 0], vertices_2_t[0, 0, 1], vertices_2_t[0, 0, 2], color='red', s=2)
    graph_2 = ax.scatter(vertices_2_t[0, 1, 0], vertices_2_t[0, 1, 1], vertices_2_t[0, 1, 2], color='red', s=2)
    graph_3 = ax.scatter(vertices_2_t[0, 2, 0], vertices_2_t[0, 2, 1], vertices_2_t[0, 2, 2], color='red', s=2)
    graph_4 = ax.scatter(vertices_2_t[0, 3, 0], vertices_2_t[0, 3, 1], vertices_2_t[0, 3, 2], color='red', s=2)

    graph_5 = ax.scatter(followers_2_t[0, 0, 0], followers_2_t[0, 0, 1], followers_2_t[0, 0, 2], color='green', s=2)
    graph_6 = ax.scatter(followers_2_t[0, 1, 0], followers_2_t[0, 1, 1], followers_2_t[0, 1, 2], color='green', s=2)
    graph_7 = ax.scatter(followers_2_t[0, 2, 0], followers_2_t[0, 2, 1], followers_2_t[0, 2, 2], color='green', s=2)
    graph_8 = ax.scatter(followers_2_t[0, 3, 0], followers_2_t[0, 3, 1], followers_2_t[0, 3, 2], color='green', s=2)


    #scatters = [graph_1]

    def update_graph(j):
        graph_1._offsets3d = ([vertices_2_t[j, 0, 0]], [vertices_2_t[j, 0, 1]], [vertices_2_t[j, 0, 2]])
        graph_2._offsets3d = ([vertices_2_t[j, 1, 0]], [vertices_2_t[j, 1, 1]], [vertices_2_t[j, 1, 2]])
        graph_3._offsets3d = ([vertices_2_t[j, 2, 0]], [vertices_2_t[j, 2, 1]], [vertices_2_t[j, 2, 2]])
        graph_4._offsets3d = ([vertices_2_t[j, 3, 0]], [vertices_2_t[j, 3, 1]], [vertices_2_t[j, 3, 2]])

        graph_5._offsets3d = ([followers_2_t[j, 0, 0]], [followers_2_t[j, 0, 1]], [followers_2_t[j, 0, 2]])
        graph_6._offsets3d = ([followers_2_t[j, 1, 0]], [followers_2_t[j, 1, 1]], [followers_2_t[j, 1, 2]])
        graph_7._offsets3d = ([followers_2_t[j, 2, 0]], [followers_2_t[j, 2, 1]], [followers_2_t[j, 2, 2]])
        graph_8._offsets3d = ([followers_2_t[j, 3, 0]], [followers_2_t[j, 3, 1]], [followers_2_t[j, 3, 2]])

        #return scatters

    ani = matplotlib.animation.FuncAnimation(fig, update_graph, 378,
                                interval=10, blit=False)

    Writer = matplotlib.animation.writers['ffmpeg']
    writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
    ani.save('results/3d-scatted-animated.mp4', writer=writer)

    print("done")
    """

    plt.show()
