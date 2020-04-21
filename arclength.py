import numpy as np
from scipy import interpolate
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

phi_t = np.load('phi_t.npy')
phi_ref_t = np.load('phi_ref_t.npy')
d_phi_t = np.load('d_phi_t.npy')
vertices_2_t = np.load('vertices_2_t.npy')
followers_2_t = np.load('followers_2_t.npy')
index = np.load('index.npy')
p = np.load('p.npy')

def to_time_trajectory(x_it, y_it, z_it):

    # update period of the virtual stick api
    dt = 0.1
    # maximum speed of the drone in any direction
    s_max = 5

    # target arc length per iteration
    # predetermine the correct arc length
    arc_length = s_max * dt

    iterations = np.arange(0, x_it.shape[0])
    iterations_new = np.linspace(0, x_it.shape[0] - 1, x_it.shape[0] *  10)

    fx = interpolate.interp1d(iterations, x_it)
    fy = interpolate.interp1d(iterations, y_it)
    fz = interpolate.interp1d(iterations, z_it)

    x_it = fx(iterations_new)
    y_it = fy(iterations_new)
    z_it = fz(iterations_new)


    # calculate the iteration count which corresponds to the correct arc length travelled
    s = 0

    i = 1

    # trajectory in increments of the drone update period
    x_t = []
    y_t = []
    z_t = []

    go = True

    while i < x_it.shape[0] - 1:
        while s < arc_length:

            dx = x_it[i] - x_it[i - 1]
            dy = y_it[i] - y_it[i - 1]
            dz = z_it[i] - z_it[i - 1]

            ds = np.sqrt(dx * dx + dy * dy + dz * dz)

            s += ds

            i += 1

        #print('Found arc length {} at {} iterations'.format(s, i))
        #print('Corresponding positions ({},{},{})'.format(x_it[i], y_it[i], z_it[i]))

        # add the points to the time Trajectory
        x_t.append(x_it[i])
        y_t.append(y_it[i])
        z_t.append(z_it[i])

        s = 0

    return np.array(x_t), np.array(y_t), np.array(z_t)

# first vertex parameteric equation as function of iteration number
x0_it, y0_it, z0_it =  vertices_2_t[:, 0, 0], vertices_2_t[:, 0, 1], vertices_2_t[:, 0, 2]
x1_it, y1_it, z1_it =  vertices_2_t[:, 1, 0], vertices_2_t[:, 1, 1], vertices_2_t[:, 1, 2]
x2_it, y2_it, z2_it =  vertices_2_t[:, 2, 0], vertices_2_t[:, 2, 1], vertices_2_t[:, 2, 2]
x3_it, y3_it, z3_it =  vertices_2_t[:, 3, 0], vertices_2_t[:, 3, 1], vertices_2_t[:, 3, 2]

x0_t, y0_t, z0_t = to_time_trajectory(x0_it, y0_it, z0_it)
x1_t, y1_t, z1_t = to_time_trajectory(x1_it, y1_it, z1_it)
x2_t, y2_t, z2_t = to_time_trajectory(x2_it, y2_it, z2_it)
x3_t, y3_t, z3_t = to_time_trajectory(x3_it, y3_it, z3_it)

# save the trajectory data
x0_t.tofile("x0_t.csv", sep="\n")
y0_t.tofile("y0_t.csv", sep="\n")
z0_t.tofile("z0_t.csv", sep="\n")
x1_t.tofile("x1_t.csv", sep="\n")
y1_t.tofile("y1_t.csv", sep="\n")
z1_t.tofile("z1_t.csv", sep="\n")
x2_t.tofile("x2_t.csv", sep="\n")
y2_t.tofile("y2_t.csv", sep="\n")
z2_t.tofile("z2_t.csv", sep="\n")
x3_t.tofile("x3_t.csv", sep="\n")
y3_t.tofile("y3_t.csv", sep="\n")
z3_t.tofile("z3_t.csv", sep="\n")

# 3d time history of the vertices
fig = plt.figure(5)
ax = fig.add_subplot(111, projection='3d')

colors = ['Greens', 'Reds', 'Blues', 'Purples']
colors2 = ['GnBu', 'OrRd', 'PuBu', 'YlGn']

it_step = 20
t_step = 10

ax.scatter(x0_it[::it_step], y0_it[::it_step], z0_it[::it_step], color='red', s=2, label='iterations')
ax.scatter(x0_t[::t_step], y0_t[::t_step], z0_t[::t_step], color='green', s=2, label='time')
ax.scatter(x1_it[::it_step], y1_it[::it_step], z1_it[::it_step], color='red', s=2, label='iterations')
ax.scatter(x1_t[::t_step], y1_t[::t_step], z1_t[::t_step], color='green', s=2, label='time')
ax.scatter(x2_it[::it_step], y2_it[::it_step], z2_it[::it_step], color='red', s=2, label='iterations')
ax.scatter(x2_t[::t_step], y2_t[::t_step], z2_t[::t_step], color='green', s=2, label='time')
ax.scatter(x3_it[::it_step], y3_it[::it_step], z3_it[::it_step], color='red', s=2, label='iterations')
ax.scatter(x3_t[::t_step], y3_t[::t_step], z3_t[::t_step], color='green', s=2, label='time')

ax.scatter(p[0], p[1], p[2], marker='x', label='Source')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# set scale to check if arclength is constant
#ax.set_xlim3d(-200, 30)
#ax.set_ylim3d(-200, 30)
#ax.set_zlim3d(-200, 30)
plt.legend()
plt.savefig('trajectory.png', dpi=330)


fig = plt.figure(6)
t = np.linspace(0, 1, len(x0_t)) * len(x0_t) * 0.1
plt.title('Trajectory in time')
plt.plot(t[::10], x0_t[::10], label='x')
plt.plot(t[::10], y0_t[::10], label='y')
plt.plot(t[::10], z0_t[::10], label='z')

plt.xlabel('time [s]')
plt.legend()
plt.ylabel('displacement [m]')
plt.savefig('time-trajectory.png', dpi=330)

fig = plt.figure(7)
plt.title('Trajectory in iterations')
plt.plot(x0_it[::100], label='x')
plt.plot(y0_it[::100], label='y')
plt.plot(z0_it[::100], label='z')

plt.xlabel('iterations')
plt.ylabel('displacement [m]')
plt.legend()
plt.savefig('iteration-trajectory.png', dpi=330)

plt.show()
