import numpy as np
from scipy import interpolate
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec



width = 12.65 / 2.54
height = width / 1.2

plt.rcParams['figure.figsize'] = width, height
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 8}

matplotlib.rc('font', **font)

phi_t = np.load('results/phi_t.npy')
phi_ref_t = np.load('results/phi_ref_t.npy')
d_phi_t = np.load('results/d_phi_t.npy')
vertices_2_t = np.load('results/vertices_2_t.npy')
followers_2_t = np.load('results/followers_2_t.npy')
index = np.load('results/index.npy')
p = np.load('results/p.npy')

# update period of the virtual stick api
dt = 0.05
# maximum speed of the drone in any direction
s_max = 3

# target arc length per iteration
# predetermine the correct arc length
arc_length = s_max * dt


def to_time_trajectory(x_it, y_it, z_it):

    # iteration array for the generated path
    iterations = np.arange(0, x_it.shape[0])


    # interpolate the generated path such that the max distance between
    # successive points is not greater than the target arc length.
    fx = interpolate.interp1d(iterations, x_it, kind=1)
    fy = interpolate.interp1d(iterations, y_it, kind=1)
    fz = interpolate.interp1d(iterations, z_it, kind=1)

    iterations_new = np.linspace(0, x_it.shape[0] - 1, x_it.shape[0] *  20)
    x_it = fx(iterations_new)
    y_it = fy(iterations_new)
    z_it = fz(iterations_new)

    # calculate the iteration count which corresponds to the correct arc length travelled
    s = 0

    # trajectory in increments of the drone update period
    x_t = []
    y_t = []
    z_t = []
    # track the calculated arclength
    s_t = []

    ds_t = []

    i = 1

    while i < x_it.shape[0] - 1:
        while s < arc_length:

            # total change in x w.r.t iterations between this point and the previous.
            dx = x_it[i] - x_it[i - 1]
            dy = y_it[i] - y_it[i - 1]
            dz = z_it[i] - z_it[i - 1]

            # total change curve x w.r.t iterations between this point and the previous.
            ds = np.sqrt(dx * dx + dy * dy + dz * dz)

            ds_t.append(ds)

            # increase the total difference travelled
            s += ds

            i += 1

        #print('Found arc length {} at {} iterations'.format(s, i))
        #print('Corresponding positions ({},{},{})'.format(x_it[i], y_it[i], z_it[i]))

        # add the points to the time Trajectory
        x_t.append(x_it[i])
        y_t.append(y_it[i])
        z_t.append(z_it[i])
        s_t.append(s)

        # reset the arclength
        s = 0

    return np.array(x_t), np.array(y_t), np.array(z_t), np.array(s_t), np.array(ds_t)

# first vertex parameteric equation as function of iteration number
x0_it, y0_it, z0_it =  vertices_2_t[:, 0, 0], vertices_2_t[:, 0, 1], vertices_2_t[:, 0, 2]
x1_it, y1_it, z1_it =  vertices_2_t[:, 1, 0], vertices_2_t[:, 1, 1], vertices_2_t[:, 1, 2]
x2_it, y2_it, z2_it =  vertices_2_t[:, 2, 0], vertices_2_t[:, 2, 1], vertices_2_t[:, 2, 2]
x3_it, y3_it, z3_it =  vertices_2_t[:, 3, 0], vertices_2_t[:, 3, 1], vertices_2_t[:, 3, 2]

x4_it, y4_it, z4_it =  followers_2_t[:, 0, 0], followers_2_t[:, 0, 1], followers_2_t[:, 0, 2]
x5_it, y5_it, z5_it =  followers_2_t[:, 1, 0], followers_2_t[:, 1, 1], followers_2_t[:, 1, 2]
x6_it, y6_it, z6_it =  followers_2_t[:, 2, 0], followers_2_t[:, 2, 1], followers_2_t[:, 2, 2]
x7_it, y7_it, z7_it =  followers_2_t[:, 3, 0], followers_2_t[:, 3, 1], followers_2_t[:, 3, 2]

x0_t, y0_t, z0_t, s0_t, ds0_t = to_time_trajectory(x0_it, y0_it, z0_it)
x1_t, y1_t, z1_t, s1_t, ds1_t = to_time_trajectory(x1_it, y1_it, z1_it)
x2_t, y2_t, z2_t, s2_t, ds2_t = to_time_trajectory(x2_it, y2_it, z2_it)
x3_t, y3_t, z3_t, s3_t, ds3_t = to_time_trajectory(x3_it, y3_it, z3_it)

# vertex order
#[r2, r6, r7, r3]

# follower order
#[r1, r4, r8, r5]

r2 = np.stack([x0_t, y0_t, z0_t], axis=1)
r6 = np.stack([x1_t, y1_t, z1_t], axis=1)
r7 = np.stack([x2_t, y2_t, z2_t], axis=1)
r3 = np.stack([x3_t, y3_t, z3_t], axis=1)

print(r3.shape)

# now that trajectories are not synchronised in time the have different total times. However we still requried an end point.
# therefore we can pad the end of the trajectories with each end value.
it_max = np.array([r.shape[0] for r in [r2, r6, r7, r3]]).max()
print(it_max)

r2 = np.pad(r2, ((0, it_max - r2.shape[0]), (0, 0)), 'edge')
r6 = np.pad(r6, ((0, it_max - r6.shape[0]), (0, 0)), 'edge')
r7 = np.pad(r7, ((0, it_max - r7.shape[0]), (0, 0)), 'edge')
r3 = np.pad(r3, ((0, it_max - r3.shape[0]), (0, 0)), 'edge')

# generate the follower trajectories from the parameterised vertex trajectories.
r23 = r3 - r2
r26 = r6 - r2
l = np.linalg.norm(r26)
d = np.cross(r23, r26)
r1 = r2 + l * (d / np.linalg.norm(d))

r32 = r2 - r3
r37 = r7 - r3
d = np.cross(r37, r32)
l = np.linalg.norm(r37)
r4 = r3 + l * (d / np.linalg.norm(d))

r73 = r3 - r7
r76 = r6 - r7
d = np.cross(r76, r73)
l = np.linalg.norm(r76)
r8 = r7 + l * (d / np.linalg.norm(d))

r67 = r7 - r6
r62 = r2 - r6
d = np.cross(r62, r67)
l = np.linalg.norm(r62)
r5 = r6 + l * (d / np.linalg.norm(d))


# save the trajectory data
from pathlib import Path
dir = 'trajectories/'
Path(dir).mkdir(parents=True, exist_ok=True)

x0_t.tofile(dir + "x0_t.csv", sep="\n")
y0_t.tofile(dir + "y0_t.csv", sep="\n")
z0_t.tofile(dir + "z0_t.csv", sep="\n")
x1_t.tofile(dir + "x1_t.csv", sep="\n")
y1_t.tofile(dir + "y1_t.csv", sep="\n")
z1_t.tofile(dir + "z1_t.csv", sep="\n")
x2_t.tofile(dir + "x2_t.csv", sep="\n")
y2_t.tofile(dir + "y2_t.csv", sep="\n")
z2_t.tofile(dir + "z2_t.csv", sep="\n")
x3_t.tofile(dir + "x3_t.csv", sep="\n")
y3_t.tofile(dir + "y3_t.csv", sep="\n")
z3_t.tofile(dir + "z3_t.csv", sep="\n")

r1[:, 0].tofile(dir + "x4_t.csv", sep="\n")
r1[:, 1].tofile(dir + "y4_t.csv", sep="\n")
r1[:, 2].tofile(dir + "z4_t.csv", sep="\n")

r4[:, 0].tofile(dir + "x5_t.csv", sep="\n")
r4[:, 1].tofile(dir + "y5_t.csv", sep="\n")
r4[:, 2].tofile(dir + "z5_t.csv", sep="\n")

r8[:, 0].tofile(dir + "x6_t.csv", sep="\n")
r8[:, 1].tofile(dir + "y6_t.csv", sep="\n")
r8[:, 2].tofile(dir + "z6_t.csv", sep="\n")

r5[:, 0].tofile(dir + "x7_t.csv", sep="\n")
r5[:, 1].tofile(dir + "y7_t.csv", sep="\n")
r5[:, 2].tofile(dir + "z7_t.csv", sep="\n")


# 3d time history of the vertices
fig = plt.figure(1, constrained_layout=True)
gs = gridspec.GridSpec(1, 1, figure=fig)
ax = fig.add_subplot(gs[0], projection='3d')

it_step = 20
t_step = 10

custom_lines = [Line2D([0], [0], color='orange', lw=2),
                Line2D([0], [0], color='blue', lw=2),
                Line2D([0], [0], color='red', lw=2),
                Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='g')
                ]

ax.plot3D(x0_it[::it_step], y0_it[::it_step], z0_it[::it_step], color='red')#, s=2)# label='iterations')
ax.plot3D(x0_t[::t_step], y0_t[::t_step], z0_t[::t_step], color='orange')#, s=2)# label='time')
ax.plot3D(x1_it[::it_step], y1_it[::it_step], z1_it[::it_step], color='red')#, s=2)# label='iterations')
ax.plot3D(x1_t[::t_step], y1_t[::t_step], z1_t[::t_step], color='orange')#, s=2)# label='time')
ax.plot3D(x2_it[::it_step], y2_it[::it_step], z2_it[::it_step], color='red')#, s=2)# label='iterations')
ax.plot3D(x2_t[::t_step], y2_t[::t_step], z2_t[::t_step], color='orange')#, s=2)# label='time')
ax.plot3D(x3_it[::it_step], y3_it[::it_step], z3_it[::it_step], color='red')#, s=2)# label='iterations')
ax.plot3D(x3_t[::t_step], y3_t[::t_step], z3_t[::t_step], color='orange')#, s=2)# label='time')

ax.plot3D(r1[:, 0], r1[:, 1], r1[:, 2], color='blue')#, s=2)# label='follower')
ax.plot3D(r8[:, 0], r8[:, 1], r8[:, 2], color='blue')#, s=2)# label='follower')
ax.plot3D(r4[:, 0], r4[:, 1], r4[:, 2], color='blue')#, s=2)# label='follower')
ax.plot3D(r5[:, 0], r5[:, 1], r5[:, 2], color='blue')#, s=2)# label='follower')

ax.scatter(p[0], p[1], p[2], marker='x', label='Source', color='green')

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('x [m]')
# set scale to check if arclength is constant
#ax.set_xlim3d(-200, 30)
#ax.set_ylim3d(-200, 30)
#ax.set_zlim3d(-200, 30)
ax.legend(custom_lines, ['Lead', 'Follow', 'Lead iterations', 'Hostile'])
fig.savefig('parameterised_trajectory.png', dpi=330)


fig = plt.figure(2, constrained_layout=True)
gs = gridspec.GridSpec(1, 2, figure=fig)

t = np.linspace(0, 1, len(x0_t)) * len(x0_t) * 0.05
#plt.title('Trajectory in time')

ax = fig.add_subplot(gs[1])
ax.plot(t, x0_t, label='x position')
ax.plot(t, y0_t, label='y position')
ax.plot(t, z0_t, label='z position')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Displacement [m]')
ax.legend()

ax = fig.add_subplot(gs[0])
#plt.title('Trajectory in iterations')
ax.plot(x0_it, label='x position')
ax.plot(y0_it, label='y position')
ax.plot(z0_it, label='z position')

ax.set_xlabel('Iterations')
ax.set_ylabel('Displacement [m]')
ax.legend()

fig.savefig('trajectory_comparison.png', dpi=330)


fig = plt.figure(3, constrained_layout=True)
plt.title('ds')
plt.plot(ds0_t, label='particle 2')
plt.plot(ds1_t, label='particle 6')
plt.plot(ds2_t, label='particle 7')
plt.plot(ds3_t, label='particle 3')

print("target arc length: {}".format(arc_length))

plt.xlabel('iterations')
plt.ylabel('displacement [m]')
plt.legend()

fig = plt.figure(4, constrained_layout=True)
plt.title('s')
plt.plot(s0_t, label='particle 2')
plt.plot(s1_t, label='particle 6')
plt.plot(s2_t, label='particle 7')
plt.plot(s3_t, label='particle 3')

plt.xlabel('iterations')
plt.ylabel('displacement [m]')
plt.legend()

plt.show()
