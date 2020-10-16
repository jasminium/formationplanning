import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from scipy.optimize import BFGS
from scipy.optimize import NonlinearConstraint
from numpy import inf
from numpy import nan


def trajectory_example():
    x_t = np.load('x_t.npy')

    x_1 = x_t[:, 0, :]

    t = np.zeros(x_1.shape[0])
    v_i = np.array((0, 0, 0))

    for i in range(x_1.shape[0] - 1):
        t_i, v_i = test_traj_min_v3(x_1[i], x_1[i+1], v_i)
        t[i+1] = t_i

    t = np.cumsum(t)

    print(x_1.shape[0])
    print(t.shape[0])

    fig = plt.figure(1)
    plt.plot(t)
    plt.show()

    from formationplanning.trajectory import interpolate_trajectory


    print('t', t[:10])

    t_i, x_t_i = interpolate_trajectory(t, x_t)

    fig = plt.figure(1)
    plt.plot(t)
    plt.show()

    x0 = np.array([0, 0])
    x1 = dynamics_1d(x0, 0.01, x_t_i[:, 0, 0]) # component of the first drone
    x0 = np.array([5, 0])
    y1 = dynamics_1d(x0, 0.01, x_t_i[:, 0, 1]) # component of the first drone
    x0 = np.array([0, 0])
    z1 = dynamics_1d(x0, 0.01, x_t_i[:, 0, 2]) # component of the first drone

    print('x0 shape', x1.shape)

    # animation
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')

    xmin = x_t[:, :, 0].min()
    xmax = x_t[:, :, 0].max()
    ymin = x_t[:, :, 1].min()
    ymax = x_t[:, :, 1].max()
    zmin = x_t[:, :, 2].min()
    zmax = x_t[:, :, 2].max()

    #c = np.arange(x1.size)
    ax.scatter(x1, y1, z1)
    #ax.plot(x_r[:, 0, 0], x_r[:, 0, 1], x_r[:, 0, 2])
    ax.scatter(x_t_i[:, 0, 0], x_t_i[:, 0, 1], x_t_i[:, 0, 2])
    print(x_t_i.shape)
    plt.show()

def dynamics_1d(x0, dt, x_r, plot=True):

    print('dynamics x0', x0)
    print('dynamics dt', dt)
    print('dynamics xr', x_r.shape, x_r[:10])

    k = 20
    A = np.array([[1, dt],[0, 1]])
    B = np.array([0.5 * dt, dt])
    
    x_t = []
    v_t = []
    u_t = []

    for i in range(x_r.shape[0]):
        u = k * (x_r[i] - x0[0])
        x0 = A.dot(x0) + B.dot(u) 
        x_t.append(x0[0])
        v_t.append(x0[1])
        u_t.append(u)
    t = np.arange(0, x_r.size) * dt
    if plot:
        plt.figure(1)
        plt.plot(t, x_r, label='ref')
        plt.plot(t, x_t, label='sim')
        plt.legend()

        plt.figure(2)
        plt.plot(t, v_t)
        plt.title('v')

        plt.figure(3)
        plt.plot(u_t)
        plt.title('u_t')
        plt.show()

    return np.array(x_t)

def dynamics_example():
    x_r = np.load('x_t_in.npy')
    x0 = np.array([0, 0])
    x1 = dynamics_1d(x0, 0.1, x_r[:, 0, 0]) # component of the first drone
    x0 = np.array([5, 0])
    y1 = dynamics_1d(x0, 0.1, x_r[:, 0, 1]) # component of the first drone
    x0 = np.array([0, 0])
    z1 = dynamics_1d(x0, 0.1, x_r[:, 0, 2]) # component of the first drone

    # animation
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')

    xmin = x_r[:, :, 0].min()
    xmax = x_r[:, :, 0].max()
    ymin = x_r[:, :, 1].min()
    ymax = x_r[:, :, 1].max()
    zmin = x_r[:, :, 2].min()
    zmax = x_r[:, :, 2].max()

    c = np.arange(x1.size)
    ax.scatter(x1, y1, z1)
    ax.plot(x_r[:, 0, 0], x_r[:, 0, 1], x_r[:, 0, 2])
    plt.show()

def test_traj_min(x0, x1, v0):
    
    # x1 starting point
    # x2 ending point
    # starting velocity
    # print(p1, p2, v1)

    # performance constraints
    v_max = 10 # max velocity
    a_max = 10 # max acceleration

    # guess - see notes
    if x1 - x0 < 0:
        v1 = -1
        v1 = (v0 * -1) - 1
    else:
        #v1 = 1
        v1 = (v0 * -1) + 1

    def obj(v1):
        # objective function is the time to travel between two points
        t = (2 * (x1 - x0)) / (v0 + v1[0])
        return t

    def cons_a(v1):
        # contraint on acceleration
        a = (v1[0]**2 - v0**2) / 2 / (x1 - x0)

        return [
            a
        ]
    
    def cons_v(v1):
        # contraint on velocity
        return v1

    def callback(xk, state):
        pass
    
    nonlinear_constraint_1 = NonlinearConstraint(cons_a, 0, a_max, jac='2-point', hess=BFGS())
    nonlinear_constraint_2 = NonlinearConstraint(cons_v, 0, v_max, jac='2-point', hess=BFGS())

    res = minimize(
        obj, x0=v1,
        method='trust-constr',
        options={
                #'xtol': 1e-12,
                 'maxiter': 20000,
                 'disp':False,
                 'verbose': 2,
                },
        #jac=jacobian_flux,
        callback=callback,
        constraints=[
            # sides must have some minimum length
            nonlinear_constraint_1,
            nonlinear_constraint_2
        ])
    
    dt = res.fun
    v1 = res.x[0]
    a = (v1 - v0) / dt

    # results

    # minimum dt
    print('dt', dt)
    # predicted final velocity
    print('v1', v1)
    # acceleration over the step
    print('acceleration', (v1 - v0) / dt)
    # distance travelled
    print('dx', x1 - x0, dt / 2 * (v1 + v0))
    #print('a', a_max, a)
    # v_2
    #print('v_2 vector', v2 * v_2_n)

    #print(dt, v2, a)

    return dt, v1

def test_traj_min_v3(x0, x1, v0):
    # p1 starting point
    # p2 ending point
    # starting velocity
    #print(p1, p2, v1)

    import sys

    dx = x1 - x0 + 1e-6

    # come up with an initial guess for v1
    # have to careful as some solution are non-physical

    sign = np.sign(dx)
    # condition x0==x1 falls under x1 - x0 >= 0
    
    sign[sign==0] = 1

    v1 = (v0 * -1) + sign * 1

    print('v0', v0)
    print('v1 guess', v1)

    # performance constraints
    v_max = 5 # max velocity
    a_max = 5 # max acceleration

    def obj(v1):
        # objective function is the time to travel between two points
        #v_s = v0 + v1 # velocity sum
        #dt = dx / v_s
        #dt = np.nan_to_num(dt) 
        #t = np.sum(dt) * 2 / 3
        v_av = (v0 + v1) / 2
        t = np.linalg.norm(dx) / np.linalg.norm(v_av)
        a = (v1**2 - v0**2) / 2 / dx


        #print('obj v average', v_av)
        #print('obj v_average mag', np.linalg.norm(v_av))
        #print('obj t', t)
        #print('obj a', a)
        #print('obj a mag', np.linalg.norm(a))

        return t

    def cons_a(v1):
        
        # calculate the acceleration
        a = (v1**2 - v0**2) / 2 / dx

        # where there is no position change the acceleration calculation is infinite.
        # however no change in position is also just 0 acceleration
        #a[a==inf] = 0

        #a = (v1 - v0) / obj(v1)
        return np.linalg.norm(a)
        
    def cons_dx(v1):
        return (v1 + v0) / 2 * obj(v1)


    def cons_v(v1):
        # contraint on velocity
        return np.linalg.norm(v1)

    def callback(xk, state):
        pass
        #print('callback', xk)
    
    # accleration performance constraint
    nonlinear_constraint_1 = NonlinearConstraint(cons_a, 0, a_max, jac='2-point', hess=BFGS())
    # velocity performance constraint
    nonlinear_constraint_2 = NonlinearConstraint(cons_v, 0, v_max, jac='2-point', hess=BFGS())
    
    nonlinear_constraint_3 = NonlinearConstraint(cons_dx, dx, dx, jac='2-point', hess=BFGS())


    res = minimize(
        obj, x0=v1,
        method='trust-constr',
        options={
                 #'xtol': 1e-12,
                 'maxiter': 20000,
                 'disp': True,
                 'verbose': 0,
                },
        #jac=jacobian_flux,
        callback=callback,
        constraints=[
            # sides must have some minimum length
            nonlinear_constraint_1,
            nonlinear_constraint_2,
            nonlinear_constraint_3
        ])
    
    dt = res.fun
    v1 = res.x

    #print('dt', obj(v1))
    print('result v mag', np.linalg.norm(v1))
    print('result a mag', cons_a(v1))
    #print('\n')
    #print('v1', v1)

    print('result dt', dt)
    print('result v1', v1)
    print('result dx', (v1 + v0) / 2 * dt)

    return dt, v1

x0 = np.array((0, 0, 0))
x1 = np.array((1, -2, 1))

v0 = np.array((1, 0, -1))

test_traj_min_v3(x0, x1, v0)

print('test dx', x1 - x0)
