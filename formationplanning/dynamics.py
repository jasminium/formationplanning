import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from scipy.optimize import BFGS
from scipy.optimize import NonlinearConstraint


def dynamics_1d(x0, dt, x_r, plot=False):
    
    k = 20
    A = np.array([[1, dt],[0, 1]])
    B = np.array([0.5 * dt, dt])
    
    x_t = []
    v_t = []
    u_t = []

    for i in range(x_r.shape[0]):
        u = k * (x_r[i] - x0[0])
        if u > 2:
            u = 2
        if u < 2:
            u = -2
        x0 = A.dot(x0) + B.dot(u) 
        x_t.append(x0[0])
        v_t.append(x0[1])
        u_t.append(u)

    if plot:
        plt.figure(1)
        plt.plot(x_r, label='ref')
        plt.plot(x_t, label='sim')
        plt.legend()

        plt.figure(2)
        plt.plot(v_t)
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

def test_traj_min():
    
    # change in position
    dx = -10
    # initial velocity
    v1 = 1
    # performance constraints
    v_max = 4 # max velocity
    a_max = 0.5 # max acceleration

    def obj(v2):
        # objective function is the time to travel between two points
        t = 2 * dx / (v1 + v2)
        return t

    def cons_a(x):
        # contraint on acceleration
        a = (x[0]**2 - v1**2) / 2 / dx

        return [
            np.abs(a)
        ]
    
    def cons_v(x):
        # contraint on velocity
        return np.abs(x)

    def callback(xk, state):
        pass
    
    nonlinear_constraint_1 = NonlinearConstraint(cons_a, -np.inf, a_max, jac='2-point', hess=BFGS())
    nonlinear_constraint_2 = NonlinearConstraint(cons_v, -np.inf, v_max, jac='2-point', hess=BFGS())

    x0 = np.array([-10])

    res = minimize(
        obj, x0=x0,
        method='trust-constr',
        options={
                #'xtol': 1e-12,
                 'maxiter': 20000,
                 'disp':True,
                 'verbose': 2,
                },
        #jac=jacobian_flux,
        callback=callback,
        constraints=[
            # sides must have some minimum length
            nonlinear_constraint_1,
            nonlinear_constraint_2
        ])
    
    dt = res.fun[0]
    v2 = res.x[0]

    # results

    # minimum dt
    print('dt', dt)
    # predicted final velocity
    print('v2', v2)
    # acceleration over the step
    print('acceleration', (v2 - v1) / dt)
    # distance travelled
    print('dx', dt / 2 * (v2 + v1))

test_traj_min()