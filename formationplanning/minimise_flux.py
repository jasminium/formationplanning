# author: John Hartley

import numpy as np
import math
import matplotlib.animation
from scipy.optimize import minimize
from scipy.optimize import BFGS
from scipy.optimize import SR1
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import matplotlib.animation
import time

target = np.array([200, 200, 200])
x_t = []

s = 5

def flux_through_triangle(A, B, C, P):
    a = P - A
    b = P - B
    c = P - C

    num = np.dot(np.cross(a, b), c)

    a_m = np.sqrt(a.dot(a))
    b_m = np.sqrt(b.dot(b))
    c_m = np.sqrt(c.dot(c))

    de = a_m * b_m * c_m + np.dot(a, b) * c_m + np.dot(a, c) * b_m + np.dot(b, c) * a_m
    phi = math.atan2(num, de) * 2

    #print('phi triangle', phi)
    return phi

def flux(x):

    """
        Evaluate the flux through a surface.
    """

    # 12 dimension

    # order = [r2, r6, r7, r3]

    # map the solution space back to the vector points
    p = x.reshape((4, 3), order='F')

    # surfaces to evaluate flux
    triangle_list = [
        # r2, r6, r7
        [0, 1, 2],
        # r2, r7, r3
        [0, 2, 3]
    ]

    # flux count
    phi = 0

    for i, tri in enumerate(triangle_list):
        # get triangle vertices
        r1 = p[tri[0]]
        r2 = p[tri[1]]
        r3 = p[tri[2]]
        # evaluate the flux through the triangle
        phi_i = flux_through_triangle(r1, r2, r3, target)
        phi += phi_i

    return phi


def jacobian_flux(x):
    
    # jacobian
    j = np.zeros(12)

    dx = 1e-4

    for i, x_i in enumerate(x):

        # evaluate the derivate using a
        # central finite difference.

        x_i_u = x_i + dx
        x_i_l = x_i - dx

        x[i] = x_i_u
        flux_u = flux(x)

        x[i] = x_i_l
        flux_l = flux(x)

        # derivative
        j_x = (flux_u - flux_l) / 2 / dx
        j[i] = j_x

        # recover original position
        x[i] = x_i
    
    return j

def test_flux():

    global target
    target = np.array([5, 0.5, 5])

    surface = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 1],
        [0, 0, 1]])
    
    x = surface.flatten(order='F')
    print('x', x)
    phi = flux(x)
    print('phi', phi)
    
def callback(xk, state):
    x_t.append(xk)
    return False

def cons_f(x):
    """"constrain the connecting edges"""
    p = x.reshape((4, 3), order='F')

    a = p[0] - p[1]
    b = p[1] - p[2]
    c = p[2] - p[3]
    d = p[3] - p[0]

    return [
        a.dot(a),
        b.dot(b),
        c.dot(c),
        d.dot(d)
    ]

def cons_h(x, v):
    return np.zeros(12)

def minimise_flux():
    
    global target
    target = np.array([10, 10, 10])
    
    surface = np.array([
            [0, 0, 0],
            [0, 5, 0],
            [0, 5, 5],
            [0, 0, 5]])

    # "Starting value guess"
    x0 = surface.flatten(order='F')

    nonlinear_constraint = NonlinearConstraint(cons_f, 5**2, 5**2, jac='2-point', hess=BFGS())

    result = minimize(
        flux, x0=x0,
        method='trust-constr',
        tol=1e-8,
        options={'maxiter': 10000,
                 'initial_tr_radius': 1e-3
                },
        jac=jacobian_flux,
        callback=callback,
        constraints=[nonlinear_constraint])
    
    print(result)

    points = result.x.reshape((4, 3), order='F')

    # plot trajectories
    fig = plt.figure(1, constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')

    colors_init = ['pink', 'green', 'red', 'blue']
    colors_final = ['yellow', 'yellow', 'yellow', 'green']
    cmaps = ['spring', 'summer', 'autumn', 'winter']

    x_j = np.reshape(x_t, (len(x_t), 4, 3), order='F')
    c_s = np.arange(0, x_j.shape[0])

    # plot positions at each iteration
    for i in range(4):
        ax.scatter(x_j[:, i, 0], x_j[:, i, 1], x_j[:, i, 2], c=c_s, cmap=cmaps[i], s=4)

    # plot the staring points
    for i, p in enumerate(surface):
        ax.scatter(p[0], p[1], p[2], s=60, linewidth=1, color=colors_init[i], label='point {} initial'.format(i))

    ax.scatter(target[0], target[1], target[2], marker='x', s=20, color='purple', label='Target')

    ax.legend()
    plt.show()

minimise_flux()
