# author: John Hartley

import numpy as np
from scipy.constants import epsilon_0
import math
import sys
import copy
import matplotlib.animation
import tqdm
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.lines import Line2D

target = np.array([200, 200, 200])
x_t = []

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

    dx = 1e-6

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

def minimise_flux():
    
    global target
    target = np.array([-200, 200, 200])
    
    surface = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 1]])

    x0 = surface.flatten(order='F')

    result = minimize(
        flux, x0=x0, method='trust-constr', tol=1e-6, options={'maxiter': 2000}, jac=jacobian_flux, callback=callback)
    
    print(result.x.reshape((4, 3), order='F'))


    points = result.x.reshape((4, 3), order='F')

    fig = plt.figure(1, constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    
    for i, p in enumerate(points):
        ax.scatter(p[0], p[1], p[2], label='point {} final'.format(i))

    #for i, p in enumerate(surface):
    #    ax.scatter(p[0], p[1], p[2], label='point {} initial'.format(i))

    # plot journey points
    for xj in x_t:
        surfacej = xj.reshape((4, 3), order='F')
        for p in surfacej:
            ax.scatter(p[0], p[1], p[2], s=2)
    
    ax.scatter(target[0], target[1], target[2], label='Target')

    ax.legend()
    plt.show()

minimise_flux()
