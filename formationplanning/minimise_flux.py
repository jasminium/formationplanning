# author: John Hartley

import numpy as np
import math
import matplotlib.animation
from scipy.optimize import minimize
from scipy.optimize import BFGS
from scipy.optimize import SR1
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import matplotlib.animation
import time

target = np.array([200, 200, 200])
x_t = []

s = 5


def curve_fit_model(x, a, b, c, d, e, f, g, h):

    p = [a, b, c, d, e, f, g, h]
    c = 0
    for i, p_i in enumerate(p):
        c += p_i * np.power(x, i)

    return c


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

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
                 'initial_tr_radius': 1
                },
        jac=jacobian_flux,
        callback=callback,
        constraints=[nonlinear_constraint])
    
    print(result)

    points = result.x.reshape((4, 3), order='F')

    # plot trajectories
    #fig = plt.figure(1, constrained_layout=True)
    #ax = fig.add_subplot(111, projection='3d')

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    colors_init = ['pink', 'green', 'red', 'blue']
    colors_final = ['yellow', 'yellow', 'yellow', 'green']
    cmaps = ['spring', 'summer', 'autumn', 'winter']

    x_j = np.reshape(x_t, (len(x_t), 4, 3), order='F')
    c_s = np.arange(0, x_j.shape[0])

    # plot in the browser
    import plotly.graph_objects as go

    style_3 =dict(
        size=2,
        color=c_s,               # set color to an array/list of desired values
        colorscale='rainbow',   # choose a colorscale
        opacity=0.8
    )

    fig = go.Figure(data=[
        go.Scatter3d(x=x_j[:, 0, 0], y=x_j[:, 0, 1], z=x_j[:, 0, 2], marker=style_3, name='Drone 1'),
        go.Scatter3d(x=x_j[:, 1, 0], y=x_j[:, 1, 1], z=x_j[:, 1, 2], marker=style_3, name='Drone 2'),
        go.Scatter3d(x=x_j[:, 2, 0], y=x_j[:, 2, 1], z=x_j[:, 2, 2], marker=style_3, name='Drone 3'),
        go.Scatter3d(x=x_j[:, 3, 0], y=x_j[:, 3, 1], z=x_j[:, 3, 2], marker=style_3, name='Drone 4')
        ])
    

    fig.update_layout(
        title="Drone Trajectories",

        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    fig.show()

    fig.write_html("optimsation.html")
    
minimise_flux()
