# author: John Hartley

# script to compute the trajectories of formation of UAV using a flux maximisation technique.

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
from tqdm import tqdm

target = np.array([200, 200, 200])
x_t = []

# max speed in each direction
s = 5
# controller time step
dt = 0.05

# starting guesses for x
p0 = np.zeros([0, 0, 0])
p1 = np.zeros([0, 0, 0])
p2 = np.zeros([0, 0, 0])
p3 = np.zeros([0, 0, 0])

def flux_through_triangle(A, B, C, P):
    """Compute the flux through a triangle with vertices A, B, C
       From a point P."""
    a = P - A
    b = P - B
    c = P - C

    num = np.dot(np.cross(a, b), c)

    a_m = np.sqrt(a.dot(a))
    b_m = np.sqrt(b.dot(b))
    c_m = np.sqrt(c.dot(c))

    de = a_m * b_m * c_m + np.dot(a, b) * c_m + np.dot(a, c) * b_m + np.dot(b, c) * a_m
    phi = math.atan2(num, de) * 2

    return phi

def flux(x):

    """
        Evaluate the flux through a surface surface formation by the formation.
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
    """calculate the jacobian of the flux about the 12d point, x using finite differences"""

    j = np.zeros(12)

    # function appears very smooth so can use very small delta
    dx = 1e-8

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

        # recover the original position
        x[i] = x_i
    
    return j
    
def callback(xk, state):
    #x_t.append(xk)
    return False

def cons_f(x):
    """"Constrain the length connecting edges. Maintain intital formation"""
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

def cons_angle(x):
    """"constrain the length connecting edges"""
    p = x.reshape((4, 3), order='F')

    a = p[0] - p[1]
    b = p[1] - p[2]
    c = p[2] - p[3]
    d = p[3] - p[0]

    a_n = np.linalg.norm(a)
    b_n = np.linalg.norm(b)
    c_n = np.linalg.norm(c)
    d_n = np.linalg.norm(d)

    return [
        np.arccos(a.dot(b) / a_n / b_n),
        np.arccos(b.dot(c) / b_n / c_n),
        np.arccos(c.dot(d) / c_n / d_n),
        np.arccos(d.dot(a) / d_n / a_n)
    ]

def cons_delta(x):
    """"constrain the abs distance travelled over the minimisation"""
    p = x.reshape((4, 3), order='F')

    a = p0 - p[0]
    b = p1 - p[1]
    c = p2 - p[2]
    d = p3 - p[3]

    return [
        a.dot(a),
        b.dot(b),
        c.dot(c),
        d.dot(d)
    ]


def cons_h(x, v):
    return np.zeros(12)

def solve_normal_equation(A, B):
    AT = A.transpose()
    inv = np.linalg.inv(np.matmul(AT, A))
    p = np.matmul(inv, AT)
    co = np.matmul(p, B)
    return co

def add_least_square_surface(fig, x_t, i):
    # fit a plane to the the end points of the trajectory. Use normal equation to solve.
    # z = px + qy + r
    i = int(i)
    A = np.array([x_t[i, :, 0], x_t[i, :, 1], [1, 1, 1, 1]], dtype=np.float64).transpose()
    B = x_t[i, :, 2]
    co = solve_normal_equation(A, B)
    
    # plot the plane. Limits are given by the bounds of the end point
    xval = np.linspace(np.amin(x_t[i, :, 0]), np.amax(x_t[i, :, 0]), 100)
    yval = np.linspace(np.amin(x_t[i, :, 1]), np.amax(x_t[i, :, 1]), 100)
    X, Y = np.meshgrid(xval, yval)
    Z = co[0] * X + co[1] * Y + co[2]

    fig.add_surface(x=xval, y=yval, z=Z, opacity=0.3)

def plot(x_t):
    """Plot the trajectory"""
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

    add_least_square_surface(fig, x_j, -1)

    fig.update_layout(
        title="Drone Trajectories",

        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    fig.show()

    fig.write_html("sequential_non_linear.html")

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
    
    iterations = 100

    for i in tqdm(range(iterations)):

        global p0
        global p1
        global p2
        global p3

        p0 = x0.reshape((4, 3), order="F")[0]
        p1 = x0.reshape((4, 3), order="F")[1]
        p2 = x0.reshape((4, 3), order="F")[2]
        p3 = x0.reshape((4, 3), order="F")[3]

        # constraint on the edge lengths
        nonlinear_constraint = NonlinearConstraint(cons_f, 5**2, 5**2, jac='2-point', hess=BFGS())
        # contraint on the radius of the final points from the intial guess
        nonlinear_constraint_2 = NonlinearConstraint(cons_delta, 0, (s * dt) ** 2, jac='2-point', hess=BFGS())

        # angles constraint (not working)
        nonlinear_constraint_3 = NonlinearConstraint(cons_angle, np.pi / 4, np.pi / 4, jac='2-point', hess=BFGS())

        result = minimize(
            flux, x0=x0,
            method='trust-constr',
            tol=1e-8,
            options={'maxiter': 1000},
            jac=jacobian_flux,
            callback=callback,
            constraints=[nonlinear_constraint, nonlinear_constraint_2])
        
        points = result.x.reshape((4, 3), order='F')
        x_t.append(result.x)
        x0 = result.x

    plot(x_t)
    
    
minimise_flux()
