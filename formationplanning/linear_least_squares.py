import numpy as np
from scipy.constants import epsilon_0
import math
import sys
import copy
import matplotlib.animation
import tqdm
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint


# flux constraint
alpha = 1000
# connectivity constraint
beta = 400

iterations = 200000

target = []
x_t = []
area_t = []
phi_t = []

def face_centre(particles):
    return 1 / 2 * (particles.r7 - particles.r2)


def jacobian_flux(x):
    """
        Evaluate the jacobian of the flux through the surface x.

        Evaluating the jacobian is significantly faster than using
        an automatic differentiation provided by numpy.

        The jacobian is evaluated using a central finite difference

    Parameters
    ----------
    x : np.array((12))
        Fortran order list of coordinates for the surface

    Returns
    -------
    np.array(12)
        The flux jacobian
    """

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

def f(x):
    """simple quadratic"""
    return x.T.dot(x)

def f_jac(x):
    return 2 * x


def flux_through_triangle(A, B, C, P):
    """
        Evaluate the solid angle illuminating a triangle.

        The triangle is defined by defined A, B, C by a point P.
    """

    """Gets and prints the spreadsheet's header columns

    Parameters
    ----------
    A : np.array((3))
        Vertex A of the triangle

    B : np.array((3))
        Vertex B of the triangle

    C : np.array((3))
        Vertex C of the triangle

    P : np.array((3))
        Source

    Returns
    -------
    float
        The value of the solid angle.
    """
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
    return phi * -1

def flux(x):
    """
        Evaluate the Flux through the surface x.

    Parameters
    ----------
    x : np.array((12))
        Fortran order list of coordinates for the surface

    Returns
    -------
    float
        The value of the solid angle.
    """

    # map the solution space back to the vector points.
    p = x.reshape((4, 3), order='F')

    # Order of vertices [r2, r6, r7, r3]

    # define the triangulation for the surface.
    triangle_list = [
        # r2, r6, r7
        [0, 2, 1],
        # r2, r7, r3
        [0, 3, 2]
    ]

    # total flux through triangulated surface
    phi = 0

    # evaluate the flux through each triangle.
    for i, tri in enumerate(triangle_list):
        # get triangle vertices
        r1 = p[tri[0]]
        r2 = p[tri[1]]
        r3 = p[tri[2]]
        # evaluate the flux through the triangle.
        phi_i = flux_through_triangle(r1, r2, r3, target)
        phi += phi_i

    return phi

def solve_constraints(targets, surface, l=None):

    global target
    global x_t
    global phi_t

    # Reset surface, and flux history.
    x_t = []
    phi_t = []

    # "Starting value guess"
    x0 = surface.flatten(order='F')

    target = targets

    # Create A matrix of length constraints
    A_s = np.array([
                  [1, -1, 0, 0],
                  [0, 1, -1, 0],
                  [0, 0, 1, -1],
                  [-1, 0, 0, 1],
                 ])

    A = np.zeros((12, 12))
    A[0:4,0:4] = A_s
    A[4:8,4:8] = A_s
    A[8:, 8:] = A_s
    AT = np.transpose(A)

    A_c = np.zeros((12))

    lc_1 = LinearConstraint(A, A_c, A_c)

    # inequality constraints on velocity
    # time step
    dt = 0.05
    # max velocity
    v = 3
    B = np.identity(12)
    # lower bound for the velocity
    B_l = np.ones((12)) * -1 * v * dt
    # upper bound for the velocity
    B_u = np.ones((12)) * 1 * v * dt
    lc_2 = LinearConstraint(B, B_l, B_u)

    n_vertices = surface.shape[0]
    # number of degrees of freedom
    n = n_vertices * 3

    phi = 0
    phi_max = 4.5

    d_phi_p = 0

    alpha_max = 1000

    for i in tqdm.tqdm(range(0, iterations)):

        #alpha = alpha_max / np.linalg.norm(p - ((r7 + r2) * 1 / 2))

        # flux through the left face of the bounding box
        phi = flux(x0)
        #if phi > 4.5:
        #    break
        phi_ref = np.exp(0.1 * phi)
        #phi_ref = 0.1
        #phi_ref = 1 / phi

        J = jacobian_flux(x0)
        J = np.array(J)
        #print(phi, d_phi)

        # least square approach
        a = np.identity(n) + (alpha * np.outer(J, J)) + (beta * np.matmul(AT, A))
        b = alpha * J * phi_ref
        delta_c = np.linalg.solve(a, b)

        # save a copy of the positions
        x_t.append(np.copy(x0))

        # update verticles
        x0 = x0 + delta_c

        # dont allow the centre of the front face to come within 10cm of the point
        x0_f = x0.reshape((4, 3), order='F')
        if np.linalg.norm(target - ((x0_f[2] + x0_f[0]) * 1 / 2)) < 2:
            break

    x_j = np.reshape(x_t, (len(x_t), 4, 3), order='F')

    return x_j, phi_t
