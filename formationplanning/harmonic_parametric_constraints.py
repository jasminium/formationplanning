"""
FormationPlanning
Copyright (C) 2020 : Northumbria University
            Author : John Hartley

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
from scipy.constants import epsilon_0
import math
import sys
import copy
import matplotlib.animation
import tqdm

class Particles():

    def __init__(self, d, h, target):
        r1 = np.array([0, 0, 0 + h], dtype=np.float64)
        r2 = np.array([d, 0, 0 + h], dtype=np.float64)
        r3 = np.array([d, 0, d + h], dtype=np.float64)
        r4 = np.array([0, 0, d + h], dtype=np.float64)

        r5 = np.array([0, d, 0 + h], dtype=np.float64)
        r6 = np.array([d, d, 0 + h], dtype=np.float64)
        r7 = np.array([d, d, d + h], dtype=np.float64)
        r8 = np.array([0, d, d + h], dtype=np.float64)

        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4
        self.r5 = r5
        self.r6 = r6
        self.r7 = r7
        self.r8 = r8

        # position of the source
        self.p = target
        self.vertices = [r1, r2, r3, r4, r5, r6, r7, r8]

        self.triangle_list = [
                            [r1, r2, r3],
                            [r1, r3, r4],
                            [r5, r1, r4],
                            [r5, r4, r8],
                            [r4, r3, r7],
                            [r4, r7, r8],
                            [r5, r6, r2],
                            [r5, r2, r1],
                            [r6, r5, r8],
                            [r6, r8, r7],

                            # closing
                            #[r2, r6, r7],
                            #[r2, r7, r3]
                        ]

        r23 = r3 - r2
        r26 = r6 - r2
        r1 = r2 + np.cross(r23, r26)

        r32 = r2 - r3
        r37 = r7 - r3
        r4 = r3 + np.cross(r37, r32)

        r73 = r3 - r7
        r76 = r6 - r7
        r8 = r7 + np.cross(r76, r73)

        r67 = r7 - r6
        r62 = r2 - r6
        r5 = r6 + np.cross(r62, r67)


        # these are the vertices of the right face.
        self.vertices_2 = [r2, r6, r7, r3]
        # these are the vertices of the left face
        self.vertices_3 = [r1, r4, r8, r5]

        #self.vertices_init = copy.deepcopy(vertices_2)
        #self.vertices_init_back = copy.deepcopy(vertices_3)

        # closing triangles
        self.triangle_list_2 = [
                            [r2, r7, r6],
                            [r2, r3, r7]
        ]

def face_centre(particles):
    return 1 / 2 * (particles.r7 - particles.r2)


def calculate_jacobian_2(particles):
    # calculate the jacobian in component order ie.

    # [d phi / d c1_x
    # [d phi / d c2_x
    # [d phi / d c3_x
    # [d phi / d c4_x
    # [d phi / d c1_y
    # [d phi / d c2_y
    # [d phi / d c3_y
    # [d phi / d c4_y
    # [d phi / d c1_z
    # [d phi / d c2_z
    # [d phi / d c3_z
    # [d phi / d c4_z

    vertices = particles.vertices_2

    # find the jacobian using a central difference

    J = []
    # total derivative of phi
    d_phi = []

    dc = 1e-4

    # iterate through each component
    for i in [0, 1, 2]:

        # iterate the the vertices
        for j, vertex in enumerate(vertices):

            c_i = vertex[i]

            vertex[i] = c_i + dc
            d_u = total_flux(particles.triangle_list_2, particles.p)

            vertex[i] = c_i - dc
            d_l = total_flux(particles.triangle_list_2, particles.p)

            # return vertex to previous state
            vertex[i] = c_i

            # change in phi over dc
            d_phi_c_i = (d_u - d_l)
            # phi derivative wrt c_i
            # central
            #d_phi_d_c_i =  (d_u - d_l) / 2 / dc

            # forward difference
            d_phi_d_c_i =  (d_u - d_l) / 2 /  dc

            J.append(d_phi_d_c_i)
            d_phi.append(d_phi_c_i)


    #print('Jacobian', J)
    #print('phi: total derivative', sum(d_phi))
    return J, sum(d_phi)
    #(total_flux(triangle_list + delta_c) + total_flux(triangle_list - delta_c)) / 2 / delta_c


def flux_through_t(A, B, C, P):
    a = P - A
    b = P - B
    c = P - C

    num = np.dot(np.cross(a, b), c)

    a_m = np.sqrt(a.dot(a))
    b_m = np.sqrt(b.dot(b))
    c_m = np.sqrt(c.dot(c))

    de = a_m * b_m * c_m + np.dot(a, b) * c_m + np.dot(a, c) * b_m + np.dot(b, c) * a_m
    phi = math.atan2(num, de) * 2
    phi *= -1
    #print('phi triangle', phi)
    return phi

def total_flux(triangle_list, p):
    f = 0
    for t in triangle_list:
        f = f + flux_through_t(t[0], t[1], t[2], p)

    return f

def solve_constraints(particles, iterations, constraints_w):

    alpha = constraints_w[0]
    beta = constraints_w[1]

    # time history variables
    phi_t = []
    phi_ref_t = []
    d_phi_t = []

    vertices_2_t = np.zeros((iterations, 4, 3), dtype=np.float64)
    followers_2_t = np.zeros((iterations, 4, 3), dtype=np.float64)

    r2_t = []

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

    n_vertices = len(particles.vertices_2)
    # number of degrees of freedom
    n = n_vertices * 3

    phi = 0
    phi_max = 4.5

    d_phi_p = 0

    alpha_max = 1000

    for i in tqdm.tqdm(range(0, iterations)):

        #alpha = alpha_max / np.linalg.norm(p - ((r7 + r2) * 1 / 2))

        # flux through the left face of the bounding box
        phi = total_flux(particles.triangle_list_2, particles.p)
        #if phi > 4.5:
        #    break
        phi_ref = np.exp(0.1 * phi)
        #phi_ref = 0.1
        #phi_ref = 1 / phi

        J, d_phi = calculate_jacobian_2(particles)

        J = np.array(J)

        a = np.identity(n) + (alpha * np.outer(J, J)) + (beta * np.matmul(AT, A))
        b = alpha * J * phi_ref

        delta_c = np.linalg.solve(a, b)
        delta_c = delta_c.reshape((n_vertices, 3), order='F')

        # update the vertices
        for j, vertex in enumerate(particles.vertices_2):
            vertex += delta_c[j]

        r23 = particles.r3 - particles.r2
        r26 = particles.r6 - particles.r2
        l = np.linalg.norm(r26)
        d = np.cross(r23, r26)
        particles.r1 = particles.r2 + l * (d / np.linalg.norm(d))

        r32 = particles.r2 - particles.r3
        r37 = particles.r7 - particles.r3
        d = np.cross(r37, r32)
        l = np.linalg.norm(r37)
        particles.r4 = particles.r3 + l * (d / np.linalg.norm(d))

        r73 = particles.r3 - particles.r7
        r76 = particles.r6 - particles.r7
        d = np.cross(r76, r73)
        l = np.linalg.norm(r76)
        particles.r8 = particles.r7 + l * (d / np.linalg.norm(d))

        r67 = particles.r7 - particles.r6
        r62 = particles.r2 - particles.r6
        d = np.cross(r62, r67)
        l = np.linalg.norm(r62)
        particles.r5 = particles.r6 + l * (d / np.linalg.norm(d))

        # save time history
        phi_t.append(phi)
        phi_ref_t.append(phi_ref)
        d_phi_t.append(d_phi)

        # save time history of vertices data
        vertices_2_t[i] = np.array(particles.vertices_2)
        followers_2_t[i] = np.array([particles.r1, particles.r4, particles.r8, particles.r5])

        # dont allow the centre of the front face to come within 10cm of the point
        if np.linalg.norm(particles.p - ((particles.r7 + particles.r2) * 1 / 2)) < 10:
            break

    return phi_t, phi_ref_t , d_phi_t, vertices_2_t, followers_2_t, i
