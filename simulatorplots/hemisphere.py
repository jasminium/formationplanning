import numpy as np

def hemisphere_points(r, origin, theta_r=0):

    x = []
    y = []
    z = []
    # generate hemisphere position set.
    theta = np.linspace(0.5 * np.pi, 3 / 2 * np.pi, 10)

    # sphere
    #theta = np.linspace(0, 2 * np.pi, 20)
    phi = np.linspace(0, np.pi, 10)

    for theta_i in theta:
        for phi_i in phi:

            x_i = r * np.cos(theta_i) * np.sin(phi_i)
            y_i = r * np.sin(theta_i) * np.sin(phi_i)
            z_i = r * np.cos(phi_i) + origin[2]

            # rotate in x y plane
            x_i_r = (np.cos(theta_r) * x_i) + (-np.sin(theta_r) * y_i)
            y_i_r = (np.sin(theta_r) * x_i) + (np.cos(theta_r) * y_i)

            x_i_r_t = x_i_r + origin[0]
            y_i_r_t = y_i_r + origin[1]

            x_i = r * np.cos(theta_i) * np.sin(phi_i) + origin[0]
            y_i = r * np.sin(theta_i) * np.sin(phi_i) + origin[1]
            z_i = r * np.cos(phi_i) + origin[2]

            x.append(x_i_r_t)
            y.append(y_i_r_t)
            z.append(z_i)

    points = [np.array([x[i], y[i], z[i]]) for i, x_i in enumerate(x)]
    points = np.unique(points, axis=0)

    return points

def hemisphere_surface(r, origin, theta_r=0):
    theta = np.linspace(0.5 * np.pi, 3 / 2 * np.pi, 30)
    # sphere
    #theta = np.linspace(0, 2 * np.pi, 20)
    phi = np.linspace(0, np.pi, 30)

    theta_i, phi_i = np.meshgrid(theta, phi)

    x = r * np.cos(theta_i) * np.sin(phi_i)
    y = r * np.sin(theta_i) * np.sin(phi_i)
    z = r * np.cos(phi_i) + origin[2]

    x_r = (np.cos(theta_r) * x) + (-np.sin(theta_r) * y)
    y_r = (np.sin(theta_r) * x) + (np.cos(theta_r) * y)

    x_r_t = x_r + origin[0]
    y_r_t = y_r + origin[1]

    return x_r_t, y_r_t, z
