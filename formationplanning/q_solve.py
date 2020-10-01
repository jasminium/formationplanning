import numpy as np
import time

eps = 1e-6

def q_solve(v, dl):


    # positions
    r = np.zeros((*v.shape, 3))

    # number of charges
    n_q = v.size

    # calculate the positions on the grid
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            for k in range(v.shape[2]):
                # position
                rijk = np.array((i * dl, j * dl, k * dl))
                r[i, j, k] = rijk

    # write as matrix equation V = v
    # Aq = v. see notes
    vf = v.flatten()
    rf = r.reshape((vf.size, 3))

    r_m = np.stack([rf]*n_q, axis=1)
    q_m = np.stack([rf]*n_q, axis=0)

    a = r_m - q_m
    a = np.linalg.norm(a, axis=2)
    a = 1 / (a + eps)

    t0 = time.perf_counter()
    r = np.linalg.lstsq(a, vf, rcond='warn')
    t1 = time.perf_counter()
    print('cpu linear solve time: ', t1 - t0)

    q = r[0]

    # charge solution to inverse problem
    q_im = q.reshape(v.shape)

    return q_im, a


def v_r(o_x, o_y, o_z, x, y, z):
    # o is the origin
    # x coordinate
    # y coordinate

    o = np.array((o_x, o_y, o_z))
    r = np.array((x, y, z))

    return 1 / (np.linalg.norm(r - o) + eps)

if __name__ == "__main__":
    
    # construct a potential function
    
    # simulate 20 x 20 x 20 m domain
    nx = 21
    ny = 21
    nz = 21
    dl = 1

    # potential field
    v = np.zeros((nx, nx, nz))
    h = np.array((20, 20, 20))
    print('origin', h)

    # calculate the potential on the grid
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            for k in range(v.shape[2]):
                # potential
                vijk = v_r(h[0] * dl, h[1] * dl, h[2] * dl,  i * dl, j * dl, k * dl)
                # position
                v[i, j, k] = vijk

    #nx = 20
    
    #v = np.zeros((nx, nx, nx))

    # uniform plate
    v[10, 5:15, 5:15] = -1

    #v[, :, :] = np.random.rand() * 10
    #v[1, :, :] = np.random.rand() * 10
    #v[2, :, :] = np.random.rand() * 10
    #v[3, :, :] = np.random.rand() * 10

    q, a = q_solve(v, dl)

    np.save('20x20x20_t_plate_test', q)
    np.save('20x20x20_t_plate_test_a', a)

    v_f = a.dot(q.flatten())
    v_f = v_f.reshape(v.shape)

    slice_index = 2
    import matplotlib.pyplot as plt
    fig = plt.figure(2)
    plt.title('Potential')
    plt.imshow(v[:,:, slice_index], cmap='coolwarm', origin='lower')
    plt.colorbar()

    fig = plt.figure(3)
    plt.imshow(q[:, :, slice_index], cmap='coolwarm', origin='lower')
    plt.colorbar()
    plt.title('inverse q')

    fig = plt.figure(4)
    plt.imshow(v_f[:, :, slice_index], cmap='coolwarm', origin='lower')
    plt.colorbar()
    plt.title('Potential inversion')

    fig = plt.figure(5)
    plt.imshow((v_f - v)[:, :, slice_index], cmap='coolwarm', origin='lower')
    plt.colorbar()
    plt.title('Potential diff')

    plt.show()
