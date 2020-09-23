import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

from android_studio_logs import read_data
from colormaps import cmaps
import colormaps
import os
from pathlib import Path

if __name__ == '__main__':

    #import numpy
    #from mpl_toolkits.mplot3d import proj3d
    #def orthogonal_proj(zfront, zback):
    #    a = (zfront+zback)/(zfront-zback)
    #    b = -2*(zfront*zback)/(zfront-zback)
    #    return numpy.array([[1,0,0,0],
    #                        [0,1,0,0],
    #                        [0,0,a,b],
    #                        [0,0,0,zback]])
    #proj3d.persp_transformation = orthogonal_proj

    # r = 4.5

    #fp_68 = 'test_68.txt'
    #data_set_name_68 = 'test_68'
    #data_68, t_68, dt_68 = read_data(fp_68, 6)
    #t_rel_68 = t_68 - t_68[0]

    fps = ['fg_0.txt', 'fg_1.txt', 'fg_2.txt', 'fg_3.txt']

    data = [read_data(fp, 6) for fp in fps]

    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 9}

    matplotlib.rc('font', **font)

    fig = plt.figure(1, constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')

    skip = 1

    # motion data
    for i, item in enumerate(data):
        d, _, _ = item
        cs = np.arange(0, d.shape[0])
        print(d.shape[0])
        ax.scatter(d[:, 10][::skip], d[:, 9][::skip], -d[:, 11][::skip], marker='o', c=cs, cmap=colormaps.cmaps[i], label="Simulation p{}".format(i+1), depthshade=False)

        # trajectory data
        ax.plot3D(d[:, 1], d[:, 0], -d[:, 2], linewidth=0.5, color='blue', label="p{} FG Trajectory".format(i+1))

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')

    ax.legend()

    #ax.tick_params(axis='y', which='major', labelsize=5)

    plt.gcf().set_size_inches(4.98, 4.98 / 1.2)

    fn_base = 'target_left_up'
    Path(fn_base + '_plots').mkdir(parents=True, exist_ok=True)
    fp = Path(fn_base + '_plots', "fn_base.png")
    plt.savefig(fp, dpi=330)
    plt.show()
