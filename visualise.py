# Create the data.
import numpy as np

v = np.load('q/target_right_up_obs_box.npy')

#v = np.zeros((5, 5, 5))

#v[2, 2, 2] = 10

# View it.
import matplotlib.pyplot as plt

fig, axs = plt.subplots(4, 6, figsize=(3, 5))

for i, ax in enumerate(axs.flat):
    try:
        f = ax.imshow(v[:, :, i], origin='lower')
        plt.colorbar(f, ax=ax)
        ax.set_title('z {}'.format(i))
    except IndexError:
        pass
plt.show()