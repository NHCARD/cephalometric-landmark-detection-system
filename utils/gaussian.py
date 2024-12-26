import numpy as np
import matplotlib.pyplot as plt

h, w = 2400, 2880
heatmap = np.zeros((h, w))
coordinateMtx = np.zeros((h, w, 2))

coordinateMtx[:, :, 0] = np.tile(np.arange(h).reshape(h, 1), (1, w))
coordinateMtx[:, :, 1] = np.tile(np.arange(w), (h, 1))


def Gaussian(center, sigma):
    center_x, center_y = center
    center_mtx = np.ones((h, w, 2))
    center_mtx[:, :, 0] *= center_y
    center_mtx[:, :, 1] *= center_x

    distance = np.sqrt(np.sum((coordinateMtx - center_mtx) ** 2, axis=2))

    heatmap = np.exp(-(distance ** 2) / (2 * sigma ** 2))
    heatmap = heatmap / np.max(heatmap)

    return heatmap


center = (1200, 1440)
sigma = 300
gaussian_heatmap = Gaussian(center, sigma)


plt.imshow(gaussian_heatmap, cmap='jet')
plt.colorbar()
plt.title("Gaussian Heatmap")
plt.show()