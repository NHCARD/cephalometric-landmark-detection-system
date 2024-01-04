import os

import numpy as np
import matplotlib.pyplot as plt
import json

h, w = 2400, 2880
std = 10

heatmap = np.zeros((h, w))
coordinateMtx = np.zeros((h, w, 2))
coordinateMtx[:, :, 0] = np.tile(np.arange(h).reshape(h, 1), (1, w))
coordinateMtx[:, :, 1] = np.tile(np.arange(w), (h, 1))


def Laplace(center, sigmaD=None):  # Gaussian Filter
    center_x, center_y = center
    center_mtx = np.ones((h, w, 2))
    center_mtx[:, :, 0] *= center_y
    center_mtx[:, :, 1] *= center_x

    distance = np.sum(np.abs(coordinateMtx - center_mtx), axis=2)

    SDMap = (1 / (2 * sigmaD)) * np.exp(-np.abs(distance) / (2 * sigmaD) ** 2)
    SDMap = SDMap / np.max(SDMap)

    return SDMap


if __name__ == "__main__":
    with open('../CL-Data/train-gt.json') as file:
        json_data = json.load(file)
        points = json_data.get('points')

    for i in range(1, 39):
        os.makedirs(f'./heatmap/1{i + 1:0>2d}', exist_ok=True)

    for i in range(0, 100):
        print(i)
        for j in range(38):
            sdmap = Laplace((points[i * 38 + j]['point'][:2]), sigmaD=std)
            Heatmap = np.maximum(sdmap, heatmap)
            plt.imsave(f'./heatmap/1{j+1:0>2d}/{i+1:0>3d}.png', Heatmap * 255, cmap='gray')
