import os

import numpy as np
import matplotlib.pyplot as plt
import json

h = 2400
w = 2880
std = 10


heatmap = np.zeros((h, w))
coordinateMtx = np.zeros((h, w, 2))
coordinateMtx[:, :, 0] = np.tile(np.arange(h).reshape(h, 1), (1, w))
coordinateMtx[:, :, 1] = np.tile(np.arange(w), (h, 1))


def Laplace(center, sigmaD=None, __coordinateMtx=None):  # Gaussian Filter
    centerX = center[0]
    centerY = center[1]
    centerMtx = np.ones((h, w, 2))
    centerMtx[:, :, 0] *= centerY
    centerMtx[:, :, 1] *= centerX

    dist = np.sum(np.abs(coordinateMtx - centerMtx), axis=2)

    SDMap = (1 / (2 * sigmaD)) * np.exp(-np.abs(dist) / (2 * sigmaD) ** 2)
    SDMap = SDMap / np.max(SDMap)

    return SDMap


file = open(r'./train-gt.json')
jsonstring = json.load(file)
points = jsonstring.get('points')

sdmap = Laplace((points[0]['point'][:2]), sigmaD=std, __coordinateMtx=coordinateMtx, img=heatmap)
Heatmap = np.maximum(sdmap, heatmap)
plt.imsave(f'./test_std10.png', Heatmap * 255, cmap='gray')

if __name__ == "__main__":
    for i in range(1, 39):
        try:
            os.mkdir(f'./heatmap/1{i + 1:0>2d}')
        except:
            pass

    for i in range(0, 100):
        print(i)
        for j in range(38):
            sdmap = Laplace((points[(i * 38) + j]['point'][:2]), sigmaD=std, __coordinateMtx=coordinateMtx, img=heatmap)
            Heatmap = np.maximum(sdmap, heatmap)
            plt.imsave(f'./heatmap/1{j+1:0>2d}/{i+1:0>3d}.png', Heatmap * 255, cmap='gray')
