# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np
import hnn

def plot(data):
	for d in data:
		img = d.reshape(5,5)
		plt.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
		plt.show()
		plt.clf()

def putNoise(data, noise):
	ret = []
	for d in data:
		ret.append(_putNoise(d, noise))
	return ret

def _putNoise(img, noise):
	ret = 2*np.copy(img)-1
	mask = np.random.rand(len(img)) > noise
	ret = ret*mask - ret*np.logical_not(mask)
	return ret

data = [np.array([
				0,1,0,1,0,
				0,1,0,1,0,
				0,0,0,0,0,
				1,0,0,0,1,
				0,1,1,1,0
				]),
		np.array([
				1,0,0,0,1,
				0,1,1,1,0,
				0,1,0,1,0,
				0,1,1,1,0,
				1,0,0,0,1
				]),
		np.array([
				0,0,1,0,0,
				0,1,0,1,0,
				0,1,1,1,0,
				0,1,0,1,0,
				0,1,0,1,0
				])
		]

network = hnn.HNN()
network.train(data)

dataWithNoise = putNoise(data, 0.25)

network.predict(dataWithNoise)

for i in range(len(data)):
	plot([network.imageHistory[i][0],network.imageHistory[i][6]])

