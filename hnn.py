# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np

class HNN(object):
	def __init__(self):
		self.W = np.array([])
		self.dim = 0
		self.Q = 0 # number of patterns
		self.threshold = 0
		self.loop = 7
		self.energyHistory = []
		self.imageHistory = []
		
	def train(self, trainingData):
		self.dim = len(trainingData[0])
		self.Q = len(trainingData)

		self.W = np.zeros((self.dim, self.dim))
		for i in range(self.Q):
			d = 2 * trainingData[i] - 1 # value: 0 → -1, 1 → 1 
			d = d.reshape((-1,1))
			self.W += np.dot(d, d.T)
		for i in range(self.dim):
			self.W[i,i] = 0 # 対角成分は0
		self.W /= self.Q

	def predict(self, inputData):
		for x in inputData:
			energyLog = []
			imageLog = []
			imageLog.append(x)
			x = 2*x-1
			V = self.potentialEnergy(x)
			energyLog.append(V)
			for i in range(self.loop):
				x = np.sign(np.dot(self.W, x) - self.threshold) # sgn(u-θ)
				V = self.potentialEnergy(x)
				imageLog.append((x+1)/2)
				energyLog.append(V)
			self.imageHistory.append(imageLog)
			self.energyHistory.append(energyLog)

	# Lyapunov function
	def potentialEnergy(self, x):
		x = x.reshape((-1,1))
		ret = -np.dot(np.dot(x.T, self.W), x)/2 + np.sum(self.threshold * x) # -(xT W x)/2 + Σ(θ xi)
		return ret