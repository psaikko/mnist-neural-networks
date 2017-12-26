import numpy

class Sigmoid:
	def value(self, X):
		return 1.0 / (1.0 + numpy.exp(-X))

	def slope(self, values):
		return values * (1 - values)

class SoftMax:
	def value(self, X):
		row_max = numpy.reshape(numpy.max(X, axis=1), (X.shape[0], 1))
		# avoid overflow
		exps = numpy.exp(X - numpy.repeat(row_max, X.shape[1], axis=1))
		row_sums = numpy.reshape(numpy.sum(exps, axis=1), (X.shape[0], 1))
		# create row-wise probability distributions
		return exps / numpy.repeat(row_sums, X.shape[1], axis=1)

	def slope(self, values):
		return values * (1 - values)

class ReLU:
	def value(self, X):
		return X * (X > 0)

	def slope(self, values):
		return 1 * (values > 0)

class LeakyReLU:
	def __init__(self, leak):
		self.leak = leak

	def value(self, X):
		# add nonzero slope to ReLU for x < 0
		return (X * (X > 0)) + (self.leak * (X * (X <= 0)))

	def slope(self, values):
		return (1 * (values > 0)) + (self.leak * (values <= 0))