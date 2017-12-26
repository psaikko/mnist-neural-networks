import numpy
import idx2numpy
from matplotlib import pyplot
from activation import *

def sigmoid(X):
	return 1.0 / (1.0 + numpy.exp(-X))

def loss(Y_pred, Y_true):
	diff = numpy.reshape(Y_true - Y_pred, (Y_true.shape[0] * Y_true.shape[1]))
	return sum(diff * diff) / len(diff)

def to_onehot(Y):
	return numpy.eye(n_cats)[Y]

def correct(Y_true, Y_pred):
	return numpy.all((1.0 * (Y_pred > 0.5)) == Y_true, axis=1)

# read data from file
test_x = idx2numpy.convert_from_file('data/t10k-images-idx3-ubyte')
test_y = idx2numpy.convert_from_file('data/t10k-labels-idx1-ubyte')
train_x = idx2numpy.convert_from_file('data/train-images-idx3-ubyte')
train_y = idx2numpy.convert_from_file('data/train-labels-idx1-ubyte')

# find some dimensions
n_cats = 10
img_width = test_x.shape[1]
img_height = test_x.shape[2]
img_size = img_width * img_height
N_train = train_y.shape[0]
N_test  = test_y.shape[0]

# flatten 2D images and scale to [0..1]
flattened_test_x = numpy.reshape(test_x, (N_test, img_size)) / 256
flattened_train_x = numpy.reshape(train_x, (N_train, img_size)) / 256

# convert labels to one-hot vectors
onehot_test_y = to_onehot(test_y)
onehot_train_y = to_onehot(train_y)

# initialize random weights with mean 0
W = 2 * numpy.random.rand(n_cats, img_width*img_height) - 1

# track development of matrix weights
history_slices = 10
W_history = []

# iterations of gradient descent
epochs = 100

# gradient descent speed
eps = 0.0001

# choose activation function
activation = SoftMax()

for i in range(epochs):
	# sigmoid activation of matrix product
	pred_y = activation.value(numpy.dot(flattened_train_x, W.T))

	# count correct predictions
	result = correct(onehot_train_y, pred_y)

	print("Epoch %-5d Correct %d/%d\tLoss %.4f" %
		(i, sum(result), N_train, loss(pred_y, onehot_train_y)))

	# compute gradients from activation function slope
	gradient = (pred_y - onehot_train_y) * activation.slope(pred_y)

	# apply gradient to weight matrix
	delta = -eps * numpy.dot(gradient.T, flattened_train_x)

	# update weight matrix
	W += delta

	# record weight development
	if i % (epochs // history_slices) == 0:
		W_history += [numpy.copy(numpy.reshape(W, (n_cats, img_height, img_width)))]

# check predictions on test data
correct_predictions = correct(onehot_test_y, activation.value(numpy.dot(flattened_test_x, W.T)))

num_correct = sum(correct_predictions)

print("%d%% correct" % (num_correct * 100.0 / N_test))

# plot weight history
pyplot.figure(1)
for j in range(history_slices):
	for i in range(n_cats):
		pyplot.subplot(history_slices, n_cats, j*history_slices + (i+1))
		pyplot.imshow(W_history[j][i])
		pyplot.axis('off')
pyplot.show()