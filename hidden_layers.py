import numpy
import idx2numpy
import matplotlib
matplotlib.use("TkAgg") # doesn't steal focus on update

from matplotlib import pyplot
from activation import *
import time

def loss(Y_pred, Y_true):
	diff = numpy.reshape(Y_true - Y_pred, (Y_true.shape[0] * Y_true.shape[1]))
	return sum(diff * diff) / len(diff)

def to_onehot(Y):
	return numpy.eye(n_cats)[Y]

def correct(Y_true, Y_pred):
	return numpy.all((1.0 * (Y_pred > 0.5)) == Y_true, axis=1)

def forward_propagate(X_in, W, activations):
	layer_output = []
	res = X_in
	for (w, f) in zip(W, activations):
		res = numpy.copy(f.value(numpy.dot(res, w.T)))
		layer_output += [res]
	return layer_output

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
flattened_test_x = (numpy.reshape(test_x, (N_test, img_size)) / 256)
flattened_train_x = (numpy.reshape(train_x, (N_train, img_size)) / 256)

# convert labels to one-hot vectors
onehot_test_y = to_onehot(test_y)
onehot_train_y = to_onehot(train_y)

layer_sizes = [img_size, 30, 30, 30, n_cats]

activations = [Sigmoid(), Sigmoid(), Sigmoid(), SoftMax()]

# initialize random weights with mean 0
W = [2.0 * numpy.random.rand(layer_sizes[i+1], layer_sizes[i]) - 1.0
     for i in range(len(layer_sizes) - 1) ]

# iterations of gradient descent
epochs = 1000

# gradient descent step size
eps = 0.0001

#pyplot.ion()
fig = pyplot.figure(1)
axes = fig.add_subplot(1,1,1)
#pyplot.axis([0,epochs-1,0,1])

losses = []
accs = []

loss_line, acc_line = pyplot.plot([],[],[])
pyplot.legend((loss_line, acc_line), ("loss", "accuracy"))

opt_start = time.time()

for i in range(epochs):
	epoch_start = time.time()
	deltas = []

	# compute output of each layer
	layer_outputs = forward_propagate(flattened_train_x, W, activations)
	pred_y = layer_outputs[-1]

	# count correct predictions
	result = correct(onehot_train_y, pred_y)

	# compute loss and accuracy
	acc = sum(result) / N_train
	lss = loss(pred_y, onehot_train_y)

	# plot loss and accuracy in realtime
	losses = numpy.append(loss_line.get_ydata(), [lss])
	loss_line.set_ydata(losses)
	accs = numpy.append(acc_line.get_ydata(), [acc])
	acc_line.set_ydata(accs)

	loss_line.set_xdata(numpy.append(loss_line.get_xdata(), [i]))
	acc_line.set_xdata(numpy.append(acc_line.get_xdata(), [i]))

	axes.relim()
	axes.autoscale_view(True,True,True)

	pyplot.pause(0.001)

	err = pred_y - onehot_train_y

	# backpropagation
	layer_inputs = [flattened_train_x] + layer_outputs[:-1]
	for (w,f,layer_input,layer_output) in reversed(list(zip(W, activations, layer_inputs, layer_outputs))):
		# error on previous layer * derivative of activation (sigmoid)
		gradient = err * f.slope(layer_output) # layer_output * (1.0 - layer_output)

		# apply gradient with step size to next layer output
		delta = -eps * numpy.dot(gradient.T, layer_input)

		# update weights after backpropagation finished
		deltas += [delta]

		# compute error on this layer weights
		err = numpy.dot(gradient, w)

	# update weight matrix
	for (w, delta) in zip(reversed(W), deltas):
		w += delta

	epoch_time = time.time() - epoch_start
	elapsed_time = time.time() - opt_start
	print("Epoch %-5d Accuracy %.4f Loss %.4f Epoch %.2fs Total %.2fs" % (i, acc, lss, epoch_time, elapsed_time))

# check predictions on test data
predictions = forward_propagate(flattened_test_x, W, activations)[-1]
correct_predictions = correct(onehot_test_y, predictions)

num_correct = sum(correct_predictions)
print(num_correct)
print("%d%% correct" % (num_correct * 100.0 / N_test))

vis_weights = [[W[0][i] for i in range(W[0].shape[0])]]

for (layer_size, layer_index) in zip(layer_sizes[2:], range(2,len(layer_sizes))):
	layer_weights = []

	for i in range(layer_size):
		node_w = W[layer_index-1][i]

		prev_wgts = []
		for j in range(len(node_w)):
			prev_wgts += [node_w[j] * vis_weights[-1][j]]

		prev_wgts = sum(prev_wgts) / len(node_w)
		layer_weights += [prev_wgts]

	vis_weights += [layer_weights]

sub_h = len(vis_weights)
sub_w = max(layer_sizes[1:])

pyplot.figure(2)
for layer in range(sub_h):
	for weights in range(len(vis_weights[layer])):
		pyplot.subplot(sub_h, sub_w, 1 + layer*sub_w + weights)
		pyplot.imshow(numpy.reshape(vis_weights[layer][weights], (img_height, img_width)))
		pyplot.axis('off')

# keep plots visible
pyplot.show()