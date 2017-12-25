import numpy
import idx2numpy
from matplotlib import pyplot

test_x = idx2numpy.convert_from_file('data/t10k-images-idx3-ubyte')
test_y = idx2numpy.convert_from_file('data/t10k-labels-idx1-ubyte')

print(test_x.shape)
print(test_y.shape)

N = len(test_y)

show_idx = numpy.random.randint(N, size=(16,))

pyplot.figure(1)
#pyplot.subplots_adjust(top=1.5)
for i, idx in enumerate(show_idx):
	pyplot.subplot(4,4,i+1)
	pyplot.imshow(test_x[idx])
	pyplot.axis('off')
	pyplot.title("%d" % test_y[idx])
pyplot.show()
