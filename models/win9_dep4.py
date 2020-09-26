from keras.layers import Conv2D,BatchNormalization
from keras.models import Sequential


# n = 9, kernel 3X3, 4 layers
def create_network(inputs, input_shape, scope="win9_dep4"):
	num_maps = 64
	kw = 3
	kh = 3

	net=Conv2D(num_maps, (kw, kw), input_shape=input_shape, padding='valid', activation='relu')(inputs)
	net=BatchNormalization()(net)

	net=Conv2D(num_maps, (kw, kw), padding='valid', activation='relu')(net)
	net=BatchNormalization()(net)

	net=Conv2D(num_maps, (kw, kw), padding='valid', activation='relu')(net)
	net=BatchNormalization()(net)

	net=Conv2D(num_maps, (kw, kw), padding='valid')(net)
	net=BatchNormalization()(net)

	return net



